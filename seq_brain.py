# -*- coding: utf-8 -*-
"""
模块名称：SequenceBrain Mk-TS
版本：Mk-TS Brain R60 (Multi-Horizon Strike Predictor)
路径: G:/LightHunter_Mk1/seq_brain.py

设计思路（R60 多任务版）：
- 训练阶段（离线，盘后）：
    * 从 ts_data.db 中读取分时快照 snapshots 表；
    * 将每只股票的分时序列切成「历史窗口 → 未来窗口」；
    * 同一套历史特征，针对多个未来窗口（例如 10 / 20 / 30 分钟）分别打标签：
        - 在 horizon 步内最高涨幅 - 当前涨幅 >= gain_threshold_pct → 记为强攻击样本(label=1)，否则 0；
    * 如果安装了 sklearn：
        - 共享一个 StandardScaler，对所有任务统一归一化；
        - 为每个 horizon 训练一套 LogisticRegression（二分类多任务：共享特征，独立头部）；
      否则：
        - 退化为“统计阈值 + 启发式”模式，并为每个 horizon 记录一套 slope/volume 阈值。
- 预测阶段（盘中，实时）：
    * 使用 Commander 累积的 price_history_map（最近 N 次涨幅序列）+ 当前成交额 / 换手率；
    * 计算与训练阶段一致的特征；
    * 输出多 horizon 的 0~1 概率：
        - predict_from_history(...)：单一 horizon（兼容老接口，默认用训练时的第一个 horizon）；
        - predict_multi_from_history(...)：多任务版，返回 {horizon: {代码: 概率}}。
"""

import os
import sqlite3
import random
import datetime
from typing import Dict, List, Optional, Iterable

import numpy as np
import pandas as pd
from colorama import Fore, Style

DB_FILE = "ts_data.db"

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False


class SequenceBrain:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path

        # 多任务：horizon_step -> 模型
        # horizon_step 是“步数”，例如快照间隔约 1~1.5 分钟时：
        #   6 步 ≈ 10 分钟，12 步 ≈ 20 分钟 ...
        self.models: Dict[int, "LogisticRegression"] = {}
        self.scaler: Optional["StandardScaler"] = None
        self.horizons: List[int] = []  # 训练时实际使用的 horizon 步数列表
        self.is_trained: bool = False

        # 启发式阈值（没有 sklearn 或样本不足时使用）
        # 每个 horizon 一套阈值，同时保留一个“默认值”兼容旧逻辑
        self.slope_thresholds: Dict[int, float] = {}
        self.vol_thresholds: Dict[int, float] = {}
        self.slope_threshold: float = 0.0
        self.vol_threshold: float = 0.0

    # --------------------------------------------------
    # 日志工具
    # --------------------------------------------------
    def _log(self, msg: str, color=Fore.CYAN):
        print(color + "[SEQ] " + msg + Style.RESET_ALL)

    # --------------------------------------------------
    # 训练：从 ts_data.db 学习多 horizon 的攻击模式
    # --------------------------------------------------
    def train_from_db(
        self,
        lookback: int = 12,
        horizon: int = 6,
        horizons: Optional[Iterable[int]] = None,
        gain_threshold_pct: float = 3.0,
        max_rows: int = 200_000,
        min_samples: int = 2_000,
        neg_keep_prob: float = 0.2,
        max_samples_per_task: int = 50_000,
    ):
        """
        :param lookback:      用多少条历史分时点作为输入 (12 条 ≈ 2~3 分钟，视采样间隔而定)
        :param horizon:       兼容旧接口的单一 horizon（步数），当 horizons 未指定时使用
        :param horizons:      可选，多任务 horizon 列表（例如 [6, 12, 18]）
        :param gain_threshold_pct: 未来窗口内最高涨幅比当前涨幅再拉 >= 该值(%) 视为强攻击
        :param max_rows:      从数据库最多读取多少行（控制内存）
        :param min_samples:   每个 horizon 至少需要多少有效训练样本，否则降级为启发式模式
        :param neg_keep_prob: 负样本保留概率，用于简单的样本均衡
        :param max_samples_per_task: 每个 horizon 最多保留多少训练样本
        """
        # 兼容：如果外部只传了 horizon，则自动封装为列表
        if horizons is None:
            horizons = [horizon]
        else:
            # 清洗：只保留正整数并去重
            _tmp = []
            for h in horizons:
                try:
                    h_int = int(h)
                    if h_int > 0 and h_int not in _tmp:
                        _tmp.append(h_int)
                except Exception:
                    continue
            horizons = _tmp or [horizon]

        horizons = sorted(horizons)
        self.horizons = horizons

        if not os.path.exists(self.db_path):
            self._log(
                f"DB not found: {self.db_path}，使用启发式时序信号。",
                Fore.YELLOW,
            )
            return

        try:
            conn = sqlite3.connect(self.db_path)
            # 取最近 max_rows 条，防止整个库太大
            query = f"""
                SELECT ts, code, price, pct, amount, turnover_rate
                FROM snapshots
                ORDER BY ts DESC
                LIMIT {max_rows}
            """
            raw_df = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            self._log(f"读取 ts_data.db 失败: {e}", Fore.RED)
            return

        if raw_df.empty or len(raw_df) < (lookback + max(horizons) + 10):
            self._log(
                "ts_data 数据量太少，暂不训练 SequenceBrain。",
                Fore.YELLOW,
            )
            return

        # 排序：按 code + ts 从早到晚
        raw_df["ts"] = pd.to_datetime(raw_df["ts"])
        raw_df = raw_df.sort_values(["code", "ts"]).reset_index(drop=True)

        # 为每个 horizon 准备容器
        X_dict: Dict[int, List[List[float]]] = {h: [] for h in horizons}
        y_dict: Dict[int, List[int]] = {h: [] for h in horizons}
        pos_slopes: Dict[int, List[float]] = {h: [] for h in horizons}
        pos_vols: Dict[int, List[float]] = {h: [] for h in horizons}

        n_all = 0

        grouped = raw_df.groupby("code")
        max_h = max(horizons)

        for code, g in grouped:
            if len(g) < lookback + max_h + 5:
                continue
            g = g.reset_index(drop=True)

            pct_arr = g["pct"].astype(float).values
            amt_arr = g["amount"].astype(float).values
            to_arr = g["turnover_rate"].astype(float).values

            # 滑动窗口：历史 lookback → 未来 max_h
            for i in range(lookback, len(g) - max_h):
                # 历史窗口
                hist_pct = pct_arr[i - lookback : i]
                hist_amt = amt_arr[i - lookback : i]
                hist_to = to_arr[i - lookback : i]

                x_idx = np.arange(len(hist_pct))
                try:
                    slope_pct = np.polyfit(x_idx, hist_pct, 1)[0]
                except Exception:
                    slope_pct = 0.0

                vol_factor = float(np.log1p(np.maximum(hist_amt, 0.0)).mean())
                mean_turn = float(np.nanmean(hist_to))
                momentum = float(hist_pct[-1] - hist_pct[0])
                last_pct = float(hist_pct[-1])

                feat = [
                    slope_pct,
                    mean_turn,
                    vol_factor,
                    momentum,
                    last_pct,
                ]

                # 每个 horizon 单独打标签
                for h in horizons:
                    # 不足 h 步的未来，跳过
                    j_end = i + h
                    if j_end >= len(g):
                        continue

                    if len(X_dict[h]) >= max_samples_per_task:
                        continue

                    fut_pct = pct_arr[i:j_end]
                    future_max = float(fut_pct.max())
                    delta = future_max - last_pct
                    label = 1 if delta >= gain_threshold_pct else 0

                    # 简单负样本下采样
                    if label == 0 and random.random() > neg_keep_prob:
                        continue

                    X_dict[h].append(feat)
                    y_dict[h].append(label)
                    n_all += 1

                    if label == 1:
                        pos_slopes[h].append(slope_pct)
                        pos_vols[h].append(vol_factor)

        # 没有任何样本
        if n_all == 0:
            self._log("未构造出任何训练样本，保持启发式模式。", Fore.YELLOW)
            return

        # 统计阈值：即使有 sklearn，也记录一下方便启发式降级
        for h in horizons:
            if pos_slopes[h]:
                self.slope_thresholds[h] = float(np.mean(pos_slopes[h]))
            if pos_vols[h]:
                self.vol_thresholds[h] = float(np.mean(pos_vols[h]))

        # 默认阈值选用第一个 horizon
        first_h = horizons[0]
        self.slope_threshold = self.slope_thresholds.get(first_h, 0.0)
        self.vol_threshold = self.vol_thresholds.get(first_h, 0.0)

        # 如果没有 sklearn，或者所有任务样本量都太小，则只用启发式
        if (not SK_AVAILABLE) or all(
            len(X_dict[h]) < min_samples for h in horizons
        ):
            self.is_trained = True
            if not SK_AVAILABLE:
                self._log(
                    "未安装 sklearn，SequenceBrain 以启发式多任务模式运行。",
                    Fore.YELLOW,
                )
            else:
                self._log(
                    f"每个 horizon 有效样本均不足 {min_samples}，采用启发式阈值模式。",
                    Fore.YELLOW,
                )
            return

        # 汇总所有样本做统一 StandardScaler
        X_all_list: List[List[float]] = []
        for h in horizons:
            X_all_list.extend(X_dict[h])
        X_all = np.array(X_all_list, dtype=float)
        scaler = StandardScaler()
        scaler.fit(X_all)

        models: Dict[int, LogisticRegression] = {}
        trained_tasks = 0

        for h in horizons:
            Xh_list = X_dict[h]
            yh_list = y_dict[h]
            if len(Xh_list) < min_samples:
                self._log(
                    f"horizon={h} 样本不足 ({len(Xh_list)} < {min_samples})，该任务降级为启发式。",
                    Fore.YELLOW,
                )
                continue

            Xh = np.array(Xh_list, dtype=float)
            yh = np.array(yh_list, dtype=int)

            try:
                Xh_scaled = scaler.transform(Xh)
                clf = LogisticRegression(
                    max_iter=400,
                    n_jobs=-1,
                )
                clf.fit(Xh_scaled, yh)
                models[h] = clf
                trained_tasks += 1
                pos_rate = float(yh.mean()) * 100.0
                self._log(
                    f"horizon={h} 训练完成: 样本={len(yh)}, 阳样本比例={pos_rate:.1f}%",
                    Fore.GREEN,
                )
            except Exception as e:
                self._log(
                    f"horizon={h} 训练失败，将对该任务降级为启发式: {e}",
                    Fore.RED,
                )

        # 记录模型
        self.models = models
        self.scaler = scaler if trained_tasks > 0 else None
        self.is_trained = True

        if trained_tasks == 0:
            self._log(
                "所有 horizon 的模型训练均失败或样本不足，SequenceBrain 将退化为纯启发式模式。",
                Fore.RED,
            )
        else:
            self._log(
                f"SequenceBrain 多任务训练完成，成功训练 {trained_tasks} 个 horizon。",
                Fore.GREEN,
            )

    # --------------------------------------------------
    # 内部：单个 horizon 的预测（统一特征）
    # --------------------------------------------------
    def _predict_one_horizon(
        self,
        feat: np.ndarray,
        horizon: int,
        mean_turn: float,
        momentum: float,
        last_pct: float,
    ) -> float:
        """
        给定单只股票的特征向量 + 当下统计，返回某个 horizon 的攻击概率。
        feat 形状: (1, 5) -> [slope_pct, mean_turn, vol_factor, momentum, last_pct]
        """
        slope_pct = float(feat[0, 0])
        vol_factor = float(feat[0, 2])

        # 1) 优先使用 sklearn 模型（如果该 horizon 的模型存在）
        if (
            SK_AVAILABLE
            and self.scaler is not None
            and horizon in self.models
        ):
            try:
                X_scaled = self.scaler.transform(feat)
                proba = float(
                    self.models[horizon].predict_proba(X_scaled)[0, 1]
                )
                return proba
            except Exception:
                # 模型异常时降级为阈值模式
                pass

        # 2) 统计阈值启发式（训练阶段至少跑过一遍）
        if self.is_trained:
            slope_thr = self.slope_thresholds.get(
                horizon, self.slope_threshold
            )
            vol_thr = self.vol_thresholds.get(horizon, self.vol_threshold)

            score = 0.5
            if slope_pct > slope_thr:
                score += 0.2
            if vol_factor > vol_thr:
                score += 0.2
            if momentum > 1.5:
                score += 0.1
            if last_pct < -2.0:
                score -= 0.1
            return float(max(0.0, min(1.0, score)))

        # 3) 完全启发式（甚至没跑过 train_from_db）
        base = 0.4
        base += 0.15 * np.tanh(slope_pct * 8.0)
        base += 0.05 * np.tanh(mean_turn / 10.0)
        base += 0.05 * np.tanh(momentum / 2.0)
        return float(np.clip(base, 0.0, 1.0))

    # --------------------------------------------------
    # 预测：单 horizon 兼容版（默认取训练时第一个 horizon）
    # --------------------------------------------------
    def predict_from_history(
        self,
        price_history_map: dict,
        market_df: pd.DataFrame,
        lookback: int = 12,
        target_horizon: Optional[int] = None,
    ) -> dict:
        """
        兼容旧接口：返回 {code: prob}，只针对一个 horizon 预测。
        :param price_history_map: Commander 中维护的 {code: deque([最近若干次 涨幅%])}
        :param market_df: 当前这一帧的全市场 snapshot DataFrame
        :param lookback: 至少取多少个历史点（不足则用当前涨幅填充）
        :param target_horizon: 指定 horizon（步数），如果为 None 则使用训练时的第一个 horizon，
                               若仍为空则默认使用 6.
        """
        result = {}
        if market_df is None or market_df.empty:
            return result

        # 决定目标 horizon
        if target_horizon is None:
            if self.horizons:
                target_horizon = self.horizons[0]
            else:
                target_horizon = 6

        for _, row in market_df.iterrows():
            code = row["代码"]
            # 1) 构造历史涨幅序列
            hist = list(price_history_map.get(code, []))
            cur_pct = float(row.get("涨幅", 0.0))

            if len(hist) < lookback:
                need = lookback - len(hist)
                hist = [cur_pct] * need + hist
            else:
                hist = hist[-lookback:]

            pct_hist = np.array(hist, dtype=float)
            x_idx = np.arange(len(pct_hist))
            try:
                slope_pct = np.polyfit(x_idx, pct_hist, 1)[0]
            except Exception:
                slope_pct = 0.0

            momentum = float(pct_hist[-1] - pct_hist[0])
            last_pct = float(pct_hist[-1])
            mean_turn = float(row.get("换手率", 0.0))
            amount = float(row.get("成交额", 0.0))
            vol_factor = float(np.log1p(max(amount, 0.0)))

            feat = np.array(
                [[slope_pct, mean_turn, vol_factor, momentum, last_pct]],
                dtype=float,
            )

            proba = self._predict_one_horizon(
                feat, target_horizon, mean_turn, momentum, last_pct
            )
            result[code] = proba

        return result

    # --------------------------------------------------
    # 预测：多 horizon 多任务版
    # --------------------------------------------------
    def predict_multi_from_history(
        self,
        price_history_map: dict,
        market_df: pd.DataFrame,
        lookback: int = 12,
        horizons: Optional[Iterable[int]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        多任务预测接口：返回 {horizon_step: {代码: 概率}}。

        :param price_history_map: Commander 中维护的 {code: deque([最近若干次 涨幅%])}
        :param market_df: 当前这一帧的全市场 snapshot DataFrame
        :param lookback: 至少取多少个历史点（不足则用当前涨幅填充）
        :param horizons: 要预测的 horizon（步数）列表，None 时使用训练过的 self.horizons，
                         若仍为空则默认 [6]。
        """
        if market_df is None or market_df.empty:
            return {}

        if horizons is None:
            horizons = self.horizons or [6]
        else:
            _tmp = []
            for h in horizons:
                try:
                    h_int = int(h)
                    if h_int > 0 and h_int not in _tmp:
                        _tmp.append(h_int)
                except Exception:
                    continue
            horizons = _tmp or (self.horizons or [6])

        horizons = sorted(horizons)
        result: Dict[int, Dict[str, float]] = {h: {} for h in horizons}

        for _, row in market_df.iterrows():
            code = row["代码"]
            # 统一构造特征
            hist = list(price_history_map.get(code, []))
            cur_pct = float(row.get("涨幅", 0.0))

            if len(hist) < lookback:
                need = lookback - len(hist)
                hist = [cur_pct] * need + hist
            else:
                hist = hist[-lookback:]

            pct_hist = np.array(hist, dtype=float)
            x_idx = np.arange(len(pct_hist))
            try:
                slope_pct = np.polyfit(x_idx, pct_hist, 1)[0]
            except Exception:
                slope_pct = 0.0

            momentum = float(pct_hist[-1] - pct_hist[0])
            last_pct = float(pct_hist[-1])
            mean_turn = float(row.get("换手率", 0.0))
            amount = float(row.get("成交额", 0.0))
            vol_factor = float(np.log1p(max(amount, 0.0)))

            feat = np.array(
                [[slope_pct, mean_turn, vol_factor, momentum, last_pct]],
                dtype=float,
            )

            for h in horizons:
                proba = self._predict_one_horizon(
                    feat, h, mean_turn, momentum, last_pct
                )
                result[h][code] = proba

        return result


if __name__ == "__main__":
    brain = SequenceBrain()
    # 示例：默认单 horizon 训练（约 10 分钟）
    brain.train_from_db(lookback=12, horizon=6)
    # 或：多 horizon 训练，例如 10m / 20m / 30m
    # brain.train_from_db(lookback=12, horizons=[6, 12, 18])
