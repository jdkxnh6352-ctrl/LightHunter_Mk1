# -*- coding: utf-8 -*-
"""
examples/run_intraday_simulation.py

LightHunter Mk3 - 日内仿真回放示例
==================================

功能：
------
- 从 Parquet 文件加载某天的多股票日内数据
- 按时间顺序回放，向 ZeroMQ 总线发送：
    1) "market.snapshot" 行情快照事件（原始 dict）
    2) alpha.signal 信号事件（SignalEvent，通过 event_schema 生成）
- 供以下组件实时订阅：
    - commander.py / OnlineListener / TradeCore：订阅 alpha.signal → 下单 → 成交
    - hud/web_server.py：订阅 snapshot/signal/order/execution，展示 Web HUD
    - tools/lighthunter_dashboard.py：命令行 Dashboard 展示系统状态

数据要求（Parquet）：
--------------------
至少包含以下列：
    - symbol : 股票代码，如 "000001.SZ"
    - ts     : 时间戳（字符串或 pandas 可识别的 datetime）
    - open, high, low, close, volume
其他列会被忽略或放到 meta 里。

用法示例：
----------
1) 启动 ZMQ 总线相关进程：
    # 1. 启动 commander（策略 + 下单链路）
    python -m commander

    # 2. 启动 Web HUD
    python -m hud.web_server --host 0.0.0.0 --port 8000

    # 3. 启动 CLI Dashboard
    python -m tools.lighthunter_dashboard

2) 启动日内仿真回放：
    python -m examples.run_intraday_simulation \\
        --parquet data/replay/2024-01-02.parquet \\
        --symbols 000001.SZ,300750.SZ \\
        --strategy-id ultrashort_sim \\
        --speed 60

"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.config_center import get_system_config
from core.logging_utils import get_logger
from bus.zmq_bus import get_zmq_bus
from bus.event_schema import SignalEvent

log = get_logger(__name__)


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------


def load_intraday_parquet(path: Path, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {path}")

    df = pd.read_parquet(path)
    if "ts" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Parquet 中必须包含列: 'ts' 和 'symbol'。")

    if symbols:
        df = df[df["symbol"].isin(symbols)]

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["ts", "symbol"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 信号生成（示例规则）
# ----------------------------------------------------------------------


def generate_signal_from_row(
    row: pd.Series,
    strategy_id: str,
) -> Optional[SignalEvent]:
    """
    一个非常简单的示例规则：
    - 若 close > open 且 当日该 symbol 的 volume 在 50% 分位以上 → 生成 BUY 信号
    - 若 close < open 且 volume 高于阈值 → 生成 SELL 信号
    - 否则不发信号（返回 None）

    实战中你会用训练好的模型来产出 score/方向，这里只是打通链路。
    """
    o = float(row.get("open", 0.0))
    c = float(row.get("close", 0.0))
    v = float(row.get("volume", 0.0))

    if o <= 0:
        return None

    ret = (c / o) - 1.0
    # 简单放大收益率作为 score
    score = ret * 100.0

    # 仅在波动略显著时发信号
    if abs(ret) < 0.001:  # 小于 0.1% 的波动不发
        return None

    direction = "BUY" if ret > 0 else "SELL"

    evt = SignalEvent(
        symbol=row["symbol"],
        ts=row["ts"].isoformat(),
        direction=direction,
        score=score,
        weight=1.0,
        strategy_id=strategy_id,
        model_id="intraday_rule_v1",
        job_id=None,
        horizon="intraday",
        extra={
            "open": o,
            "close": c,
            "high": float(row.get("high", c)),
            "low": float(row.get("low", c)),
            "volume": v,
        },
    )
    return evt


# ----------------------------------------------------------------------
# 回放主逻辑
# ----------------------------------------------------------------------


def run_intraday_replay(
    parquet_path: Path,
    symbols: Optional[List[str]],
    strategy_id: str,
    speed: float,
) -> None:
    cfg = get_system_config()
    bus = get_zmq_bus(cfg)
    if bus is None:
        raise SystemExit("ZMQBus 不可用，请检查 config.system_config.event_bus 或 pyzmq 是否安装。")

    df = load_intraday_parquet(parquet_path, symbols)
    if df.empty:
        log.warning("回放数据为空，退出。")
        return

    log.info(
        "IntradayReplay: 加载数据完成 rows=%d symbols=%s 第一条时间=%s",
        len(df),
        sorted(df["symbol"].unique().tolist()),
        df["ts"].iloc[0],
    )

    prev_ts: Optional[datetime] = None
    total_rows = len(df)

    for idx, row in df.iterrows():
        ts: datetime = row["ts"].to_pydatetime()

        # 控制回放节奏（按真实时间差 / speed）
        if prev_ts is not None:
            delta_real = (ts - prev_ts).total_seconds()
            if delta_real > 0 and speed > 0:
                time.sleep(delta_real / speed)
        prev_ts = ts

        # 1) 发布 market.snapshot（原始 dict）
        snapshot_payload: Dict[str, any] = {
            "symbol": row["symbol"],
            "ts": ts.isoformat(),
            "open": float(row.get("open", 0.0)),
            "high": float(row.get("high", 0.0)),
            "low": float(row.get("low", 0.0)),
            "close": float(row.get("close", 0.0)),
            "volume": float(row.get("volume", 0.0)),
        }
        bus.publish_raw("market.snapshot", snapshot_payload)

        # 2) 生成并发布 alpha.signal（示例规则）
        sig = generate_signal_from_row(row, strategy_id=strategy_id)
        if sig is not None:
            bus.publish_event(sig)

        if (idx + 1) % 1000 == 0:
            log.info(
                "IntradayReplay: 已回放 %d / %d 行，当前时间=%s",
                idx + 1,
                total_rows,
                ts,
            )

    log.info("IntradayReplay: 回放结束，合计行数=%d。", total_rows)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightHunter Mk3 - 日内仿真回放示例"
    )
    parser.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="日内数据 Parquet 文件路径",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="逗号分隔的 symbol 列表，例如 000001.SZ,300750.SZ；为空表示全部。",
    )
    parser.add_argument(
        "--strategy-id",
        type=str,
        default="ultrashort_sim",
        help="信号中使用的 strategy_id 标识。",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="回放加速倍数，例如 60 表示 1 分钟数据 1 秒播完。",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    parquet_path = Path(args.parquet).expanduser().resolve()
    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )
    run_intraday_replay(
        parquet_path=parquet_path,
        symbols=symbols,
        strategy_id=args.strategy_id,
        speed=args.speed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
