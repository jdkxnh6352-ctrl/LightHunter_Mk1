# -*- coding: utf-8 -*-
"""
test_small_capture.py

小规模抓数测试：
- 只抓 5 只示例股票
- 只用腾讯（主力源）+ 东方财富（探针）
- 连续跑几轮，每轮间隔 10 秒
"""

import time
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd

from request_engine import get
from config.config_center import get_system_config


SAMPLE_CODES: List[str] = ["000001", "600000", "300750", "002230", "600519"]


def _with_exchange(code: str) -> str:
    """给股票代码加上 sh/sz 前缀，喂给腾讯接口用。"""
    code = str(code)
    if code.startswith("6"):
        return "sh" + code
    return "sz" + code


def fetch_from_tencent(codes: List[str]) -> pd.DataFrame:
    """
    从腾讯一次性抓一批代码的快照，并解析出：
    code / name / price / pct
    """
    q = ",".join(_with_exchange(c) for c in codes)
    url = f"https://qt.gtimg.cn/q={q}"

    resp = get(url, site="tencent", timeout=5)
    txt = resp.text

    rows = []
    for line in txt.splitlines():
        if "v_" not in line or "=" not in line:
            continue
        try:
            _, right = line.split("=", 1)
            right = right.strip().strip('";')
            parts = right.split("~")

            # 这一套 index 是根据腾讯返回格式手工对过的
            name = parts[1]
            code = parts[2]
            last = float(parts[3] or 0.0)         # 最新价
            prev_close = float(parts[4] or 0.0)   # 昨收
            if prev_close > 0:
                pct = (last / prev_close - 1.0) * 100
            else:
                pct = 0.0

            rows.append(
                {
                    "code": code,
                    "name": name,
                    "price": last,
                    "pct": pct,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def fetch_from_eastmoney(code: str) -> Dict[str, Any]:
    """
    从东方财富抓单只股票，主要用来验证：
    - 能否正常连通
    - JSON 能否解析
    这里字段先简单取，后面接入正式多源引擎再精细化。
    """
    base_url = "https://push2.eastmoney.com/api/qt/stock/get"
    secid = ("1." if code.startswith("6") else "0.") + code
    params = {
        "secid": secid,
        # 字段列表先写一个比较宽松的集合，实际能拿到哪些以后再精修
        "fields": "f2,f3,f4,f5,f6,f7,f8,f12,f14,f15,f16,f17,f18,f20,f30,f31,f32,f33,f34,f35,f37,f38,f39,f43,f57,f58",
    }

    resp = get(base_url, site="eastmoney", params=params, timeout=5)

    result: Dict[str, Any] = {
        "code": code,
        "status": resp.status_code,
        "name": None,
        "price": None,
        "pct": None,
        "raw_len": len(resp.text),
    }

    try:
        j = resp.json()
    except Exception:
        # 只要 status=200 且 raw_len>0，说明至少没被直接屏蔽
        return result

    data = j.get("data") or {}
    # 常见字段：f58 = 名称，f2 = 现价，f3 = 涨跌幅（百分比）
    result["name"] = data.get("f58") or data.get("name")
    result["price"] = data.get("f2")
    result["pct"] = data.get("f3")

    return result


def main() -> None:
    cfg = get_system_config()
    net = cfg.get("network", {}) or {}

    print("当前网络 / 代理配置快照：")
    print("proxies =", net.get("proxies"))
    print("sites   =", net.get("sites"))
    print("示例股票池 =", SAMPLE_CODES)
    print("\n== 准备开始小规模抓数（共 3 轮，每轮间隔 10 秒） ==\n")

    rounds = 3
    interval_sec = 10

    for r in range(1, rounds + 1):
        print("=" * 70)
        print(f"第 {r} 轮采样 @ {datetime.now().strftime('%H:%M:%S')}")

        # 1）腾讯：一口气抓 5 只
        try:
            df_tx = fetch_from_tencent(SAMPLE_CODES)
            print("\n[腾讯行情] 返回 %d 行：" % len(df_tx))
            if not df_tx.empty:
                # 为了终端好看一点
                print(df_tx.to_string(index=False, justify='left', col_space=8))
            else:
                print("（空表，说明解析失败或返回为空）")
        except Exception as e:
            print("[腾讯行情] 抓取失败：", repr(e))

        # 2）东财：只抓 600000 一只，做“探针”
        try:
            em = fetch_from_eastmoney("600000")
            print("\n[东财探针] 返回：", em)
        except Exception as e:
            print("[东财探针] 请求失败：", repr(e))

        if r < rounds:
            print(f"\n等待 {interval_sec} 秒后继续下一轮...\n")
            time.sleep(interval_sec)

    print("\n== 小规模抓数测试结束 ==\n")


if __name__ == "__main__":
    main()
