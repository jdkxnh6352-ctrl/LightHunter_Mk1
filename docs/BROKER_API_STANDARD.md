# LightHunter Mk3 - RealBroker 接口标准

> 目标：  
> 定义一套统一的 **实盘券商 Broker 接口标准**，让 `TradeCore / Commander / PortfolioManager`
> 只依赖抽象接口 `BaseBroker`，而不关心具体券商、柜台、协议，确保 **PaperBroker → RealBroker**
> 可以无缝替换。

---

## 1. 范围与定位

### 1.1 当前已有结构（broker/broker_api.py）

在现有工程中，`broker/broker_api.py` 里已经定义了：

- `BaseBroker`：抽象基类（逻辑标准）
- `PaperBroker`：本地撮合 / 仿真实现
- `RealBroker`：预留的实盘接入类（目前可为空壳）

本标准文档约定：

- **对下**：RealBroker 如何对接券商/柜台（东方财富、证券 API、期货 CTA 等）。  
- **对上**：RealBroker 必须对 `TradeCore`、`Commander` 提供与 `PaperBroker` 完全一致的行为表现。

---

## 2. 核心概念与数据结构

### 2.1 订单请求 `OrderReq`

用于发单的输入结构（可以是 `dataclass` / `pydantic` / 普通 dict，逻辑上包含这些字段）：

```python
OrderReq = {
  "client_order_id": str,       # 系统内唯一的本地订单 ID（由 TradeCore 生成）
  "account_id": str,            # 账户 ID（映射到券商账号）
  "symbol": str,                # 标的，如 "600000.SH"
  "side": "BUY" | "SELL",       # 买卖方向
  "order_type": "LIMIT" | "MARKET" | "FOK" | "FAK" | ...,
  "price": float | None,        # 限价单必填，市价单可为 None
  "quantity": int,              # 手数/股数（考虑 A 股 100 股一手）
  "time_in_force": "DAY" | "IOC" | "FOK" | ...,
  "strategy_id": str,           # 策略 ID（如 "U1", "U2"）
  "extra": dict                 # 预留扩展字段，如 algo 参数、冰山单等
}
