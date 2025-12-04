# LightHunter – 系统架构总览（Mk4 版）

> 目标：A 股超短线选股 · 科研级框架 · 可持续进化  
> 版本：Mk4 · 包含多源质量评估、防封策略、实验矩阵、U1/U2/U3 主战策略等

---

## 0. 系统定位与设计哲学

- **方向**：A 股超短线（1–3 日持股为主，兼容日内/隔日博弈）
- **诉求**
  1. **数据第一性**：多源抓取（东财/同花顺/备用源），追求准确性、完整性、连续性；
  2. **方法论可实验**：因子/标签/模型/策略全链路可配置、可回测、可对比；
  3. **工程可长期运行**：有防封策略、风控、监控、Chaos 测试、影子券商预演；
  4. **科研可复现**：ExperimentLab + PerformanceLab + 实验矩阵/Walk‑Forward。

---

## 1. 顶层架构分层

### 1.1 配置与中枢

- `config/system_config.json`
  - 统一定义：`paths` / `network` / `event_bus` / `storage` / `data_sources` /  
    `data_quality` / `broker` / `trade_core` / `portfolio` / `risk` / `alpha` / `hud` / `ops` / `jobs` 等。:contentReference[oaicite:0]{index=0}
- `config/config_center.py`
  - 提供 `get_system_config()` 等方法，内部带缓存，统一读取/覆盖配置。
- `config/jobs.yaml`
  - 采集任务矩阵 & NightOps/DayOps/回归测试 等调度配置（Mk4 新增/强化）。
- `config/experiment_matrix.yaml`
  - 实验矩阵配置（不同任务/模型/因子子集）的描述文件。

> 这一层的职责：**“一句话说清楚系统是谁、在哪里、要做什么”**，所有模块只从这里拿配置。

---

### 1.2 网络 & 防封：RequestEngine + ProxyManager

- `network/proxy_manager.py`
  - 负责维护代理池（直连 / V2rayN HTTP / SOCKS 等），根据 `system_config.network.proxies` 管理。:contentReference[oaicite:1]{index=1}
  - 支持按域名路由：`network.routes`（如 `eastmoney.com` → `v2rayn_http`）。
- `request_engine.py`
  - 统一 HTTP 客户端封装：
    - **异步内核**：基于 `core/async_http.py`；
    - **UA 池 & TLS 指纹模拟**（Mk2 / Mk3 阶段引入）；
    - **站点级限流**：`network.rate_limits.per_site`（max_qps / max_concurrency / timeout）；
    - **熔断器**：`network.circuit_breakers`（error_ratio_threshold / open_state_sec 等）；
    - 支持错误统计、重试、随机 UA、代理切换。

> 作用：让东财/同花顺等**以为你是正常浏览器 + 合理人类**，并在封禁风险前自动收缩/切换。

---

### 1.3 事件总线：ZeroMQ Event Bus

- `bus/event_schema.py`
  - 定义统一事件模型：`market.tick` / `ts.snapshot` / `alpha.signal` / `trade.order` /  
    `trade.execution` / `risk.alert` / `metrics.*` 等。
- `bus/zmq_bus.py`
  - 使用 ZeroMQ 实现发布/订阅（PUB/SUB），从 `system_config.event_bus.zmq` 读取端口配置。:contentReference[oaicite:2]{index=2}
  - 提供 `publish(topic, payload)` / `subscribe(topic, handler)` 等接口。

> 整个系统通过 event bus **解耦**：采集、特征、策略、风控、监控、HUD 不再直接互相调用，而是通过 Topic 通信。

---

### 1.4 数据采集层：Collectors & TS 管线

- 核心目标：多源 L1 快照 + 衍生 L2/订单流信息 + 概念/舆情。
- 主要组件：
  - `ts_collector.py` / `market_ts_collector.py`
    - 对接东财/同花顺等站点，按 `jobs.yaml` 中任务矩阵进行周期采集；
    - 使用 RequestEngine + ProxyManager，遵守站点级 QPS / 并发/熔断策略；
    - 把结果写入 `TSStorage`，并通过 ZeroMQ 发布 `market.tick` / `ts.snapshot`。
  - `collectors/sentiment_scraper.py`
    - 抓取舆情/新闻/论坛文本，为情绪因子提供原材料。
  - `tools/replay_and_healthcheck.py`
    - 对指定区间/标的进行历史回放，检查采集完整性与延迟（Mk4 新增）。

---

### 1.5 存储与时序引擎：TSStorage + DuckDB

- `storage/duckdb_client.py`
  - 封装 DuckDB 连接与读写操作（time‑series 优化，Parquet 读写）。:contentReference[oaicite:3]{index=3}
- `storage/ts_storage.py`
  - **双写层**：统一入口写入 SQLite + DuckDB/Parquet，或仅 DuckDB；
  - 提供高层 API：写入分钟线、日线、L1/L2 snapshot、订单流重构结果。
- `ts_recorder.py` / `ts_data_pipeline.py`
  - 负责把 collector 的原始数据整理成规范化 TS 表结构，如 `multi_source_daily`。
- `core/data_bus.py`
  - （如有）在进程内提供零拷贝的 dataframe 传递。

---

### 1.6 数据质量与体检：DataGuardian Ω

- `tools/data_source_quality_checker.py`
  - 基于 `data_quality` 配置，对多源字段（如 close_price/chg_pct/volume/turnover）进行一致性统计、有效性规则校验，生成日报。:contentReference[oaicite:4]{index=4}
- `tools/replay_and_healthcheck.py`
  - 做覆盖率/连续性/延迟等健康检查。
- `data_guardian.py`
  - **Ω 级调优版本**：汇总多源对比、回放检查结果和数据异常分布，输出“数据健康评分报告”（按表/字段/日期分层）。
- `tools/factor_label_report.py`
  - 对因子与标签进行 IC/RankIC/行情分段分析，检测因子有效性与标签稳定性。

---

### 1.7 因子 / 标签 / 数据集

- 特征与因子：
  - `features/order_flow_reconstructor.py`：从 L1 快照重构订单流（主动买/卖、盘口压力等）；
  - `features/order_flow_engine.py`：构造订单流相关因子（主动买卖力度、短期 order imbalance 等）；
  - `features/concept_engine.py` / `features/concept_graph_builder.py` / `features/concept_graph_features.py`：概念图谱与图特征；
  - `features/sentiment_engine.py`：舆情/情绪特征。
- 因子引擎：
  - `factor_engine.py`
    - 统一读取 TS + 订单流 + 概念 + 情绪 + 微观结构（如有）并生成 `factor_panel`；
    - Mk3/Mk4 阶段重点收敛出 **A 股超短主战因子集合**。
- 标签体系：
  - `labels/label_center.py` + `config/label_spec.json`
    - 定义 T+1/T+2/T+3 收益、最高/最低回撤、连板阶段、情绪周期等多种标签；
    - 保证所有训练/回测共享同一标签定义。
- 数据集构建：
  - `alpha/dataset_builder.py`
    - 根据配置构建 **多任务数据集**：
      - 超短主任务（U1/U2/U3）；
      - GNN 图任务；
      - 微观结构任务等。

---

### 1.8 模型与训练流水线

- 模型动物园：
  - `alpha/model_zoo.py`
    - 注册多类模型：
      - 基线线性/树模型；
      - 深度时序模型（TCN/LSTM/Transformer 等）；
      - GNN 模型（基于概念/连板图）；
      - 多任务模型（共享 backbone，分任务 head）。
- 训练管线：
  - `alpha/training_pipelines.py` / `alpha/gnn_training_pipeline.py`
    - 通过 `(job_id, model_id, task_type)` 启动训练；
    - 自动读取 `DatasetBuilder` 输出、从 ModelZoo 取模型、写入 ExperimentLab。
- 实验与绩效：
  - `lab/experiment_lab.py`：记录所有训练/回测/仿真实验（参数、版本、指标、路径等）；
  - `lab/performance_lab.py`：统一计算 Sharpe、MaxDD、胜率、因子 IC 等指标，并写回 ExperimentLab。

---

### 1.9 回测 / 执行 / 风控 / 账户

- 回测：
  - `backtest_core.py`
    - 支持常规回测与 **Walk‑Forward 滚动回测**；
    - 与 PerformanceLab 联动，输出按年份/阶段的绩效统计。
- 执行：
  - `execution/execution_model.py`
    - 建模滑点、成交概率、撮合延迟等；
  - Broker 抽象与实现：
    - `broker/broker_api.py`
      - `BaseBroker` 抽象；
      - `PaperBroker`：多账户纸上交易撮合；
      - `ShadowBroker`：影子券商（模拟延迟/部分成交/拒单/风控）。
- 交易核心：
  - `trade_core.py`
    - 收到 `trade.order` 事件 → 调用 Broker → 发出 `trade.execution`；
    - 维护订单状态，并向 EventBus/HUD 推送账户 NAV/持仓等。:contentReference[oaicite:6]{index=6}
- 策略与组合：
  - `strategy/portfolio_manager.py`
    - 维护 U1/U2/U3 等策略的资金权重、多账户配置；
  - `commander.py`
    - 订阅 `alpha.signal` & 行情 → 按 `strategy_id` 路由到 PortfolioManager → 发单给 TradeCore；
    - 连接 RiskBrain 做前置风控。

---

### 1.10 风险与监控

- 风险脑：
  - `risk_brain.py` + `risk/microstructure_engine.py`
    - 订阅行情/订单流/成交事件，计算账户级/标的级综合风险评分；
    - 输出 `risk.alert` 事件，被 HUD / MonitorDaemon / Commander 共同消费。
- 监控守护进程：
  - `ops/monitor_daemon.py`
    - 订阅 `metrics.latency.*` / `metrics.error.*` / `metrics.bus.queue` / `risk.alert` 等 Topic，统计：
      - 数据延迟（EWMA / max）；
      - 错误率；
      - 队列堆积；
      - 账户风险概要；
    - 周期写入 `metrics.jsonl`，并输出日志摘要。
- HUD：
  - `hud/web_server.py` + `hud/templates/*.html`
    - Flask Web 服务，展示：
      - 账户净值/回撤/仓位；
      - 风险告警流；
      - 最近实验结果；
- Chaos 测试：
  - `tools/chaos_injector.py`
    - 注入网络故障、延迟、总线堆积、Broker 异常等场景，验证系统韧性。

---

### 1.11 运维 & Ops

- 夜间流水线：
  - `ops/night_ops.py`
    - 从采集/体检/修复 → 因子/标签 → 数据集 → 训练 → 回测 → 报告 一条龙，并可集成 walk‑forward/实验矩阵子任务。
- 盘前 Ops：
  - `ops/day_ops.py`
    - 检查昨夜夜盘结果 & 当日运行前体检（数据健康、模型版本、风险阈值等）。
- 调度中心：
  - `ops/scheduler.py`
    - 读取 `config/jobs.yaml` 或 `system_config.jobs`，按 cron 风格触发 NightOps / DayOps / 回归测试 / 实验矩阵等。

---

## 2. 关键数据流总结

### 2.1 采集 → TS → 因子/标签 → 模型 → 策略 → 成交路径

1. **采集**
   - `ts_collector` / `market_ts_collector` 按任务矩阵抓取 L1/L2/行情/舆情；
   - 通过 RequestEngine + ProxyManager 访问东财/同花顺/备用源。
2. **时序入库**
   - 经 `TSStorage` 写入 DuckDB/Parquet + SQLite；
   - 对一些关键表维护多源字段。
3. **质量体检**
   - `data_source_quality_checker` + `replay_and_healthcheck` + `data_guardian`；
   - 生成数据健康报告。
4. **因子与标签**
   - `factor_engine` 产出主战因子；
   - `label_center` 按规范生成标签。
5. **数据集与训练**
   - `DatasetBuilder` 构建不同任务数据集（超短/GNN/微观）；
   - `training_pipelines` & `gnn_training_pipeline` 训练模型，写入 ExperimentLab。
6. **回测与评估**
   - `backtest_core` 执行普通 & Walk‑Forward 回测；
   - `performance_lab` 输出统一绩效指标；HUD 展示部分结果。
7. **实时/仿真执行**
   - 策略引擎 → Commander → PortfolioManager → TradeCore + Broker；
   - RiskBrain & MonitorDaemon & HUD 实时守护。

---

## 3. Mk4 阶段新增能力一览

- **多源字段质量体系**：`data_quality` 配置 + `tools/data_source_quality_checker.py`
- **防封策略落地**：ProxyManager + RequestEngine 站点级 QPS/并发/熔断 + V2rayN
- **采集任务矩阵**：`config/jobs.yaml` + Scheduler/NightOps 整合
- **回放与健康检查**：`tools/replay_and_healthcheck.py`
- **因子/标签体检**：`tools/factor_label_report.py`
- **实验矩阵**：`config/experiment_matrix.yaml` + `tools/run_experiment_matrix.py`
- **U1/U2/U3 主战策略**：文档化 + PortfolioManager/Commander 接入
- **Walk‑Forward 回测**：`tools/run_walkforward_backtests.py`
- **两周战役计划 & 战役监控视图**：MonitorDaemon + HUD 扩展指标
- **Chaos 测试与 ShadowBroker**：故障注入 + 影子券商预演

> 这一版架构已经具备“机构级原型系统”的完整形态，后续更多是**打磨与扩展**，而不是“推倒重建”。
