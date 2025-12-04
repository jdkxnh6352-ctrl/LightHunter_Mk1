# LightHunter Mk3 最终战备 Checklist

> 目标：在系统进入“长期连续运行”（科研 + 日内仿真 / 预备实盘）前，
> 用这份清单逐项验收，确保从 **环境、数据、因子、模型、风控、监控、运维** 到
> **未来接 RealBroker** 的各环节都在可控范围之内。

勾选方式建议：

- `[ ]` 未完成 / 未验证
- `[x]` 已完成且通过验收

---

## 1. 环境与基础设施

### 1.1 Python & 依赖环境

- [ ] Python 版本已固定（如 `3.10.x`），并记录在文档中  
  - 验收标准：`python --version` 输出与文档一致

- [ ] 已使用虚拟环境或 conda 环境隔离  
  - 验收标准：`which python` / `where python` 指向项目专用环境

- [ ] `pip install -r requirements.txt` 无错误完成  
  - 验收标准：重新安装到一台干净机器上仍能成功

- [ ] 所有关键三方库在版本上已锁定（`requirements.txt` 带具体版本号）  
  - 验收标准：requirements 中不存在裸包名（如 `pandas`），而是 `pandas==x.y.z`

### 1.2 配置中心 & 路径

- [ ] `config/system_config.json` 字段完整：`paths / network / event_bus / broker / portfolio / jobs / monitor / hud / lab` 等  
  - 验收标准：`python -m tools.regression_runner --quick` 中 `config_paths` 检查通过

- [ ] `config/config_center.py` 对系统路径（数据、因子、DuckDB、实验等）统一管理  
  - 验收标准：所有模块只通过 ConfigCenter 获取路径，不再散落硬编码路径

- [ ] 数据/日志/实验目录有充足磁盘空间（预估至少能支撑 3–6 个月）  
  - 验收标准：`df -h` 显示剩余空间 > 30%，并在文档中备注数据增长速度估计

---

## 2. 数据获取与网络层（天眼系统）

### 2.1 RequestEngine & 网络匿名性

- [ ] RequestEngine 已支持异步内核（`core/async_http.py` + `request_engine.py`）  
  - 验收标准：核心采集任务可以并发跑，且 CPU 使用率和 QPS 在可控范围内

- [ ] 已配置并验证 `ProxyManager` + V2rayN / VPN 代理  
  - 验收标准：  
    - 在不开代理时，东财/同花顺出现封 IP 现象  
    - 开启代理后，连续采集一段时间未出现封禁

- [ ] 请求头、UA 池、TLS 指纹（如使用 curl_cffi）等防爬参数已配置  
  - 验收标准：爬虫连续运行多天，目标站无“强制 302 到人机验证 / 验证码 / 频繁 403” 等问题

### 2.2 数据源与数据完整性

- [ ] 所有主要数据源已在 `config/system_config.json` 的 `data_sources` 段登记（同花顺、东方财富、其他备选源）  
  - 验收标准：`data/source_catalog.py` 能枚举出所有可用的数据源

- [ ] TSCollector / MarketTSCollector 对 A 股全市场股票（或选定子集）有定时完整采集任务  
  - 验收标准：  
    - 随机抽样若干股票，检查最近 N 个交易日分钟 K / Tick 是否连续  
    - DataGuardian 报告中“缺口率”低于设定阈值（如 < 0.1%）

- [ ] 对“被封 IP / 超时 / 5xx” 等异常有重试与熔断策略  
  - 验收标准：RequestEngine 的日志中有清晰的重试与熔断记录，无无限重试/风暴

---

## 3. 存储层（DuckDB / Parquet / TSStorage）

### 3.1 DuckDB & Parquet

- [ ] 已替换 SQLite 为 DuckDB + Parquet 作为主要时序存储  
  - 验收标准：所有历史查询/回测/特征计算的入口不再依赖 SQLite

- [ ] DuckDB 文件路径及表结构已稳定，不会频繁变化  
  - 验收标准：`storage/duckdb_client.py` 的 schema 在 docs 中有说明

- [ ] Parquet 分区（按日期 / 标的）策略已确定，并在文档里记录  
  - 验收标准：`data/factors/`、`data/datasets/` 等目录结构清晰无混乱

### 3.2 TSStorage 双写层

- [ ] `storage/ts_storage.py` 双写层已接入：  
  - 盘中实时写入 + 盘后持久化  
  - 验收标准：随机抽取一个交易日，对比内存数据与落地 Parquet / DuckDB，记录数一致

- [ ] `TSRecorder / MarketTSCollector / TSMinuteBus` 统一通过 TSStorage 读写  
  - 验收标准：不存在绕过 TSStorage 直接写文件的代码路径（已在 grep 中确认）

- [ ] 回归测试中 `duckdb_storage` 检查通过  
  - 验收标准：`python -m tools.regression_runner --quick` duckdb 项目 OK

---

## 4. 因子、标签与数据集（主战武器）

### 4.1 因子产出（FactorEngine）

- [ ] `factor_engine.py` 已统一产出“主战因子集合” + 扩展因子  
  - 验收标准：在文档中列出主战因子的清单（订单流、情绪、概念、微观结构、基本技术指标等）

- [ ] 因子文件格式统一（Parquet），且最近多日列集合一致  
  - 验收标准：  
    - `python -m tools.regression_runner` 中 `factor_storage_consistency` 为 OK  
    - `tests/test_factor_consistency.py` 全部通过

- [ ] 因子泄露检查工具已运行（`tools/factor_leakage_checker.py`）  
  - 验收标准：  
    - 对关键标签（隔日收益、3 日收益等）无明显泄露因子  
    - 对发现的可疑泄露因子已记入说明并剔除

### 4.2 标签体系（LabelCenter）

- [ ] `labels/label_center.py` + `config/label_spec.json` 定义了完整的超短线标签体系  
  - 如：隔日收益、T+1 gap、3 日最大回撤、情绪阶段等  
  - 验收标准：标签说明文档已写入（如 `docs/LABEL_SPEC.md`）

- [ ] `ts_labeler.py` 按 label_spec 产出标签，并支持多任务标签  
  - 验收标准：  
    - 随机抽样几只票，手算几日收益，对比标签值  
    - labels 中无大面积 NaN / Inf

### 4.3 DatasetBuilder & 多任务数据集

- [ ] `alpha/dataset_builder.py` 能根据配置构建：  
  - 超短主任务（股票-时间维度预判）  
  - GNN 任务（图结构输入）  
  - 微观结构任务（订单流 / L2 相关）  

- [ ] DatasetBuilder 构建一致性通过测试  
  - 验收标准：  
    - `python -m tools.regression_runner` 中 `dataset_builder_consistency` OK  
    - `tests/test_dataset_builder_consistency.py` 通过

---

## 5. 模型与训练流水线（ModelZoo / TrainingPipelines）

### 5.1 ModelZoo

- [ ] `alpha/model_zoo.py` 中已注册：  
  - 主战模型（如专门针对超短 T+1/T+3）  
  - 基线模型（如简单线性/树模型）  
  - GNN 模型  
  - 多任务模型（价格 + 情绪 + 微观任务）

- [ ] 每类模型在文档中有推荐配置（超参、输入因子集合）  
  - 验收标准：`docs/MODEL_ZOO.md` 或 ARCHITECTURE 文档中有清晰说明

### 5.2 TrainingPipelines & ExperimentLab

- [ ] `alpha/training_pipelines.py` 支持以 `job_id / model_id / task_type` 启动训练  
  - 验收标准：  
    - 可以通过一条命令（或 NightOps）启动一次完整训练  
    - TrainingPipelines 自动记录实验到 ExperimentLab

- [ ] `lab/experiment_lab.py` 与 `lab/performance_lab.py` 已联通  
  - 验收标准：  
    - 每次训练 / 回测结束，ExperimentLab 中有一条实验记录  
    - PerformanceLab 将 IC / RankIC / Sharpe / MaxDD 等指标写入实验记录

- [ ] `tests/test_backtest_stability.py` 通过（冒烟回测稳定）  
  - 验收标准：重复两次的指标相对误差 < 5%

---

## 6. 日常运维与 SOP（NightOps / DayOps / Scheduler）

### 6.1 NightOps（盘后科研流水线）

- [ ] `ops/night_ops.py` 已实现标准流程：  
  - 数据体检（DataGuardian）  
  - TS 数据修复（TSDataRepair）  
  - 特征/标签更新  
  - Dataset 构建  
  - 训练 + 回测  
  - 实验记录 + 绩效写入  

- [ ] NightOps 有清晰的运行报告（日志 + 简要汇总）  
  - 验收标准：NightOps 日志中有一个“NightOps SUMMARY”或类似总览

### 6.2 DayOps（盘前体检）

- [ ] `ops/day_ops.py` 已实现：  
  - 今日是否交易日判定  
  - 昨日数据完整性复查  
  - 模型文件存在性与读取测试  
  - 关键任务计划检查（今日 NightOps / 回归测试等）

- [ ] DayOps 体检结果有明确“OK / WARNING / FAIL”分级  
  - 验收标准：  
    - FAIL 时，盘中作战流程禁止自动启动（可配置）

### 6.3 Scheduler（任务调度）

- [ ] `ops/scheduler.py` 已配置：  
  - 夜间 NightOps  
  - 盘前 DayOps  
  - 回归测试 RegressionRunner  
  - （可选）其他日常任务

- [ ] `config.system_config.json` 中 `jobs` 段配置已生效  
  - 验收标准：Scheduler 启动后，在日志中能看到按预定时间触发任务

---

## 7. 监控、风控与 HUD（MonitorDaemon / RiskBrain / HUD）

### 7.1 MonitorDaemon

- [ ] `ops/monitor_daemon.py` 正常运行  
  - 订阅数据延迟、错误率、队列堆积、风险告警等  
  - 写入 metrics.jsonl 文件

- [ ] Monitor 日志中周期性输出 SUMMARY  
  - 验收标准：  
    - 延迟 / 错误 / 队列 / 风险 概要信息一目了然

### 7.2 RiskBrain & 微观结构引擎

- [ ] `risk_brain.py` 集成了：  
  - 订单流因子异常  
  - 微观结构异常引擎（例如孤立森林/规则）  
  - 账户级/标的级综合风险评分

- [ ] RiskBrain 通知机制打通（ZeroMQ + HUD）  
  - 验收标准：  
    - 风险事件能在 Web HUD & Dashboard 中正确显示  
    - 对高风险事件有明显的 visual 标记（level=warning/critical）

### 7.3 Web HUD / 命令行 Dashboard

- [ ] `hud/web_server.py` 提供：  
  - 账户净值 & 回撤 & 仓位  
  - 风险告警列表  
  - 最近实验结果  

- [ ] `tools/lighthunter_dashboard.py` 命令行 HUD 可用  
  - 验收标准：可以在终端中快速查看当日运行状态与关键指标

---

## 8. 事件总线 & 日内仿真（ZMQBus / Commander / TradeCore）

### 8.1 ZeroMQ 事件总线

- [ ] `bus/event_schema.py` 定义了标准事件 schema（行情、订单、成交、风险、监控等）  
- [ ] `bus/zmq_bus.py` 正常工作，关键 Topic 已定义清晰：

  - 行情类：`market.tick` / `collector.snapshot`  
  - 策略 & 订单：`alpha.signal` / `trade.order_update` / `trade.execution`  
  - 账户：`account.nav` / `account.position`  
  - 风险：`risk.alert`  
  - 监控：`metrics.bus.queue` / `metrics.error` 等  

- [ ] 事件总线健康检查通过（RegressionRunner 中 `event_bus` 项 OK）

### 8.2 Commander / TradeCore / PaperBroker

- [ ] `commander.py` 已按 Mk3 版本：  
  - 订阅行情 & 信号  
  - 将策略输出路由到不同 strategy_id → model_id → account_id  
  - 与 PortfolioManager 协作进行资金分配与风险限制

- [ ] `trade_core.py` 使用 Broker 抽象：  
  - PaperBroker 用于纸上交易 / 仿真  
  - 未来 RealBroker 将替换为实盘接口

- [ ] 简单日内仿真脚本可跑通：  
  - `examples/run_intraday_simulation.py`  
  - 验收标准：  
    - 能回放一段历史数据  
    - HUD & Dashboard 能实时展示账户与订单动态

---

## 9. RealBroker 预备与实盘安全阈值

> 虽然现在还没接实盘，但标准先定好，避免未来乱写。

- [ ] 已阅读并理解 `docs/BROKER_API_SPEC.md`（RealBroker 接口标准）  
- [ ] `broker/broker_api.py` 中 BaseBroker / PaperBroker 接口实现稳定  
  - 验收标准：所有调用 Broker 的地方只用 BaseBroker 定义的方法

- [ ] 在文档中写明：**从 PaperBroker 切换到 RealBroker 必须满足的前置条件**：  
  - 仿真环境连续跑 N 天无严重异常  
  - 回归测试每次更新后必跑且通过  
  - RiskBrain 阈值与策略最大回撤可控  
  - 账户/券商侧手工风控上限明确（最大持仓、单票上限等）

---

## 10. 连续运行与战备演练

### 10.1 连续稳定运行验证

- [ ] 至少进行一次 **7 天不间断仿真运行** 演练（NightOps + DayOps + 日内仿真全开）  
  - 验收标准：  
    - 无系统级崩溃  
    - MonitorDaemon 无长期高延迟/高错误率告警  
    - 数据完整率保持在预设水平

### 10.2 故障演练（Drill）

- [ ] 演练“盘中策略异常撤退”：  
  - 人工触发风险事件 / 模拟异常  
  - 验收标准：  
    - Commander 退场（停止交易）  
    - TradeCore 停止生成新订单  
    - HUD & Dashboard 中有明显状态变化

- [ ] 演练“网络中断 / 代理失效”：  
  - 短时间关闭代理或网络  
  - 验收标准：  
    - 系统自动检测、记录告警  
    - 无疯狂重试导致自我 DDOS 的情况  
    - 网络恢复后能正常继续运行或按 SOP 重启

### 10.3 文档与知识转移

- [ ] ARCHITECTURE 文档（系统架构）已完成并与现状一致  
- [ ] WAR SOP（战备操作手册）已完成，并能按照手册从零重启系统  
- [ ] 本 Checklist 已完成首轮勾选，并在重要更新后定期复查

---

> ✅ 当你能在这份清单上 **绝大多数项目打上 `[x]`**，  
> 且连续运行 1–2 轮完整“NightOps + DayOps + 日内仿真 + 回归测试”循环没有出现致命问题，  
> 就可以认为 **LightHunter Mk3 已具备“长期科研 + 准实盘战备”能力**，  
> 下一步就是——在极小仓位下 **谨慎上线 RealBroker**，让它真正变成 “能打仗、打胜仗” 的硬核超短武器。
