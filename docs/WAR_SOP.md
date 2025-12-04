# LightHunter 战备 SOP（Mk4 版）

> 面向：自己 + 少数合作者  
> 目标：在 **A 股超短线研究 & 仿真实战** 中，有一套可重复、可执行、不会忘步骤的操作手册。

---

## 0. 概念约定

- **NightOps**：夜间科研流水线（体检/修复/因子/标签/训练/回测）。
- **DayOps**：盘前运行状况检查。
- **Scheduler**：任务调度器（根据 jobs.yaml 自动触发 NightOps/DayOps/回归测试等）。
- **ExperimentLab**：记录所有实验与回测。
- **PerformanceLab**：统一计算绩效指标。
- **HUD**：Web 实时看板。
- **MonitorDaemon**：监控守护进程。
- **PaperBroker / ShadowBroker**：纸上和影子券商。

---

## 1. 日常启停顺序

### 1.1 首次部署 / 环境冷启动

1. 安装依赖 & 初始化目录：
   - `pip install -r requirements.txt`
   - 创建 `data/`, `logs/`, `experiments/`, `monitor/`, `reports/` 等目录（如未自动创建）。
2. 配置检查：
   - 打开 `config/system_config.json`，确认：
     - `paths.*` 路径存在；
     - `duckdb_path`、`parquet_dir` 可写；
     - `network.proxies` / `routes` 根据本机 V2rayN 情况设置；
     - `event_bus.zmq.*` 端口号不冲突。:contentReference[oaicite:9]{index=9}
3. 快速自检：
   - 运行：`python -m tools.regression_runner --mode quick`
   - 确认所有基础模块加载正常（配置、DuckDB、ZeroMQ、ExperimentLab、Broker 等）。

### 1.2 每日标准流程（研究/仿真日）

**前一晚（或清晨 Linux 定时）—— NightOps**

1. 确认 Scheduler 正常运行（如使用 `ops/scheduler.py`）；
2. 自动/手动触发：
   - `python -m ops.night_ops --mode full`
3. NightOps 完成后：
   - 检查 `logs/` 与 `reports/` 下的 NightOps 报告；
   - 若失败，第二天盘前重点排查。

**盘前 08:30–09:15 —— DayOps**

1. 启动 EventBus（通常 ZMQ Bus 由各模块自动初始化）；
2. 启动 MonitorDaemon：
   - `python -m ops.monitor_daemon`
3. 启动 HUD：
   - `python -m hud.web_server`
   - 浏览器访问 `http://127.0.0.1:8000`，确认页面加载成功。
4. 运行：
   - `python -m ops.day_ops`
   - 核查 DayOps 报告中的：
     - 昨夜 NightOps 是否成功；
     - 数据健康评分（DataGuardian Ω）；
     - 训练/回测是否产出最新模型；
     - 关键风险参数（RiskBrain 阈值）。

**盘中（仿真阶段）**

1. 启动采集与策略仿真：
   - 采集：按 jobs.yaml 中配置，由 Scheduler 触发或手动运行 collector；
   - 策略链路：启动 Commander / TradeCore / PortfolioManager 相关脚本。
2. 观察 HUD + Monitor：
   - HUD：关注账户净值曲线变化、风险告警、实验状态；
   - MonitorDaemon：关注延迟/错误率/队列堆积/风险简报。

**盘后**

1. 回看当天日志与报告；
2. 若有新的实验矩阵/Walk‑Forward 需要执行，可在盘后夜间触发；
3. 保证第二天 DayOps 前 NightOps 已完成。

---

## 2. 手动操作手册

### 2.1 NightOps 手动全流程

```bash
python -m ops.night_ops --mode full \
  --start-date 2024-01-01 --end-date 2024-03-31 \
  --tasks ts_check,repair,factor,label,train,backtest
