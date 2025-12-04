下面是 **Mk4‑Step‑24：《从空环境拉起 + 故障回滚演练手册》**（Markdown 版）。
你可以直接保存为 `docs/BOOTSTRAP_AND_ROLLBACK.md` 使用。

---

# LightHunter · 从空环境拉起 + 故障回滚演练手册

> 目标：
>
> * 从一台“完全干净”的机器，把 LightHunter 整个科研系统拉起来；
> * 跑通一轮完整的 **NightOps → DayOps → 回测/仿真**；
> * **人为制造一次失败**；
> * 再用 **Git + 配置 + 模型快照** 完成彻底回滚。
>
> 通过这套演练，验证系统已经具备“能打仗，打败仗，还能原地满血复活”的战备能力。

---

## 0. 前置假设与环境约定

* 操作系统：Linux / macOS / WSL 任意一种。
* Python：建议 `3.10+`（与你当前工程保持一致）。
* 依赖：

  * 已安装 `git`；
  * 已安装 `pip`，能访问 PyPI；
  * 机器上可安装 `virtualenv` 或使用 `python -m venv`。

工程结构假定为：

```text
LightHunter/
  config/
    system_config.json
    jobs.yaml
  ops/
    night_ops.py
    day_ops.py
    scheduler.py
    monitor_daemon.py
  hud/
    web_server.py
    templates/
  tools/
    regression_runner.py
    lighthunter_dashboard.py
  examples/
    run_end_to_end_backtest.py
    run_intraday_simulation.py
  broker/
    broker_api.py
  ...
```

系统配置位于 `config/system_config.json`，包含 `paths / network / event_bus / storage / broker / trade_core / portfolio / risk / alpha / hud / ops / jobs` 等段。

---

## 1. 从空环境拉起：Step‑by‑Step

### 1.1 克隆代码仓库

```bash
# 任选目录
cd ~/work

# 克隆你的仓库（以 github 为例，按你实际地址替换）
git clone git@github.com:YOUR_NAME/LightHunter.git
cd LightHunter
```

确认代码状态：

```bash
git status
# 预期：工作区干净（clean）
```

> **建议**：为本次演练创建专用分支，例如 `war-drill-YYYYMMDD`：

```bash
git checkout -b war-drill-2025xxxx
```

---

### 1.2 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 升级基础工具
pip install --upgrade pip setuptools wheel

# 安装依赖
pip install -r requirements.txt
```

> 如果你之前做过 Mk1–Mk4 的步骤，`requirements.txt` 已经包含：
>
> * web 抓取 & 异步：`aiohttp`, `requests`, `curl_cffi` 等
> * 数据处理：`pandas`, `numpy`, `duckdb`, `pyarrow`
> * ZeroMQ：`pyzmq`
> * Web HUD：`flask`
> * 测试 & 工具：`pytest` 等

---

### 1.3 初始化配置与目录

1. **准备 `config/system_config.json`**

   * 如果仓库中有 `config/system_config.example.json`，先复制一份：

     ```bash
     cp config/system_config.example.json config/system_config.json
     ```

   * 如果没有 example，使用当前你上传/维护的最新版 `config/system_config.json` 作为基准，并按本机实际路径检查这些字段是否存在：

     * `paths.*`：`data_dir`, `ts_data_dir`, `parquet_dir`, `logs_dir`, `experiments_dir`, `monitor_dir`, `jobs_yaml` 等。
     * `network.*`：超时时间、重试、代理配置（是否开启 V2rayN）。
     * `event_bus.*`：ZeroMQ 的 `pub_endpoint / sub_endpoint / monitor_endpoint`。
     * `storage.ts.*`：DuckDB 路径 `duckdb_path`、Parquet 目录。
     * `broker / trade_core / portfolio / risk / alpha / hud / ops / jobs`。

2. **初始化目录**

```bash
mkdir -p data/raw data/ts data/parquet
mkdir -p logs experiments monitor reports
mkdir -p cache/alpha
```

3. **配置 Job 文件（如果使用 `config/jobs.yaml`）**

   * 检查 `config/jobs.yaml` 是否存在；
   * 确认至少包含：

     * `night_ops_full`：夜间全流程任务；
     * `day_ops_morning`：盘前体检；
     * `nightly_regression`：回归测试（可默认关闭）。

---

### 1.4 环境自检（Smoke Test）

运行最轻量的回归脚本，验证：

* Python 依赖无缺；
* DuckDB 可写；
* ZeroMQ 可以 import；
* 基础模块可 import。

```bash
python -m tools.regression_runner --mode smoke
```

预期结果：

* 脚本成功退出（Exit Code 0）；
* `logs/` 下有基础日志；
* 无明显 ImportError / ModuleNotFoundError。

> **如果 Smoke Test 失败**：在继续之前先修，避免把“半瘫痪”的环境拉去跑战役。

---

## 2. 跑一轮完整 NightOps / DayOps / 仿真

这一节的目标是：**从冷启动一路跑到“有一轮校园模拟实战结果”**。

---

### 2.1 跑一轮 NightOps（夜间科研流水线）

NightOps 的典型职责：

* 数据体检（DataGuardian）、修复（TSDataRepairer）、补数据；
* 因子计算（FactorEngine）、标签生成（LabelCenter/ts_labeler）；
* 数据集构建（DatasetBuilder）；
* 训练与回测（TrainingPipelines + BacktestCore）；
* 实验记录（ExperimentLab）。

> 有对应的 CLI（以 Mk3/ Mk4 方案为基础）：

```bash
python -m ops.night_ops --mode full
# 或：只指定系统默认配置即可
# python -m ops.night_ops
```

运行过程中重点关注：

* 控制台日志 / `logs/ops/night_ops.log`；
* DuckDB 是否生成或被更新：`data/lighthunter.duckdb`；
* Parquet 目录是否产生新文件：`data/parquet/`；
* 实验记录是否写入：`experiments/experiments.jsonl`（或 ExperimentLab 的存储）。

**NightOps 成功标志：**

1. 命令正常结束（Exit Code 0）；
2. 日志中无致命 ERROR；
3. `experiments.jsonl` 中新增 1 条或多条记录；
4. DuckDB / Parquet 中有最新日期的数据分区。

---

### 2.2 跑一轮 DayOps（盘前体检）

DayOps 的职责：

* 检查昨晚 NightOps 是否成功；
* 检查今日交易前的数据健康度（数据缺失/异常、风险参数等）；
* 输出盘前体检报告（可为日志/报告文件）。

运行：

```bash
python -m ops.day_ops
```

关注输出：

* `logs/ops/day_ops.log`；
* 体检报告（例如 `reports/day_ops_YYYYMMDD.json` 或 `.md`）；
* 是否提示某些关键数据缺失或异常。

---

### 2.3 跑一轮完整回测 / 仿真

1. **跑一次端到端回测**

```bash
python examples/run_end_to_end_backtest.py
```

预期：

* 使用当前最新的因子+标签+模型配置跑一轮回测；
* 回测结果写入 `reports/` 或 `experiments/`；
* PerformanceLab 写入统一绩效指标集（年化收益、最大回撤、Sharpe、胜率等）。

2. **跑一次日内仿真**

```bash
python examples/run_intraday_simulation.py
```

预期：

* 使用历史分钟级数据进行回放；
* 通过 ZeroMQ EventBus 发布行情与信号事件；
* TradeCore + Broker/PaperBroker 完成交互，产生 ExecutionEvent；
* MonitorDaemon / Web HUD 可以看到账户净值变动和风险告警。

3. **启动命令行 Dashboard 与 Web HUD（可选但推荐）**

```bash
# 命令行监控面板
python -m tools.lighthunter_dashboard

# Web HUD
python -m hud.web_server
# 浏览器打开 http://127.0.0.1:8000 （按 hud.port 配置）  
```

---

## 3. 制造一次失败：故障注入演练

下面给出一套**可重复、可恢复**的失败场景，方便你在战备演练中使用。

### 3.1 失败场景 A：DuckDB 路径错误导致 NightOps 崩溃

1. **保证当前工作树干净**

```bash
git status  # 确保 clean
```

2. **修改配置制造错误**

编辑 `config/system_config.json`，将 `storage.ts.duckdb_path` 改成一个明显非法的路径，例如：

```jsonc
"storage": {
  "ts": {
    "engine": "duckdb",
    "write_mode": "dual",
    "sqlite_path": "data/ts.sqlite",
    "duckdb_path": "data/__invalid_path__/lighthunter.duckdb",  // 故意错误
    "parquet_dir": "data/parquet"
  }
}
```

3. **执行 NightOps**

```bash
python -m ops.night_ops --mode full
```

预期现象：

* NightOps 在尝试初始化 DuckDBClient 或 TSStorage 时抛出异常；
* 日志中出现类似“无法打开数据库/路径不存在”的错误；
* 命令非 0 退出。

4. **观察系统行为**

* 检查 `logs/ops/night_ops.log` 中 NightOps 对错误的处理（是否优雅 fail）。
* 如果 MonitorDaemon 正在运行，应能在 `logs/monitor/metrics.jsonl` 中看到错误率增加。
* Web HUD 不会崩，但不会有新的实验记录出现。

---

### 3.2 失败场景 B：策略异常导致 TradeCore / Portfolio 报错

1. 修改某个策略参数（例如在 `strategy/portfolio_manager.py` 中，把某个权重设为非法值 > 1）。

2. 运行 `examples/run_end_to_end_backtest.py` 或 `run_intraday_simulation.py`。

3. 观察：

   * TradeCore 是否对异常做了捕获；
   * ExperimentLab 是否记录这次实验失败状态；
   * MonitorDaemon 是否记录 error 事件。

> **注意**：这种“逻辑错误”故障更接近实际研发中经常发生的情况，非常适合演练回滚。

---

## 4. 回滚演练：Git + 配置 + 模型快照

目标：

* 经历一次失败后，通过标准流程，在**有限时间内把系统恢复到“最近一次确认无误”的状态**。

演练包括 3 个层面：

1. **代码回滚（Git）**
2. **配置回滚（system_config / jobs / 实验矩阵等）**
3. **模型快照回滚（选择稳定模型）**

---

### 4.1 代码层回滚（Git）

#### 4.1.1 建立“可回滚”基线

在你认为“当前状态是 OK 的”时：

```bash
git status          # 确保 clean
git log -1          # 记住当前 commit id

# 打一个标记（可选但推荐）
git tag war-baseline-YYYYMMDD
```

#### 4.1.2 发生故障后的代码恢复

假设你在这之后改动了若干文件，导致 NightOps / Backtest 大面积失败：

1. 查看变化：

   ```bash
   git status
   git diff
   ```

2. 如果不需要保留这些改动，直接恢复到最近一次提交：

   ```bash
   git reset --hard HEAD
   # 或恢复到特定 tag
   # git reset --hard war-baseline-YYYYMMDD
   ```

3. 再次检查：

   ```bash
   git status  # clean
   ```

> **实践建议**：
>
> * **配置文件尽量纳入 Git 管理**，不要在生产环境长期存在“未提交但生效”的配置。
> * 针对不同环境（dev/prod）可以用不同的分支 / 配置文件（如 `system_config.dev.json` / `system_config.prod.json`），而不是只依赖本地修改。

---

### 4.2 配置层回滚

配置文件包括：

* `config/system_config.json`；
* `config/jobs.yaml`（任务矩阵）；
* `config/experiment_matrix.yaml`（实验矩阵，如已实现）；
* 其他如 `label_spec.json` / `data_schema.json` 等。

#### 4.2.1 配置版本管理建议

1. **配置全部纳入 Git**（敏感信息通过环境变量或单独加密文件存储）。

2. 针对关键配置改动，一律使用 commit，并写清楚说明，例如：

   ```bash
   git commit -am "tune: increase eastmoney rate_limit and update jobs schedule"
   ```

3. 保留一份“稳定版本”的配置 tag：

   ```bash
   git tag war-config-stable-YYYYMMDD
   ```

#### 4.2.2 出问题后的配置回滚

当你发现某次配置调整导致系统持续报错，例如：

* 修改 `network.proxies` / `rate_limits` 后东财/同花顺全部 403；
* 修改 `jobs` 定时导致 NightOps 不再按预期时间执行；

可以：

```bash
# 回到上一次稳定配置的 commit / tag
git checkout war-config-stable-YYYYMMDD -- config/system_config.json config/jobs.yaml

# 确认 diff
git diff

# 如果确认无误，提交这次“配置回滚”
git commit -am "rollback: revert config to war-config-stable-YYYYMMDD"
```

---

### 4.3 模型快照回滚

这里的目标是：

> 当最新一次训练/调参的模型在回测 / 仿真中表现恶化时，能够**快速切回上一个稳定模型**。

#### 4.3.1 推荐的模型管理结构

假设 `system_config.alpha` 中已经约定：

```jsonc
"alpha": {
  "default_job": "ultrashort_main",
  "model_dir": "models",
  "cache_dir": "cache/alpha"
}
```

建议在 `models/` 下使用类似结构：

```text
models/
  ultrashort_main/
    20250101_exp1234/     # 某次实验保存的模型（稳定）
    20250110_exp5678/     # 新模型（效果待观察）
    current -> 20250110_exp5678  # symlink / 记录当前正在实战/仿真使用的模型
```

这样回滚时，只需要：

```bash
cd models/ultrashort_main

# 把 current 指回旧模型
rm current
ln -s 20250101_exp1234 current
```

或在配置中显式指定 `model_id`（如果 TrainingPipelines / ModelZoo 支持从配置读取模型 ID）。

#### 4.3.2 演练过程

1. 记录当前稳定模型目录或 experiment_id；
2. 训练一个新模型（NightOps / 单独训练脚本），并将其设置为 `current`；
3. 跑一轮 walk-forward 回测 / 仿真，发现新模型表现劣化；
4. 回滚步骤：

   * 修改 `models/ultrashort_main/current` 指向旧模型；
   * 或修改 `config/system_config.json` / 实验配置中的 `model_id` 或 `job-id`，让系统重新使用旧模型；
   * 再跑一轮回测 / 仿真验证恢复正常。

---

## 5. 一次完整“拉起 + 故障 + 回滚”演练流程（建议脚本化）

最终你可以把以上内容总结为一条标准“战备演练流程”，示例：

1. **环境准备**

   * [ ] 新建或清理一台干净环境（容器/虚拟机更好）；
   * [ ] 克隆仓库 + 切换到演练分支；
   * [ ] 创建虚拟环境并安装依赖；
   * [ ] 初始化配置与目录；
   * [ ] 跑 Smoke Test 通过。

2. **第一次完整跑通**

   * [ ] `python -m ops.night_ops --mode full` 成功；
   * [ ] `python -m ops.day_ops` 成功；
   * [ ] `python examples/run_end_to_end_backtest.py` 成功；
   * [ ] HUD/Monitor 能看到数据流与实验结果。

3. **故障注入**

   * [ ] 修改配置（如 DuckDB 路径）或策略逻辑，确保下次 NightOps / 回测必然失败；
   * [ ] 记录故障日志与系统行为（MonitorDaemon、HUD、日志文件等）。

4. **回滚**

   * [ ] 使用 Git 将代码恢复到基线（`war-baseline-YYYYMMDD`）；
   * [ ] 使用 Git/配置回滚 `system_config.json` / `jobs.yaml` 到稳定版本；
   * [ ] 将模型 `current` 指回稳定模型；
   * [ ] 再跑一轮 NightOps / DayOps / 回测，确认系统恢复到预期状态。

5. **复盘**

   * [ ] 记录本次演练的耗时、出现的问题、需要自动化的环节；
   * [ ] 将高频手工步骤固化成脚本（例如 `scripts/war_drill.sh`）；
   * [ ] 更新文档（WAR_SOP / WAR_CHECKLIST / ARCHITECTURE）以反映最新流程。

---

如果你愿意，接下来我们可以做两件事：

1. 把这份手册拆成几个具体脚本（例如 `scripts/bootstrap_env.sh`、`scripts/war_drill.sh`）；
2. 根据你本机实际的 `system_config.json` 和目录结构，微调一版 **“适配你当前环境的专属战备手册”**。
