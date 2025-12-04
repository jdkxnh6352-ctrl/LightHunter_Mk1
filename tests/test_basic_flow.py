# -*- coding: utf-8 -*-
"""
tests/test_basic_flow.py

LightHunter Mk3 - 基础回归测试（pytest）

运行方式：
    pytest -q
或通过 regression_runner 调用：
    python -m tools.regression_runner --mode full --with-pytest
"""

from __future__ import annotations

from config.config_center import get_system_config
import importlib


def test_config_can_load_and_has_core_sections():
    cfg = get_system_config()
    assert isinstance(cfg, dict)
    for key in ["paths", "network", "event_bus", "broker", "portfolio", "jobs"]:
        assert key in cfg, f"system_config 中缺少关键字段：{key}"


def test_duckdb_basic_query():
    """
    DuckDB 连接与简单查询测试。
    """
    cfg = get_system_config()
    duckdb_mod = importlib.import_module("storage.duckdb_client")
    # 优先 get_duckdb_client(cfg)
    if hasattr(duckdb_mod, "get_duckdb_client"):
        client = duckdb_mod.get_duckdb_client(cfg)  # type: ignore
    else:
        DuckDBClient = getattr(duckdb_mod, "DuckDBClient")
        duck_cfg = cfg.get("duckdb") or {}
        db_path = duck_cfg.get("database") or duck_cfg.get("db_path")
        if not db_path:
            paths = cfg.get("paths") or {}
            db_path = paths.get("duckdb_path") or "data/duckdb/lighthunter.duckdb"
        client = DuckDBClient(db_path=db_path)  # type: ignore

    if hasattr(client, "execute"):
        cur = client.execute("SELECT 1 AS v")
        row = cur.fetchone()
        v = row[0] if row else None
    elif hasattr(client, "conn"):
        cur = client.conn.execute("SELECT 1 AS v")  # type: ignore
        row = cur.fetchone()
        v = row[0] if row else None
    else:
        raise AssertionError("DuckDBClient 缺少 execute/conn.execute 方法")

    assert v == 1


def test_zmq_bus_init():
    """
    ZeroMQ 总线初始化测试。
    """
    cfg = get_system_config()
    zmq_mod = importlib.import_module("bus.zmq_bus")
    get_zmq_bus = getattr(zmq_mod, "get_zmq_bus")
    bus = get_zmq_bus(cfg)  # type: ignore
    assert bus is not None
    assert any(
        hasattr(bus, name) for name in ("publish_event", "publish", "send")
    ), "ZMQBus 缺少 publish/publish_event/send 之一"


def test_experiment_lab_smoke_run():
    """
    ExperimentLab 能否创建 run 并关闭。
    """
    cfg = get_system_config()
    lab_mod = importlib.import_module("lab.experiment_lab")

    if hasattr(lab_mod, "get_experiment_lab"):
        lab = lab_mod.get_experiment_lab(cfg)  # type: ignore
    else:
        ExperimentLab = getattr(lab_mod, "ExperimentLab")
        lab = ExperimentLab(cfg)  # type: ignore

    assert lab is not None

    if hasattr(lab, "start_run"):
        run_id = lab.start_run(  # type: ignore
            name="pytest_basic_smoke",
            run_type="pytest",
            config={"suite": "basic_flow"},
        )
    elif hasattr(lab, "create_run"):
        run_id = lab.create_run(  # type: ignore
            name="pytest_basic_smoke",
            kind="pytest",
            config={"suite": "basic_flow"},
        )
    else:
        raise AssertionError("ExperimentLab 缺少 start_run/create_run")

    assert run_id

    if hasattr(lab, "end_run"):
        lab.end_run(run_id, status="ok")  # type: ignore


def test_broker_snapshot_accounts():
    """
    Broker 抽象层能否返回账户快照。
    """
    cfg = get_system_config()
    broker_mod = importlib.import_module("broker.broker_api")
    get_default_broker = getattr(broker_mod, "get_default_broker")
    broker = get_default_broker(cfg)  # type: ignore

    snap = broker.snapshot_all_accounts()  # type: ignore
    assert isinstance(snap, dict)
    accounts = snap.get("accounts") or {}
    assert isinstance(accounts, dict)
    assert accounts, "snapshot_all_accounts 返回 accounts 为空"
