# -*- coding: utf-8 -*-
"""
模块名称：BackupManager Mk-Shield
版本：Mk-Shield R10 (DB/CSV/JSON/LOG Snapshot)
路径: G:/LightHunter_Mk1/backup_manager.py

功能：
- 一键打包当前项目下的所有 db/csv/json/log 文件；
- 按日期时间命名备份文件，自动保留最近 N 份（由 config.backup.keep_last_n 控制）；
- 可从指定备份（或最新一份）恢复，覆盖本地文件。

注意：
- 恢复操作会覆盖现有 db/csv/json/log，请谨慎确认。
"""

import os
import sys
import zipfile
import datetime
from typing import List, Dict, Any

# ------------- ConfigCenter 接入 ----------------
try:
    from config.config_manager import get_backup_config, get_config
except Exception:  # 兼容老环境
    def get_backup_config() -> Dict[str, Any]:
        return {
            "output_dir": "backups",
            "compress": True,
            "keep_last_n": 10,
            "include_db": True,
            "include_csv": True,
            "include_json": True,
            "include_log": True,
            "extra_files": [],
            "exclude_dirs": ["__pycache__", ".git", ".idea", ".vscode", "venv"],
        }

    def get_config() -> Dict[str, Any]:
        return {}
# ------------------------------------------------


class BackupManager:
    def __init__(self, root_dir: str | None = None, backup_cfg: Dict[str, Any] | None = None):
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = root_dir

        self.backup_cfg = backup_cfg or get_backup_config() or {}
        self.output_dir = os.path.join(
            self.root_dir, self.backup_cfg.get("output_dir", "backups")
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.include_db = bool(self.backup_cfg.get("include_db", True))
        self.include_csv = bool(self.backup_cfg.get("include_csv", True))
        self.include_json = bool(self.backup_cfg.get("include_json", True))
        self.include_log = bool(self.backup_cfg.get("include_log", True))
        self.extra_files = self.backup_cfg.get("extra_files") or []
        self.exclude_dirs = set(self.backup_cfg.get("exclude_dirs") or [])

    # -----------------------------
    # 收集需要备份的文件
    # -----------------------------
    def _collect_files(self) -> List[str]:
        targets: List[str] = []
        exts_map = []
        if self.include_db:
            exts_map.append(".db")
        if self.include_csv:
            exts_map.append(".csv")
        if self.include_json:
            exts_map.append(".json")
        if self.include_log:
            exts_map.append(".log")

        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # 跳过备份目录本身以及排除目录
            rel_dir = os.path.relpath(dirpath, self.root_dir)
            if rel_dir == os.path.relpath(self.output_dir, self.root_dir):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]

            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts_map:
                    full = os.path.join(dirpath, fn)
                    targets.append(full)
                # 特判 log 文本（.txt）
                if self.include_log and ext == ".txt":
                    if "log" in dirpath.lower() or "log" in fn.lower():
                        full = os.path.join(dirpath, fn)
                        targets.append(full)

        # 额外强制打包的关键文件
        for rel in self.extra_files:
            full = os.path.join(self.root_dir, rel)
            if os.path.exists(full) and full not in targets:
                targets.append(full)

        return sorted(set(targets))

    # -----------------------------
    # 创建备份
    # -----------------------------
    def create_backup(self, tag: str | None = None) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if tag:
            tag = "".join(c for c in tag if c.isalnum() or c in ("-", "_"))
            fname = f"lh_backup_{ts}_{tag}.zip"
        else:
            fname = f"lh_backup_{ts}.zip"

        backup_path = os.path.join(self.output_dir, fname)
        files = self._collect_files()

        if not files:
            print("[BACKUP] No files to backup.")
            return backup_path

        print(f"[BACKUP] Creating backup: {backup_path}")
        with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for full in files:
                arcname = os.path.relpath(full, self.root_dir)
                zf.write(full, arcname)
        print(f"[BACKUP] Done. {len(files)} files archived.")

        self._cleanup_old_backups()
        return backup_path

    # -----------------------------
    # 备份清理：只保留最近 N 份
    # -----------------------------
    def _cleanup_old_backups(self) -> None:
        keep_n = int(self.backup_cfg.get("keep_last_n", 10))
        if keep_n <= 0:
            return

        files = [
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.lower().endswith(".zip")
        ]
        if len(files) <= keep_n:
            return

        files.sort(key=lambda p: os.path.getmtime(p))
        to_remove = files[:-keep_n]
        for p in to_remove:
            try:
                os.remove(p)
                print(f"[BACKUP] Old backup removed: {p}")
            except Exception:
                pass

    # -----------------------------
    # 列出已有备份
    # -----------------------------
    def list_backups(self) -> List[str]:
        files = [
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.lower().endswith(".zip")
        ]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files

    # -----------------------------
    # 恢复备份
    # -----------------------------
    def restore_backup(self, backup_path: str | None = None) -> None:
        if backup_path is None:
            backups = self.list_backups()
            if not backups:
                print("[RESTORE] No backup found.")
                return
            backup_path = backups[0]

        if not os.path.exists(backup_path):
            print(f"[RESTORE] Backup file not found: {backup_path}")
            return

        print(f"[RESTORE] Ready to restore from: {backup_path}")
        confirm = input("  This will OVERWRITE local db/csv/json/log files. Continue? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("[RESTORE] Aborted by user.")
            return

        with zipfile.ZipFile(backup_path, "r") as zf:
            members = zf.namelist()
            for m in members:
                # 安全保护：禁止解压出项目根目录之外
                target_path = os.path.normpath(os.path.join(self.root_dir, m))
                if not target_path.startswith(self.root_dir):
                    continue
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                zf.extract(m, self.root_dir)

        print("[RESTORE] Done. Please re-check your environment before trading.")


def main():
    bm = BackupManager()

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python backup_manager.py backup  [tag]\n"
            "  python backup_manager.py restore [backup_file]\n"
        )
        return

    cmd = sys.argv[1].lower()
    if cmd in ("backup", "b"):
        tag = sys.argv[2] if len(sys.argv) > 2 else None
        bm.create_backup(tag)
    elif cmd in ("restore", "r"):
        backup_file = sys.argv[2] if len(sys.argv) > 2 else None
        bm.restore_backup(backup_file)
    else:
        print(
            "Unknown command.\n"
            "  python backup_manager.py backup  [tag]\n"
            "  python backup_manager.py restore [backup_file]\n"
        )


if __name__ == "__main__":
    main()
