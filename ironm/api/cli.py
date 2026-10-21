"""管理员命令（spec [2026-10-21] §14）：添加用户、给设备发一次性设备码、列出与停用用户。

用法（数据库路径等配置同服务，取自 IRONM_ 环境变量）：
    uv run python -m ironm.api.cli add-user <名字>        # 添加用户并输出第一张设备码
    uv run python -m ironm.api.cli device-code <用户编号>  # 给已有用户的新设备发设备码
    uv run python -m ironm.api.cli users                  # 列出用户
    uv run python -m ironm.api.cli disable <用户编号>      # 停用用户（所有设备立即失效）
"""

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing

from ironm.core.clock import Clock, SystemClock
from ironm.core.config import load_settings
from ironm.store.auth import CODE_TTL, AuthStore
from ironm.store.db import connect
from ironm.store.migrate import MIGRATIONS_DIR, migrate

__all__ = ["main"]


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    clock: Clock | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="ironm-admin", description="IronM 管理员命令")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add-user").add_argument("name")
    sub.add_parser("device-code").add_argument("user_id")
    sub.add_parser("users")
    sub.add_parser("disable").add_argument("user_id")
    args = parser.parse_args(argv)
    settings = load_settings(os.environ if environ is None else environ)
    clock = clock or SystemClock()
    hours = int(CODE_TTL.total_seconds() // 3600)
    with closing(connect(settings.db_path)) as conn:
        migrate(conn, MIGRATIONS_DIR, clock)
        store = AuthStore(conn, clock)
        if args.cmd == "add-user":
            user_id = store.add_user(args.name)
            print(f"已添加用户 {args.name}（{user_id}）")
            print(f"设备码（{hours} 小时内有效，用一次即作废）：")
            print(store.new_code(user_id))
        elif args.cmd == "device-code":
            try:
                code = store.new_code(args.user_id)
            except KeyError:
                print("没有这个用户，或该用户已停用", file=sys.stderr)
                return 1
            print(f"设备码（{hours} 小时内有效，用一次即作废）：")
            print(code)
        elif args.cmd == "users":
            for u in store.users():
                state = "已停用" if u.disabled else "正常"
                print(f"{u.user_id}\t{u.name}\t{state}\t{u.created_at:%Y-%m-%d}")
        elif args.cmd == "disable":
            if not store.disable(args.user_id):
                print("没有这个用户，或已停用", file=sys.stderr)
                return 1
            print("已停用，该用户所有设备立即失效")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
