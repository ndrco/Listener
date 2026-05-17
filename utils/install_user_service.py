#!/usr/bin/env python3
"""Install Listener as a systemd user service for the current checkout."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = PROJECT_ROOT / "deploy" / "systemd" / "listener.service"
DEFAULT_TEMPLATE_ROOT = "/home/re/src/Listener"


def build_unit_text(project_root: Path, *, template_path: Path = TEMPLATE_PATH) -> str:
    text = template_path.read_text(encoding="utf-8")
    return text.replace(DEFAULT_TEMPLATE_ROOT, str(project_root.resolve()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install Listener as a Linux systemd --user service.",
    )
    parser.add_argument(
        "--name",
        default="listener",
        help="systemd user service name without .service suffix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated unit and do not write files or call systemctl.",
    )
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Do not run systemctl --user enable.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the service after installation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service_name = str(args.name).strip()
    if not service_name:
        raise SystemExit("service name must not be empty")
    if service_name.endswith(".service"):
        service_name = service_name[: -len(".service")]
    unit_text = build_unit_text(PROJECT_ROOT)
    unit_path = Path.home() / ".config" / "systemd" / "user" / f"{service_name}.service"

    if args.dry_run:
        print(unit_text, end="" if unit_text.endswith("\n") else "\n")
        return 0

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit_text, encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    if not args.no_enable:
        subprocess.run(["systemctl", "--user", "enable", unit_path.name], check=True)
    if args.start:
        subprocess.run(["systemctl", "--user", "start", unit_path.name], check=True)
    print(f"Installed {unit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
