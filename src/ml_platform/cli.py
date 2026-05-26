"""Thin CLI entrypoint for local workflow commands."""

from __future__ import annotations

import argparse
import importlib


COMMAND_MODULES = {
    "active-mine": "ml_platform.data.mining",
    "evaluate-promotion": "ml_platform.deployment.gates",
    "package-model": "ml_platform.deployment.package",
    "train-distributed": "ml_platform.training.distributed",
    "train": "ml_platform.training.train",
    "serve": "ml_platform.inference.api",
    "retrain-watcher": "ml_platform.orchestration.retrain_watcher",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-ml-platform")
    parser.add_argument("command", choices=sorted(COMMAND_MODULES))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    module = importlib.import_module(COMMAND_MODULES[args.command])
    if hasattr(module, "main"):
        module.main()
        return
    parser.error(f"Command {args.command!r} does not expose a main() function.")


if __name__ == "__main__":
    main()
