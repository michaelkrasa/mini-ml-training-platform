"""Helpers for launching local PyTorch distributed training jobs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TorchrunConfig:
    nproc_per_node: int = 2
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    training_args: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_torchrun_command(config: TorchrunConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(config.nproc_per_node),
        "--nnodes",
        str(config.nnodes),
        "--node-rank",
        str(config.node_rank),
        "--master-addr",
        config.master_addr,
        "--master-port",
        str(config.master_port),
        "-m",
        "ml_platform.training.train",
        *config.training_args,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the torchrun command instead of executing it.",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to ml_platform.training.train. Prefix with --.",
    )
    return parser


def _normalize_passthrough(args: list[str]) -> tuple[str, ...]:
    if args and args[0] == "--":
        return tuple(args[1:])
    return tuple(args)


def main() -> None:
    args = build_parser().parse_args()
    config = TorchrunConfig(
        nproc_per_node=args.nproc_per_node,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        training_args=_normalize_passthrough(args.training_args),
    )
    command = build_torchrun_command(config)
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=Path.cwd(), check=True)


if __name__ == "__main__":
    main()
