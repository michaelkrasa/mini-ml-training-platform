import sys

from ml_platform.training.distributed import TorchrunConfig, build_torchrun_command


def test_build_torchrun_command_wraps_training_module() -> None:
    command = build_torchrun_command(
        TorchrunConfig(
            nproc_per_node=4,
            master_port=29999,
            training_args=("--epochs", "1"),
        )
    )

    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc-per-node" in command
    assert "4" in command
    assert command[-4:] == ["-m", "ml_platform.training.train", "--epochs", "1"]
