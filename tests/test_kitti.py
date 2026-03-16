from pathlib import Path

from ml_platform.data.kitti import parse_kitti_label_file


def test_parse_kitti_label_file_ignores_dontcare(tmp_path: Path) -> None:
    label_path = tmp_path / "000000.txt"
    label_path.write_text(
        "\n".join(
            [
                "Car 0.0 0 0.0 10 10 20 20 0 0 0 0 0 0 0",
                "DontCare 0.0 0 0.0 10 10 20 20 0 0 0 0 0 0 0",
                "Cyclist 0.0 0 0.0 15 15 25 25 0 0 0 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )

    labels = parse_kitti_label_file(label_path)

    assert labels == ("Car", "Cyclist")
