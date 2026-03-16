"""Generate a synthetic KITTI-style dataset for local platform demos."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw

from ml_platform.config import get_settings
from ml_platform.constants import KITTI_CLASS_NAMES
from ml_platform.data.profile import DatasetProfile, write_dataset_profile

IMAGE_SIZE = (192, 128)

CLASS_STYLES = {
    "Car": {"shape": "rectangle", "color": (220, 80, 80), "bbox": (18, 76, 78, 112)},
    "Van": {"shape": "rectangle", "color": (228, 148, 59), "bbox": (84, 74, 146, 112)},
    "Truck": {"shape": "rectangle", "color": (240, 205, 78), "bbox": (132, 68, 186, 110)},
    "Pedestrian": {"shape": "rectangle", "color": (70, 122, 240), "bbox": (88, 28, 102, 90)},
    "Person_sitting": {"shape": "ellipse", "color": (144, 87, 227), "bbox": (32, 88, 70, 118)},
    "Cyclist": {"shape": "triangle", "color": (87, 192, 96), "bbox": (112, 36, 154, 92)},
    "Tram": {"shape": "rectangle", "color": (68, 193, 210), "bbox": (8, 10, 184, 36)},
    "Misc": {"shape": "diamond", "color": (250, 250, 250), "bbox": (148, 18, 182, 52)},
}


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    class_names: tuple[str, ...]


def _triangle_points(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    left, top, right, bottom = bbox
    return [(left, bottom), ((left + right) // 2, top), (right, bottom)]


def _diamond_points(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    left, top, right, bottom = bbox
    mid_x = (left + right) // 2
    mid_y = (top + bottom) // 2
    return [(mid_x, top), (right, mid_y), (mid_x, bottom), (left, mid_y)]


def _draw_shape(draw: ImageDraw.ImageDraw, class_name: str) -> tuple[int, int, int, int]:
    style = CLASS_STYLES[class_name]
    bbox = style["bbox"]
    shape = style["shape"]
    color = style["color"]
    if shape == "rectangle":
        draw.rectangle(bbox, fill=color, outline=(20, 20, 20))
    elif shape == "ellipse":
        draw.ellipse(bbox, fill=color, outline=(20, 20, 20))
    elif shape == "triangle":
        draw.polygon(_triangle_points(bbox), fill=color, outline=(20, 20, 20))
    elif shape == "diamond":
        draw.polygon(_diamond_points(bbox), fill=color, outline=(20, 20, 20))
    return bbox


def _label_row(class_name: str, bbox: tuple[int, int, int, int]) -> str:
    left, top, right, bottom = bbox
    return (
        f"{class_name} 0.00 0 0.00 {left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
        "1.50 1.60 3.90 0.00 0.00 0.00 0.00"
    )


def _build_sample_specs(sample_count: int, seed: int) -> list[SampleSpec]:
    rng = Random(seed)
    specs: list[SampleSpec] = []
    for index in range(sample_count):
        sample_id = f"{index:06d}"
        label_count = rng.randint(1, 3)
        chosen = tuple(sorted(rng.sample(list(KITTI_CLASS_NAMES), k=label_count)))
        specs.append(SampleSpec(sample_id=sample_id, class_names=chosen))
    return specs


def generate_dataset(output_root: Path, sample_count: int, force: bool, seed: int) -> None:
    settings = get_settings()
    settings.ensure_directories()

    image_dir = output_root / "images"
    label_dir = output_root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(image_dir.iterdir()):
        raise FileExistsError(
            f"{image_dir} already contains files. Pass --force to overwrite."
        )

    for existing in image_dir.glob("*"):
        if existing.name == ".gitkeep":
            continue
        existing.unlink()
    for existing in label_dir.glob("*"):
        if existing.name == ".gitkeep":
            continue
        existing.unlink()

    specs = _build_sample_specs(sample_count=sample_count, seed=seed)
    background_rng = Random(seed + 99)
    for spec in specs:
        background = (
            background_rng.randint(16, 54),
            background_rng.randint(26, 64),
            background_rng.randint(40, 88),
        )
        image = Image.new("RGB", IMAGE_SIZE, color=background)
        draw = ImageDraw.Draw(image)

        labels: list[str] = []
        for class_name in spec.class_names:
            bbox = _draw_shape(draw, class_name)
            labels.append(_label_row(class_name, bbox))

        image.save(image_dir / f"{spec.sample_id}.png")
        (label_dir / f"{spec.sample_id}.txt").write_text(
            "\n".join(labels) + "\n", encoding="utf-8"
        )

    write_dataset_profile(
        settings,
        DatasetProfile(
            dataset_name="synthetic_kitti_scene_presence",
            source="local_generator",
            description="Synthetic KITTI-style dataset used for smoke testing the platform.",
            class_names=KITTI_CLASS_NAMES,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=get_settings().dataset_root,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_dataset(
        output_root=args.output_root,
        sample_count=args.sample_count,
        force=args.force,
        seed=args.seed,
    )
    print(
        f"Generated {args.sample_count} KITTI-style samples under {args.output_root}."
    )


if __name__ == "__main__":
    main()
