"""Constants shared across training, registry, and inference."""

from __future__ import annotations

KITTI_CLASS_NAMES = (
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
)

DEFAULT_IMAGE_SIZE = (96, 96)

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
