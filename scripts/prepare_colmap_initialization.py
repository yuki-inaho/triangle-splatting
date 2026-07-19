#!/usr/bin/env python3
"""Stage a COLMAP reconstruction as a Triangle Splatting training scene.

The script preserves the COLMAP cameras, poses, and sparse point cloud.  Images
are converted to three-channel RGB because Triangle Splatting expects RGB
training targets, including when the source modality is monochrome.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scene.colmap_loader import (  # noqa: E402
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
)


SUPPORTED_CAMERA_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}
MODEL_FILENAMES = ("cameras.bin", "images.bin", "points3D.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a COLMAP model and images for Triangle Splatting."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing the images referenced by COLMAP images.bin.",
    )
    parser.add_argument(
        "--reconstruction",
        type=Path,
        required=True,
        help="COLMAP model directory containing cameras.bin, images.bin, and points3D.bin.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or empty output directory for the staged training scene.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help=(
            "Optional deterministic cap for initial splat centers. The complete "
            "COLMAP points3D.bin is still copied unchanged."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used only with --max-points (default: 0).",
    )
    return parser.parse_args()


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")


def resolve_model_directory(path: Path) -> Path:
    """Accept either a COLMAP model directory or a reconstruction root."""
    if all((path / filename).is_file() for filename in MODEL_FILENAMES):
        return path
    nested = path / "sparse" / "0"
    if all((nested / filename).is_file() for filename in MODEL_FILENAMES):
        return nested
    expected = ", ".join(MODEL_FILENAMES)
    raise ValueError(f"Could not find {expected} in {path} or {nested}")


def prepare_output_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"Output directory must be empty to avoid overwriting data: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def validate_model(model_dir: Path) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    cameras = read_intrinsics_binary(model_dir / "cameras.bin")
    images = read_extrinsics_binary(model_dir / "images.bin")
    xyz, rgb, _ = read_points3D_binary(model_dir / "points3D.bin")

    if not cameras:
        raise ValueError("COLMAP model contains no cameras")
    if not images:
        raise ValueError("COLMAP model contains no registered images")
    if len(xyz) == 0:
        raise ValueError("COLMAP model contains no 3D points")

    unsupported = sorted({camera.model for camera in cameras.values()} - SUPPORTED_CAMERA_MODELS)
    if unsupported:
        raise ValueError(
            "Triangle Splatting supports only undistorted PINHOLE or SIMPLE_PINHOLE "
            f"cameras; found: {', '.join(unsupported)}"
        )

    missing_camera_ids = sorted(
        {image.camera_id for image in images.values()} - set(cameras)
    )
    if missing_camera_ids:
        raise ValueError(f"images.bin references missing camera IDs: {missing_camera_ids}")

    image_names = [image.name for image in images.values()]
    if len(image_names) != len(set(image_names)):
        raise ValueError("images.bin contains duplicate image names")
    if any(Path(name).name != name for name in image_names):
        raise ValueError(
            "Nested image paths are not supported by this Triangle Splatting loader; "
            "use basenames in images.bin"
        )
    return cameras, images, xyz, rgb


def select_initial_points(
    xyz: np.ndarray, rgb: np.ndarray, max_points: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None:
        return xyz, rgb
    if max_points <= 0:
        raise ValueError("--max-points must be positive")
    if max_points >= len(xyz):
        return xyz, rgb
    indices = np.random.default_rng(seed).choice(len(xyz), size=max_points, replace=False)
    indices.sort()
    return xyz[indices], rgb[indices]


def write_initial_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write the exact selected points in the PLY schema consumed by the loader."""
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertices = np.empty(len(xyz), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0.0
    vertices["red"], vertices["green"], vertices["blue"] = rgb.astype(np.uint8).T
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def stage_images(images_dir: Path, images: dict, output_images_dir: Path) -> list[dict]:
    output_images_dir.mkdir(parents=True, exist_ok=False)
    manifest_images = []
    for image in sorted(images.values(), key=lambda value: value.name):
        source = images_dir / image.name
        if not source.is_file():
            raise ValueError(f"Image referenced by COLMAP model is missing: {source}")

        destination = output_images_dir / image.name
        with Image.open(source) as source_image:
            rgb_image = source_image.convert("RGB")
            rgb_image.save(destination)
            width, height = rgb_image.size

        camera = image.camera_id
        manifest_images.append(
            {
                "name": image.name,
                "camera_id": camera,
                "width": width,
                "height": height,
                "mode": "RGB",
            }
        )
    return manifest_images


def check_image_dimensions(cameras: dict, images: dict, staged_images: list[dict]) -> None:
    by_name = {item["name"]: item for item in staged_images}
    for image in images.values():
        staged = by_name[image.name]
        camera = cameras[image.camera_id]
        if (staged["width"], staged["height"]) != (camera.width, camera.height):
            raise ValueError(
                f"Image dimensions for {image.name} are {staged['width']}x{staged['height']}, "
                f"but COLMAP camera {camera.id} expects {camera.width}x{camera.height}"
            )


def main() -> None:
    args = parse_args()
    images_dir = args.images.resolve()
    reconstruction_dir = args.reconstruction.resolve()
    output_dir = args.output.resolve()

    require_directory(images_dir, "--images")
    require_directory(reconstruction_dir, "--reconstruction")
    model_dir = resolve_model_directory(reconstruction_dir)
    prepare_output_directory(output_dir)

    cameras, images, xyz, rgb = validate_model(model_dir)
    selected_xyz, selected_rgb = select_initial_points(
        xyz, rgb, args.max_points, args.seed
    )
    staged_images = stage_images(images_dir, images, output_dir / "images")
    check_image_dimensions(cameras, images, staged_images)

    output_model_dir = output_dir / "sparse" / "0"
    output_model_dir.mkdir(parents=True, exist_ok=False)
    for filename in MODEL_FILENAMES:
        shutil.copy2(model_dir / filename, output_model_dir / filename)
    if args.max_points is not None:
        write_initial_ply(output_model_dir / "points3D.ply", selected_xyz, selected_rgb)

    manifest = {
        "format": "triangle-splatting-colmap-stage-v1",
        "image_count": len(images),
        "colmap_point_count": len(xyz),
        "initial_splat_point_count": len(selected_xyz),
        "initial_splat_selection": {
            "method": "all-points" if args.max_points is None else "deterministic-uniform-sample",
            "seed": args.seed if args.max_points is not None else None,
        },
        "camera_models": sorted({camera.model for camera in cameras.values()}),
        "source_images_are_staged_as": "RGB",
        "images": staged_images,
    }
    (output_dir / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Staged {len(images)} images, {len(selected_xyz)} initial splat points, "
        f"and {len(xyz)} COLMAP points in {output_dir}"
    )
    print("COLMAP cameras, poses, and point coordinates were copied without modification.")


if __name__ == "__main__":
    main()
