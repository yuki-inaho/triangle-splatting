#!/usr/bin/env python3
"""Render a checkpoint along the source COLMAP camera trajectory.

Unlike create_video.py, this script does not synthesize an ellipse path. It
renders the cameras loaded from the source scene, preserving image-name order,
and then writes those frames to an mp4.
"""

from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import mediapy as media
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from scene import Scene  # noqa: E402
from triangle_renderer import TriangleModel, render  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def select_views(scene: Scene, camera_set: str) -> list:
    train_views = list(scene.getTrainCameras())
    test_views = list(scene.getTestCameras())

    if camera_set == "train":
        views = train_views
    elif camera_set == "test":
        views = test_views
    elif camera_set == "all":
        views = train_views + test_views
    else:
        raise ValueError(f"Unsupported camera_set: {camera_set}")

    return sorted(views, key=lambda camera: camera.image_name)


def save_manifest(
    manifest_path: Path,
    *,
    model_path: str,
    source_path: str,
    iteration: int,
    camera_set: str,
    fps: int,
    views: list,
    video_path: Path,
) -> None:
    frames = [
        {
            "frame": idx,
            "image_name": view.image_name,
            "colmap_id": int(view.colmap_id),
            "uid": int(view.uid),
        }
        for idx, view in enumerate(views)
    ]
    manifest = {
        "created_utc": utc_now(),
        "model_path": model_path,
        "source_path": source_path,
        "iteration": iteration,
        "camera_set": camera_set,
        "frame_count": len(views),
        "fps": fps,
        "video_path": str(video_path),
        "frames": frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_video(frames_dir: Path, video_path: Path, frame_count: int, fps: int) -> None:
    first_frame = frames_dir / "00000.png"
    if not first_frame.exists():
        raise FileNotFoundError(f"Missing first rendered frame: {first_frame}")

    first = np.array(Image.open(first_frame).convert("RGB"), dtype=np.float32) / 255.0
    video_kwargs = {
        "shape": first.shape[:2],
        "codec": "h264",
        "fps": fps,
        "crf": 18,
    }

    print(f"VIDEO_SHAPE={first.shape[:2]}")
    print(f"VIDEO_PATH={video_path}")
    with media.VideoWriter(video_path, **video_kwargs, input_format="rgb") as writer:
        for idx in tqdm(range(frame_count), desc="Video writing progress"):
            frame_path = frames_dir / f"{idx:05d}.png"
            if not frame_path.exists():
                raise FileNotFoundError(f"Missing rendered frame: {frame_path}")
            image = np.array(Image.open(frame_path).convert("RGB"), dtype=np.float32) / 255.0
            writer.add_image(image)


def main() -> None:
    parser = ArgumentParser(description="Render source COLMAP camera trajectory to video")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--camera_set", choices=["train", "test", "all"], default="all")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--max_frames", default=None, type=int)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--out_name", default=None, type=str)
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    max_frames = getattr(args, "max_frames", None)
    out_dir_arg = getattr(args, "out_dir", None)
    out_name_arg = getattr(args, "out_name", None)

    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max_frames must be positive when provided")

    start = time.monotonic()
    print(f"START_UTC={utc_now()}")
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=None,
        init_size=None,
        nb_points=None,
        set_sigma=None,
        no_dome=False,
        load_iteration=args.iteration,
        shuffle=False,
    )

    views = select_views(scene, args.camera_set)
    views = views[:: args.stride]
    if max_frames is not None:
        views = views[:max_frames]
    if not views:
        raise RuntimeError(f"No cameras selected for camera_set={args.camera_set}")

    output_dir = (
        Path(out_dir_arg)
        if out_dir_arg
        else Path(dataset.model_path)
        / "colmap_traj"
        / f"{args.camera_set}_ours_{scene.loaded_iter}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = output_dir / "renders"
    gt_dir = output_dir / "gt"
    renders_dir.mkdir(parents=True, exist_ok=True)
    if args.save_gt:
        gt_dir.mkdir(parents=True, exist_ok=True)

    out_name = out_name_arg or f"render_colmap_{args.camera_set}"
    video_path = output_dir / f"{out_name}_color.mp4"
    manifest_path = output_dir / "frame_manifest.json"

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"MODEL_PATH={dataset.model_path}")
    print(f"SOURCE_PATH={dataset.source_path}")
    print(f"ITERATION={scene.loaded_iter}")
    print(f"CAMERA_SET={args.camera_set}")
    print(f"FRAME_COUNT={len(views)}")
    print(f"FPS={args.fps}")
    print(f"OUTPUT_DIR={output_dir}")

    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, triangles, pipe, background)["render"]
            torchvision.utils.save_image(rendering, renders_dir / f"{idx:05d}.png")
            if args.save_gt:
                gt = view.original_image[0:3, :, :]
                torchvision.utils.save_image(gt, gt_dir / f"{idx:05d}.png")

    write_video(renders_dir, video_path, len(views), args.fps)
    save_manifest(
        manifest_path,
        model_path=dataset.model_path,
        source_path=dataset.source_path,
        iteration=scene.loaded_iter,
        camera_set=args.camera_set,
        fps=args.fps,
        views=views,
        video_path=video_path,
    )

    print(f"MANIFEST_PATH={manifest_path}")
    print(f"END_UTC={utc_now()}")
    print(f"ELAPSED_SECONDS={time.monotonic() - start:.3f}")


if __name__ == "__main__":
    main()
