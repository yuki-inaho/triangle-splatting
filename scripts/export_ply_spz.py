#!/usr/bin/env python3
"""Export Triangle Splatting checkpoints to PLY and approximate SPZ.

This writes three artifacts:
  1. a triangle mesh PLY preserving the optimized triangles,
  2. a Gaussian Splat PLY with one approximate Gaussian per triangle,
  3. an SPZ v3 file generated from the same approximate Gaussian data.

The SPZ path is approximate because SPZ stores 3D Gaussians, not triangle
primitives. The conversion uses each triangle centroid as a splat center,
triangle in-plane extents as two Gaussian scales, and the learned sigma as
the normal scale.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sh_utils import eval_sh  # noqa: E402


COLOR_SCALE = 0.15
SPZ_MAGIC = 0x5053474E  # "NGSP" little-endian.
SPZ_VERSION = 3
SPZ_MIN_FIXED = -(1 << 23)
SPZ_MAX_FIXED = (1 << 23) - 1
SPZ_DEFAULT_MAX_FRACTIONAL_BITS = 12


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def latest_iteration(model_path: Path) -> int:
    point_cloud_dir = model_path / "point_cloud"
    iterations: list[int] = []
    for child in point_cloud_dir.glob("iteration_*"):
        if child.is_dir():
            try:
                iterations.append(int(child.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    if not iterations:
        raise FileNotFoundError(f"No iteration_* directories under {point_cloud_dir}")
    return max(iterations)


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, int | None]:
    if args.checkpoint_path:
        checkpoint = Path(args.checkpoint_path).expanduser().resolve()
        return checkpoint, args.iteration

    model_path = Path(args.model_path).expanduser().resolve()
    iteration = args.iteration if args.iteration is not None else latest_iteration(model_path)
    checkpoint = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    return checkpoint, iteration


def sh_dim_for_degree(degree: int) -> int:
    if degree == 0:
        return 0
    if degree == 1:
        return 3
    if degree == 2:
        return 8
    if degree == 3:
        return 15
    raise ValueError(f"SPZ v3 supports SH degree 0..3, got {degree}")


def infer_sh_degree(features_dc: torch.Tensor, features_rest: torch.Tensor) -> int:
    coeff_count = features_dc.shape[1] + features_rest.shape[1]
    degree = int(round(math.sqrt(coeff_count) - 1))
    if (degree + 1) ** 2 != coeff_count:
        raise ValueError(f"Cannot infer SH degree from {coeff_count} coefficients")
    if degree > 3:
        raise ValueError(f"SPZ v3 writer supports SH degree <= 3, got {degree}")
    return degree


def chunk_slices(count: int, chunk_size: int):
    for start in range(0, count, chunk_size):
        yield slice(start, min(start + chunk_size, count))


def safe_normalize(v: torch.Tensor, fallback: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    fallback = fallback.to(device=v.device, dtype=v.dtype).expand_as(v)
    return torch.where(norm > eps, v / norm.clamp_min(eps), fallback)


def matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert right-handed 3x3 rotation matrices to w, x, y, z quaternions."""
    m00 = matrix[:, 0, 0]
    m01 = matrix[:, 0, 1]
    m02 = matrix[:, 0, 2]
    m10 = matrix[:, 1, 0]
    m11 = matrix[:, 1, 1]
    m12 = matrix[:, 1, 2]
    m20 = matrix[:, 2, 0]
    m21 = matrix[:, 2, 1]
    m22 = matrix[:, 2, 2]

    qw = torch.empty_like(m00)
    qx = torch.empty_like(m00)
    qy = torch.empty_like(m00)
    qz = torch.empty_like(m00)

    trace = m00 + m11 + m22
    mask = trace > 0
    s = torch.sqrt(torch.clamp(trace[mask] + 1.0, min=1e-12)) * 2.0
    qw[mask] = 0.25 * s
    qx[mask] = (m21[mask] - m12[mask]) / s
    qy[mask] = (m02[mask] - m20[mask]) / s
    qz[mask] = (m10[mask] - m01[mask]) / s

    mask_x = (~mask) & (m00 > m11) & (m00 > m22)
    s = torch.sqrt(torch.clamp(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x], min=1e-12)) * 2.0
    qw[mask_x] = (m21[mask_x] - m12[mask_x]) / s
    qx[mask_x] = 0.25 * s
    qy[mask_x] = (m01[mask_x] + m10[mask_x]) / s
    qz[mask_x] = (m02[mask_x] + m20[mask_x]) / s

    mask_y = (~mask) & (~mask_x) & (m11 > m22)
    s = torch.sqrt(torch.clamp(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y], min=1e-12)) * 2.0
    qw[mask_y] = (m02[mask_y] - m20[mask_y]) / s
    qx[mask_y] = (m01[mask_y] + m10[mask_y]) / s
    qy[mask_y] = 0.25 * s
    qz[mask_y] = (m12[mask_y] + m21[mask_y]) / s

    mask_z = (~mask) & (~mask_x) & (~mask_y)
    s = torch.sqrt(torch.clamp(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z], min=1e-12)) * 2.0
    qw[mask_z] = (m10[mask_z] - m01[mask_z]) / s
    qx[mask_z] = (m02[mask_z] + m20[mask_z]) / s
    qy[mask_z] = (m12[mask_z] + m21[mask_z]) / s
    qz[mask_z] = 0.25 * s

    quat = torch.stack((qw, qx, qy, qz), dim=1)
    quat = quat / torch.linalg.norm(quat, dim=1, keepdim=True).clamp_min(1e-12)
    quat = torch.where(quat[:, :1] < 0, -quat, quat)
    return quat


def triangle_gaussian_geometry(
    points: torch.Tensor,
    sigma_raw: torch.Tensor,
    min_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centers, log scales, quats wxyz, and quats xyzw as numpy arrays."""
    points = points.to(torch.float32)
    p0 = points[:, 0]
    p1 = points[:, 1]
    p2 = points[:, 2]
    centers = points.mean(dim=1)

    e0 = p1 - p0
    e1 = p2 - p0
    fallback_x = torch.tensor([1.0, 0.0, 0.0], dtype=points.dtype, device=points.device)
    u = safe_normalize(e0, fallback_x)

    raw_n = torch.cross(e0, e1, dim=1)
    tmp_z = torch.tensor([0.0, 0.0, 1.0], dtype=points.dtype, device=points.device)
    tmp_y = torch.tensor([0.0, 1.0, 0.0], dtype=points.dtype, device=points.device)
    tmp = torch.where(torch.abs(u[:, 2:3]) < 0.9, tmp_z.expand_as(u), tmp_y.expand_as(u))
    fallback_n = safe_normalize(torch.cross(u, tmp, dim=1), tmp_y)
    n = safe_normalize(raw_n, fallback_n)
    v = safe_normalize(torch.cross(n, u, dim=1), tmp_y)
    n = safe_normalize(torch.cross(u, v, dim=1), fallback_n)

    rel = points - centers[:, None, :]
    radius_u = torch.max(torch.abs(torch.sum(rel * u[:, None, :], dim=2)), dim=1).values
    radius_v = torch.max(torch.abs(torch.sum(rel * v[:, None, :], dim=2)), dim=1).values
    sigma = 0.01 + torch.exp(sigma_raw.reshape(-1).to(torch.float32))

    scales = torch.stack((radius_u, radius_v, sigma), dim=1).clamp_min(min_scale)
    scale_log = torch.log(scales)

    rotation = torch.stack((u, v, n), dim=2)
    quat_wxyz = matrix_to_quaternion_wxyz(rotation)
    quat_xyzw = torch.stack(
        (quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]),
        dim=1,
    )

    return (
        centers.detach().cpu().numpy().astype(np.float32, copy=False),
        scale_log.detach().cpu().numpy().astype(np.float32, copy=False),
        quat_wxyz.detach().cpu().numpy().astype(np.float32, copy=False),
        quat_xyzw.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def chunk_rgb_u8(
    points: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    sh_degree: int,
) -> np.ndarray:
    features = torch.cat((features_dc, features_rest), dim=1)
    shs = features.permute(0, 2, 1).contiguous()
    centers = points.mean(dim=1)
    dirs_norm = centers / torch.linalg.norm(centers, dim=1, keepdim=True).clamp_min(1e-12)
    rgb = eval_sh(sh_degree, shs, dirs_norm)
    colors_f = torch.nan_to_num(torch.clamp(rgb + 0.5, 0.0, 1.0))
    return (colors_f * 255).to(torch.uint8).detach().cpu().numpy()


def write_triangle_ply(
    path: Path,
    sd: dict,
    count: int,
    sh_degree: int,
    chunk_size: int,
) -> dict:
    vertex_count = count * 3
    face_count = count
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment triangle_splatting\n"
        "comment generated_by scripts/export_ply_spz.py\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar alpha\n"
        "property float sigma\n"
        f"element face {face_count}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
            ("sigma", "<f4"),
        ]
    )
    face_dtype = np.dtype([("count", "u1"), ("vertex_indices", "<i4", (3,))])

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="triangle ply vertices"):
            points = sd["triangles_points"][sl]
            rgb = chunk_rgb_u8(points, sd["features_dc"][sl], sd["features_rest"][sl], sh_degree)
            alpha = (torch.sigmoid(sd["opacity"][sl].reshape(-1)).clamp(0.0, 1.0) * 255).to(torch.uint8)
            sigma = 0.01 + torch.exp(sd["sigma"][sl].reshape(-1).to(torch.float32))

            flat_points = points.reshape(-1, 3).detach().cpu().numpy().astype(np.float32, copy=False)
            vertex_data = np.empty(flat_points.shape[0], dtype=vertex_dtype)
            vertex_data["x"] = flat_points[:, 0]
            vertex_data["y"] = flat_points[:, 1]
            vertex_data["z"] = flat_points[:, 2]
            vertex_data["red"] = np.repeat(rgb[:, 0], 3)
            vertex_data["green"] = np.repeat(rgb[:, 1], 3)
            vertex_data["blue"] = np.repeat(rgb[:, 2], 3)
            vertex_data["alpha"] = np.repeat(alpha.detach().cpu().numpy(), 3)
            vertex_data["sigma"] = np.repeat(sigma.detach().cpu().numpy().astype(np.float32, copy=False), 3)
            f.write(vertex_data.tobytes())

        for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="triangle ply faces"):
            start = sl.start or 0
            stop = sl.stop or count
            base = np.arange(start * 3, stop * 3, 3, dtype=np.int32)
            face_data = np.empty(base.shape[0], dtype=face_dtype)
            face_data["count"] = 3
            face_data["vertex_indices"][:, 0] = base
            face_data["vertex_indices"][:, 1] = base + 1
            face_data["vertex_indices"][:, 2] = base + 2
            f.write(face_data.tobytes())

    return {"vertices": vertex_count, "faces": face_count, "bytes": path.stat().st_size}


def gaussian_ply_dtype(sh_rest_count: int) -> np.dtype:
    fields: list[tuple[str, str]] = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
    ]
    fields.extend((f"f_rest_{idx}", "<f4") for idx in range(sh_rest_count))
    fields.extend(
        [
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    return np.dtype(fields)


def write_gaussian_ply(
    path: Path,
    sd: dict,
    count: int,
    sh_degree: int,
    chunk_size: int,
    min_scale: float,
) -> dict:
    rest_count = sh_dim_for_degree(sh_degree) * 3
    dtype = gaussian_ply_dtype(rest_count)
    property_lines = []
    for name in dtype.names or ():
        property_lines.append(f"property float {name}")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment triangle_splatting_approx_gaussian\n"
        "comment one_gaussian_per_triangle_centroid\n"
        f"element vertex {count}\n"
        + "\n".join(property_lines)
        + "\nend_header\n"
    )

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="gaussian ply"):
            centers, scale_log, quat_wxyz, _ = triangle_gaussian_geometry(
                sd["triangles_points"][sl],
                sd["sigma"][sl],
                min_scale,
            )
            f_dc = sd["features_dc"][sl, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)
            rest = sd["features_rest"][sl, : sh_dim_for_degree(sh_degree), :]
            rest_flat = rest.detach().cpu().numpy().reshape(centers.shape[0], -1).astype(np.float32, copy=False)
            opacity = sd["opacity"][sl].reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)

            data = np.empty(centers.shape[0], dtype=dtype)
            data["x"] = centers[:, 0]
            data["y"] = centers[:, 1]
            data["z"] = centers[:, 2]
            data["nx"] = 0.0
            data["ny"] = 0.0
            data["nz"] = 0.0
            data["f_dc_0"] = f_dc[:, 0]
            data["f_dc_1"] = f_dc[:, 1]
            data["f_dc_2"] = f_dc[:, 2]
            for idx in range(rest_count):
                data[f"f_rest_{idx}"] = rest_flat[:, idx]
            data["opacity"] = opacity
            data["scale_0"] = scale_log[:, 0]
            data["scale_1"] = scale_log[:, 1]
            data["scale_2"] = scale_log[:, 2]
            data["rot_0"] = quat_wxyz[:, 0]
            data["rot_1"] = quat_wxyz[:, 1]
            data["rot_2"] = quat_wxyz[:, 2]
            data["rot_3"] = quat_wxyz[:, 3]
            f.write(data.tobytes())

    return {"vertices": count, "faces": 0, "bytes": path.stat().st_size}


def to_u8(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)


def scan_centroid_bounds(
    sd: dict,
    count: int,
    chunk_size: int,
) -> dict:
    min_xyz = np.full(3, np.inf, dtype=np.float64)
    max_xyz = np.full(3, -np.inf, dtype=np.float64)
    max_abs = 0.0
    for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="spz bounds"):
        centers = sd["triangles_points"][sl].mean(dim=1).detach().cpu().numpy().astype(np.float64, copy=False)
        min_xyz = np.minimum(min_xyz, centers.min(axis=0))
        max_xyz = np.maximum(max_xyz, centers.max(axis=0))
        max_abs = max(max_abs, float(np.max(np.abs(centers))))
    return {
        "min": min_xyz.tolist(),
        "max": max_xyz.tolist(),
        "max_abs": max_abs,
    }


def choose_fractional_bits(max_abs: float, max_fractional_bits: int = SPZ_DEFAULT_MAX_FRACTIONAL_BITS) -> int:
    if max_abs <= 0.0:
        return max_fractional_bits
    bits = math.floor(math.log2(SPZ_MAX_FIXED / max_abs))
    return max(0, min(max_fractional_bits, bits))


def parse_fractional_bits(value: str, bounds: dict) -> int:
    if value == "auto":
        return choose_fractional_bits(float(bounds["max_abs"]))
    bits = int(value)
    if bits < 0 or bits > 23:
        raise ValueError("--spz-fractional-bits must be auto or an integer in 0..23")
    return bits


def pack_positions_24bit(positions: np.ndarray, fractional_bits: int) -> tuple[np.ndarray, int]:
    position_scale = 1 << fractional_bits
    fixed = np.rint(positions.astype(np.float64) * position_scale).astype(np.int64)
    clipped = np.clip(fixed, SPZ_MIN_FIXED, SPZ_MAX_FIXED)
    clipped_count = int(np.count_nonzero(fixed != clipped))
    fixed24 = (clipped.astype(np.int64) & 0xFFFFFF).astype(np.uint32)
    out = np.empty((positions.shape[0], 3, 3), dtype=np.uint8)
    out[:, :, 0] = fixed24 & 0xFF
    out[:, :, 1] = (fixed24 >> 8) & 0xFF
    out[:, :, 2] = (fixed24 >> 16) & 0xFF
    return out.reshape(-1), clipped_count


def pack_quaternion_smallest_three_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    q = quat_xyzw.astype(np.float64, copy=False)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(norm > 1e-12, q / np.maximum(norm, 1e-12), np.array([0.0, 0.0, 0.0, 1.0]))
    i_largest = np.argmax(np.abs(q), axis=1).astype(np.uint32)
    largest = q[np.arange(q.shape[0]), i_largest]
    negate = largest < 0.0

    all_idx = np.broadcast_to(np.arange(4), q.shape)
    order = all_idx[all_idx != i_largest[:, None]].reshape(q.shape[0], 3)
    vals = np.take_along_axis(q, order, axis=1)
    negbits = np.logical_xor(vals < 0.0, negate[:, None]).astype(np.uint32)
    mags = np.floor(((1 << 9) - 1) * (np.abs(vals) / math.sqrt(0.5)) + 0.5)
    mags = np.minimum(mags.astype(np.uint32), (1 << 9) - 1)

    comp = i_largest
    for idx in range(3):
        comp = (comp << 10) | (negbits[:, idx] << 9) | mags[:, idx]

    out = np.empty((q.shape[0], 4), dtype=np.uint8)
    out[:, 0] = comp & 0xFF
    out[:, 1] = (comp >> 8) & 0xFF
    out[:, 2] = (comp >> 16) & 0xFF
    out[:, 3] = (comp >> 24) & 0xFF
    return out.reshape(-1)


def quantize_sh(rest_flat: np.ndarray) -> np.ndarray:
    steps = np.full(rest_flat.shape[1], 16, dtype=np.int32)
    steps[: min(9, rest_flat.shape[1])] = 8
    q = np.rint(np.nan_to_num(rest_flat, nan=0.0, posinf=0.0, neginf=0.0) * 128.0 + 128.0).astype(np.int32)
    q = (q // steps[None, :]) * steps[None, :]
    return np.clip(q, 0, 255).astype(np.uint8).reshape(-1)


def write_spz_v3(
    path: Path,
    sd: dict,
    count: int,
    sh_degree: int,
    chunk_size: int,
    min_scale: float,
    fractional_bits: int,
) -> dict:
    sh_dim = sh_dim_for_degree(sh_degree)
    clipped_positions = 0
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(struct.pack("<iiiBBBB", SPZ_MAGIC, SPZ_VERSION, count, sh_degree, fractional_bits, 0, 0))

        for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="spz positions"):
            centers, _, _, _ = triangle_gaussian_geometry(sd["triangles_points"][sl], sd["sigma"][sl], min_scale)
            packed, clipped = pack_positions_24bit(centers, fractional_bits)
            clipped_positions += clipped
            f.write(packed.tobytes())

        for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="spz alpha"):
            alpha = torch.sigmoid(sd["opacity"][sl].reshape(-1)).clamp(0.0, 1.0)
            f.write(to_u8(alpha.detach().cpu().numpy() * 255.0).tobytes())

        for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="spz color"):
            f_dc = sd["features_dc"][sl, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)
            f.write(to_u8(np.nan_to_num(f_dc, nan=0.0, posinf=0.0, neginf=0.0) * (COLOR_SCALE * 255.0) + 127.5).tobytes())

        for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="spz scale"):
            _, scale_log, _, _ = triangle_gaussian_geometry(sd["triangles_points"][sl], sd["sigma"][sl], min_scale)
            f.write(to_u8((scale_log + 10.0) * 16.0).tobytes())

        for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="spz rotation"):
            _, _, _, quat_xyzw = triangle_gaussian_geometry(sd["triangles_points"][sl], sd["sigma"][sl], min_scale)
            f.write(pack_quaternion_smallest_three_xyzw(quat_xyzw).tobytes())

        if sh_dim:
            for sl in tqdm(list(chunk_slices(count, chunk_size)), desc="spz sh"):
                rest = sd["features_rest"][sl, :sh_dim, :].detach().cpu().numpy()
                f.write(quantize_sh(rest.reshape(rest.shape[0], -1)).tobytes())

    return {
        "points": count,
        "sh_degree": sh_degree,
        "version": SPZ_VERSION,
        "fractional_bits": fractional_bits,
        "position_values_clipped": clipped_positions,
        "bytes": path.stat().st_size,
    }


def parse_ply_header(path: Path) -> dict:
    header_lines: list[str] = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header missing end_header: {path}")
            text = line.decode("ascii", errors="replace").rstrip("\n")
            header_lines.append(text)
            if text == "end_header":
                break

    elements: dict[str, int] = {}
    properties: list[str] = []
    for line in header_lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            elements[parts[1]] = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property":
            properties.append(parts[-1])
    return {
        "format": next((line for line in header_lines if line.startswith("format ")), ""),
        "elements": elements,
        "properties": properties,
        "header_lines": len(header_lines),
    }


def validate_spz_v3(path: Path, expected_count: int, sh_degree: int, expected_fractional_bits: int) -> dict:
    sh_dim = sh_dim_for_degree(sh_degree)
    expected_size = 16 + expected_count * (9 + 1 + 3 + 3 + 4 + sh_dim * 3)
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError("SPZ gzip payload is shorter than the header")
        magic, version, count, degree, fractional_bits, flags, reserved = struct.unpack("<iiiBBBB", header)
        total = 16
        while True:
            data = f.read(1024 * 1024 * 8)
            if not data:
                break
            total += len(data)

    ok = (
        magic == SPZ_MAGIC
        and version == SPZ_VERSION
        and count == expected_count
        and degree == sh_degree
        and fractional_bits == expected_fractional_bits
        and total == expected_size
    )
    return {
        "ok": ok,
        "magic": hex(magic),
        "version": version,
        "points": count,
        "sh_degree": degree,
        "fractional_bits": fractional_bits,
        "flags": flags,
        "reserved": reserved,
        "decompressed_bytes": total,
        "expected_decompressed_bytes": expected_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Model output directory containing point_cloud/iteration_*")
    parser.add_argument("--checkpoint-path", help="Direct path to point_cloud_state_dict.pt")
    parser.add_argument("--iteration", type=int, help="Iteration number to export")
    parser.add_argument("--output-dir", help="Directory for exported artifacts")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--limit", type=int, help="Export only the first N triangles for smoke testing")
    parser.add_argument("--min-scale", type=float, default=1e-4)
    parser.add_argument("--skip-triangle-ply", action="store_true")
    parser.add_argument("--skip-gaussian-ply", action="store_true")
    parser.add_argument("--skip-spz", action="store_true")
    parser.add_argument(
        "--spz-fractional-bits",
        default="auto",
        help="SPZ fixed-point fractional bits, or auto to fit the scene bounds without clipping",
    )
    args = parser.parse_args()

    if not args.model_path and not args.checkpoint_path:
        parser.error("--model-path or --checkpoint-path is required")

    start = time.monotonic()
    start_utc = utc_now()
    checkpoint, iteration = resolve_checkpoint(args)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else checkpoint.parents[2]
    iteration_label = str(iteration) if iteration is not None else checkpoint.parent.name.replace("iteration_", "")
    suffix = f"it{iteration_label}"
    if args.limit:
        suffix += f"_first{args.limit}"

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else model_path / "exports" / f"iteration_{iteration_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"START_UTC={start_utc}")
    print(f"checkpoint={checkpoint}")
    print(f"output_dir={output_dir}")

    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    total_count = int(sd["triangles_points"].shape[0])
    count = min(total_count, args.limit) if args.limit else total_count
    sh_degree = infer_sh_degree(sd["features_dc"], sd["features_rest"])
    print(f"triangles={count} total_checkpoint_triangles={total_count} sh_degree={sh_degree}")

    centroid_bounds = scan_centroid_bounds(sd, count, args.chunk_size) if not args.skip_spz else None
    spz_fractional_bits = parse_fractional_bits(args.spz_fractional_bits, centroid_bounds) if centroid_bounds else None
    if spz_fractional_bits is not None:
        print(f"spz_fractional_bits={spz_fractional_bits} centroid_max_abs={centroid_bounds['max_abs']:.6f}")

    manifest: dict = {
        "started_utc": start_utc,
        "checkpoint": str(checkpoint),
        "model_path": str(model_path),
        "iteration": iteration,
        "total_checkpoint_triangles": total_count,
        "exported_triangles": count,
        "sh_degree": sh_degree,
        "chunk_size": args.chunk_size,
        "min_scale": args.min_scale,
        "spz_fractional_bits": spz_fractional_bits,
        "centroid_bounds": centroid_bounds,
        "artifacts": {},
        "validation": {},
        "notes": [
            "Triangle PLY preserves triangle primitives and includes face topology.",
            "Gaussian PLY and SPZ are approximate: one Gaussian is generated per triangle centroid.",
            "SPZ v3 is used because it is a compact Niantic-compatible gzip layout and avoids the current PlayCanvas CLI native WebGPU/glibc dependency in this environment.",
            "Known pitfalls checked: Niantic spz issues mention quality/rotation/color quantization and fragile PLY parsing; PlayCanvas issues mention unsupported non-3DGS PLY variants and large-SPZ regressions; Triangle Splatting issues point to create_off.py/train_game_engine.py as the official game-engine export direction.",
        ],
    }

    if not args.skip_triangle_ply:
        triangle_path = output_dir / f"triangles_{suffix}.ply"
        print(f"writing {triangle_path}")
        manifest["artifacts"]["triangle_ply"] = {
            "path": str(triangle_path),
            **write_triangle_ply(triangle_path, sd, count, sh_degree, args.chunk_size),
        }
        header = parse_ply_header(triangle_path)
        expected = {"vertex": count * 3, "face": count}
        manifest["validation"]["triangle_ply"] = {"ok": header["elements"] == expected, **header}

    if not args.skip_gaussian_ply:
        gaussian_path = output_dir / f"gaussian_splats_approx_{suffix}.ply"
        print(f"writing {gaussian_path}")
        manifest["artifacts"]["gaussian_ply"] = {
            "path": str(gaussian_path),
            **write_gaussian_ply(gaussian_path, sd, count, sh_degree, args.chunk_size, args.min_scale),
        }
        header = parse_ply_header(gaussian_path)
        manifest["validation"]["gaussian_ply"] = {
            "ok": header["elements"] == {"vertex": count},
            **header,
        }

    if not args.skip_spz:
        spz_path = output_dir / f"gaussian_splats_approx_{suffix}.spz"
        print(f"writing {spz_path}")
        manifest["artifacts"]["spz"] = {
            "path": str(spz_path),
            **write_spz_v3(
                spz_path,
                sd,
                count,
                sh_degree,
                args.chunk_size,
                args.min_scale,
                spz_fractional_bits,
            ),
        }
        manifest["validation"]["spz"] = validate_spz_v3(spz_path, count, sh_degree, spz_fractional_bits)

    end_utc = utc_now()
    elapsed = time.monotonic() - start
    manifest["ended_utc"] = end_utc
    manifest["elapsed_seconds"] = round(elapsed, 3)

    manifest_path = output_dir / f"export_manifest_{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    print(f"END_UTC={end_utc}")
    print(f"ELAPSED_SECONDS={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
