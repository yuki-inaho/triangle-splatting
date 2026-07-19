# Initializing Triangle Splatting from a COLMAP reconstruction

Triangle Splatting can initialize its triangles from an existing COLMAP sparse
model.  This keeps the supplied camera intrinsics, poses, and 3D point
coordinates unchanged; it does not run camera pose estimation.

The helper stages a model into the layout expected by `train.py` and converts
monochrome source images to three-channel RGB.  The latter is intentional:
Triangle Splatting trains against RGB tensors, so a monochrome modality must be
replicated into three channels before training.

```bash
pixi run python scripts/prepare_colmap_initialization.py \
  --images /path/to/images \
  --reconstruction /path/to/colmap-model-or-reconstruction-root \
  --output /path/to/staged-scene
```

By default, every COLMAP 3D point becomes an initial triangle.  If the initial
point cloud is too dense for the GPU rasterizer, keep the original COLMAP model
but deterministically select a bounded set of initial splat centers:

```bash
pixi run python scripts/prepare_colmap_initialization.py \
  --images /path/to/images \
  --reconstruction /path/to/colmap-model-or-reconstruction-root \
  --output /path/to/staged-scene \
  --max-points 5000
```

With `--max-points`, the staged `points3D.bin` remains a byte-for-byte copy of
the source reconstruction.  A selected `points3D.ply` takes precedence only
for Triangle Splatting initialization, and the manifest records the selection.

The output is:

```text
staged-scene/
├── images/                  # RGB images, with names from COLMAP images.bin
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── stage_manifest.json
```

Run a short, no-dome smoke test before a full optimization:

```bash
pixi run python train.py \
  -s /path/to/staged-scene \
  -m /path/to/model-output \
  --no_dome \
  --iterations 20 \
  --test_iterations 1000
```

Use `--no_dome` for indoor or bounded scans so no synthetic background points
are added.  Omit `--eval` for a reconstruction run that trains on every input
image.  Add `--eval` only when an eighth-image holdout split is wanted.

The loader supports COLMAP `PINHOLE` and `SIMPLE_PINHOLE` cameras.  The output
directory must be new or empty so the staging command cannot overwrite an
existing scene.

On an RTX 4090, build the CUDA extensions for compute capability 8.9 before
training:

```bash
TORCH_CUDA_ARCH_LIST=8.9 pixi run install-extensions
```

The Pixi tasks target compute capability 8.9 and include a guard for degenerate projected
triangles, which otherwise can cause the rasterizer to request an invalidly
large GPU binning buffer.
