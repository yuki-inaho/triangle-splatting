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

## Short Optuna tuning before a full run

For a fixed-pose scene, tune a small set of initialization and loss parameters
against the built-in holdout split before committing to a longer run.  The
runner always disables densification: this prevents a sparse or noisy initial
cloud from changing primitive count during comparisons.

```bash
pixi run python scripts/tune_triangle_splatting.py \
  --source /path/to/staged-scene \
  --output-dir /path/to/tuning-output \
  --tuning-seconds 1200 \
  --max-trials 32 \
  --trial-iterations 1000 \
  --confirmation-iterations 2000
```

`best_config.json` records the selected parameter set and its confirmation
metrics.  The per-trial logs and the confirmation checkpoints stay below the
specified output directory.  Copy the selected values into the full training
command, normally without `--eval` when every image should contribute to the
final reconstruction:

```bash
pixi run python train.py \
  -s /path/to/staged-scene \
  -m /path/to/final-model \
  --no_dome --iterations 8000 \
  --densify_from_iter 999999 --densify_until_iter 999999 \
  --set_sigma VALUE --triangle_size VALUE \
  --lr_triangles_points_init VALUE --lambda_dssim VALUE
```

TensorBoard event files are created in the model directory.  Inspect them with:

```bash
pixi run python -m tensorboard.main --logdir /path/to/model-or-parent-directory
```

It reports training loss, timing, primitive count, and—at every requested test
iteration—L1, PSNR, SSIM, and LPIPS.
