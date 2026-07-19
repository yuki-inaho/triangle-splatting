#!/usr/bin/env bash
set -euo pipefail

# The upstream kernel stores tile bounds as unsigned integers.  A projected
# triangle with no valid bound intersections can leave max < min; subtracting
# those values then underflows and requests an impossibly large binning buffer.
python - <<'PY'
from pathlib import Path

path = Path("submodules/diff-triangle-rasterization/cuda_rasterizer/forward.cu")
text = path.read_text()
old = """\tif ((rect_max_triangle_test.x - rect_min_triangle_test.x) * (rect_max_triangle_test.y - rect_min_triangle_test.y) == 0){
\t\tradii[idx] = 0;
\t\ttiles_touched[idx] = 0;
\t\tscaling[idx] = 0.0f;
\t\treturn;
\t}
"""
new = """\tif (rect_max_triangle_test.x <= rect_min_triangle_test.x ||
\t\trect_max_triangle_test.y <= rect_min_triangle_test.y) {
\t\tradii[idx] = 0;
\t\ttiles_touched[idx] = 0;
\t\tscaling[idx] = 0.0f;
\t\treturn;
\t}
"""

if new in text:
    raise SystemExit("binning guard is already present")
if old not in text:
    raise SystemExit(f"Could not find binning guard in {path}")
path.write_text(text.replace(old, new, 1))
print(f"patched {path}")
PY
