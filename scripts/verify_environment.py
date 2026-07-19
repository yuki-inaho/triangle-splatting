import argparse
from importlib.metadata import PackageNotFoundError, version
import shutil
import subprocess


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extensions", action="store_true")
    args = parser.parse_args()

    import cv2
    import lpips
    import mediapy
    import numpy as np
    import open3d
    import plyfile
    import torch
    import torchvision

    print("python packages ok")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"numpy={np.__version__}")
    print(f"opencv={cv2.__version__}")
    print(f"open3d={open3d.__version__}")
    print(f"plyfile={package_version('plyfile')}")
    print(f"lpips={package_version('lpips')}")
    print(f"mediapy={package_version('mediapy')}")
    print(f"torch cuda available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(f"device capability={torch.cuda.get_device_capability(0)}")

    nvcc = shutil.which("nvcc")
    print(f"nvcc={nvcc}")
    if nvcc:
        out = subprocess.check_output([nvcc, "--version"], text=True)
        print(out.strip().splitlines()[-1])

    if args.extensions:
        import diff_triangle_rasterization
        import simple_knn._C

        print(f"diff_triangle_rasterization={diff_triangle_rasterization.__file__}")
        print(f"simple_knn._C={simple_knn._C.__file__}")


if __name__ == "__main__":
    main()
