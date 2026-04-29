"""Check bundled PG-SAM source files, input images, and optional checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = PACKAGE_ROOT / "data" / "images"
DEFAULT_PGSAM_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "pgsam_best_model.pt"
DEFAULT_SAM2_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_DOWNLOAD_URL = "https://huggingface.co/facebook/sam2.1-hiera-large"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_SOURCE_FILES = [
    PACKAGE_ROOT / "infer.py",
    PACKAGE_ROOT / "map_sam2" / "modeling.py",
    PACKAGE_ROOT / "map_sam2" / "inference.py",
    PACKAGE_ROOT / "sam2" / "build_sam.py",
    PACKAGE_ROOT / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the PG-SAM input images.")
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--check-weights", action="store_true", help="Also check the default SAM2 and PG-SAM weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok = True
    for path in REQUIRED_SOURCE_FILES:
        if path.is_file():
            print(f"[ok] source: {path.relative_to(PACKAGE_ROOT)}")
        else:
            print(f"[missing] source: {path}")
            ok = False

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"[missing] image directory: {image_dir}")
        ok = False
        images = []
    else:
        images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
        for path in images:
            print(f"[ok] image: {path.name}")
        if not images:
            print(f"[missing] no input images found in: {image_dir}")
            ok = False

    if args.check_weights:
        for label, path in (("SAM2 checkpoint", DEFAULT_SAM2_CHECKPOINT), ("PG-SAM checkpoint", DEFAULT_PGSAM_CHECKPOINT)):
            if path.is_file():
                print(f"[ok] {label}: {path}")
            else:
                print(f"[missing] {label}: {path}")
                if path == DEFAULT_SAM2_CHECKPOINT:
                    print(f"[hint] Download SAM2.1 Hiera-L from: {SAM2_DOWNLOAD_URL}")
                    print(f"[hint] Save it as: {DEFAULT_SAM2_CHECKPOINT}")
                ok = False

    if not ok:
        raise SystemExit(1)
    print(f"[ok] {len(images)} sample image(s) are available.")


if __name__ == "__main__":
    main()
