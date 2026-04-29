"""PG-SAM / MMA-SAM2 reproducible inference entry point.

This script is intentionally thin: it reuses the repository implementation in
``map_sam2`` instead of duplicating model code. The default command runs the
included sample images and writes only the final segmentation masks.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CONFIG = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_PGSAM_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "pgsam_best_model.pt"
DEFAULT_IMAGE_DIR = PACKAGE_ROOT / "data" / "images"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "inference_run"
SAM2_DOWNLOAD_URL = "https://huggingface.co/facebook/sam2.1-hiera-large"
PROJECT_MODULES = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PG-SAM/MMA-SAM2 inference. With no image argument, the bundled "
            "sample images in data/images are used."
        )
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=str, default=None, help="Path to one input image.")
    inputs.add_argument("--image-dir", type=str, default=None, help="Directory containing input images.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for inference outputs.")
    parser.add_argument(
        "--pgsam-checkpoint",
        "--mma-checkpoint",
        dest="pgsam_checkpoint",
        type=str,
        default=os.getenv("PGSAM_CHECKPOINT", str(DEFAULT_PGSAM_CHECKPOINT)),
        help="Fine-tuned PG-SAM/MMA-SAM2 checkpoint, e.g. best_model.pt.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default=os.getenv("SAM2_CHECKPOINT", str(DEFAULT_SAM2_CHECKPOINT)),
        help="Original SAM2.1 Hiera-L checkpoint.",
    )
    parser.add_argument("--config", type=str, default=None, help="SAM2 config yaml.")
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu. Defaults to CUDA if available.")
    parser.add_argument("--image-size", type=int, default=None, help="Model input size. Defaults to checkpoint value.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for final masks.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images to process.")
    parser.add_argument("--bf16", dest="bf16", action="store_true", help="Use BF16 autocast on CUDA when supported.")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="Disable BF16 autocast.")
    parser.set_defaults(bf16=True)

    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--refine-attn-dim", type=int, default=None)
    parser.add_argument("--refine-downsample-ratio", type=int, default=None)
    parser.add_argument("--refine-iter", type=int, default=None)
    parser.add_argument("--fullres-blocks", type=int, default=None)
    parser.add_argument("--fullres-hidden-dim", type=int, default=None)

    parser.add_argument("--post-close-radius", type=int, default=None)
    parser.add_argument("--post-min-area-ratio", type=float, default=None)
    parser.add_argument("--post-boundary-band-radius", type=int, default=None)
    parser.add_argument("--post-boundary-prob-threshold", type=float, default=None)
    parser.add_argument("--edge-search-radius", type=int, default=None)
    return parser.parse_args()


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_project_modules():
    """Import project modules after CLI parsing so --help works without SAM2 deps."""
    global PROJECT_MODULES
    if PROJECT_MODULES is not None:
        return PROJECT_MODULES
    try:
        from map_sam2.geometry import edge_snapping_postprocess, postprocess_main_region
        from map_sam2.inference import forward_mapsam
        from map_sam2.modeling import LoRALinear, build_trainable_modules
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise ModuleNotFoundError(
            f"Missing Python dependency or project module: {missing}\n"
            "Install the local package and dependencies from this release folder:\n"
            "  pip install -e .\n"
            "  pip install -r requirements.txt"
        ) from exc
    PROJECT_MODULES = SimpleNamespace(
        LoRALinear=LoRALinear,
        build_trainable_modules=build_trainable_modules,
        edge_snapping_postprocess=edge_snapping_postprocess,
        forward_mapsam=forward_mapsam,
        postprocess_main_region=postprocess_main_region,
    )
    return PROJECT_MODULES


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        extra = ""
        if path.name == "sam2.1_hiera_large.pt":
            extra = (
                f"\nDownload SAM2.1 Hiera-L from:\n  {SAM2_DOWNLOAD_URL}\n"
                f"and save it as:\n  {DEFAULT_SAM2_CHECKPOINT}"
            )
        raise FileNotFoundError(
            f"{label} was not found: {path}\n"
            f"Please place the file there or pass the correct path with the matching command-line option."
            f"{extra}"
        )


def get_ckpt_args(checkpoint: dict) -> dict:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def pick(cli_value, ckpt_args: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(key, default)


def build_model_args(args: argparse.Namespace, ckpt_args: dict) -> SimpleNamespace:
    config = args.config or ckpt_args.get("config") or DEFAULT_CONFIG
    return SimpleNamespace(
        config=config,
        checkpoint=str(Path(args.sam2_checkpoint).resolve()),
        use_lora=bool(ckpt_args.get("use_lora", True)),
        lora_rank=int(pick(args.lora_rank, ckpt_args, "lora_rank", 16)),
        lora_alpha=float(pick(args.lora_alpha, ckpt_args, "lora_alpha", 32.0)),
        prompt_source=str(ckpt_args.get("prompt_source", "sgn_auto")),
        refine_attn_dim=int(pick(args.refine_attn_dim, ckpt_args, "refine_attn_dim", 64)),
        refine_downsample_ratio=int(pick(args.refine_downsample_ratio, ckpt_args, "refine_downsample_ratio", 2)),
        refine_use_checkpoint=False,
        fullres_blocks=int(pick(args.fullres_blocks, ckpt_args, "fullres_blocks", 5)),
        fullres_hidden_dim=int(pick(args.fullres_hidden_dim, ckpt_args, "fullres_hidden_dim", 48)),
        use_decoder_reconstruction=bool(ckpt_args.get("use_decoder_reconstruction", True)),
        highres_feature_mode=str(ckpt_args.get("highres_feature_mode", "stage1_stage2")),
        use_boundary_refinement=bool(ckpt_args.get("use_boundary_refinement", True)),
        use_fullres_refinement=bool(ckpt_args.get("use_fullres_refinement", True)),
    )


def load_lora_state(module: torch.nn.Module, lora_state: dict, device: torch.device, lora_cls: type[torch.nn.Module]) -> None:
    if not lora_state:
        return
    for name, submodule in module.named_modules():
        if not isinstance(submodule, lora_cls):
            continue
        key_a = f"{name}.lora_A"
        key_b = f"{name}.lora_B"
        if key_a in lora_state:
            submodule.lora_A.data.copy_(lora_state[key_a].to(device=device, dtype=submodule.lora_A.dtype))
        if key_b in lora_state:
            submodule.lora_B.data.copy_(lora_state[key_b].to(device=device, dtype=submodule.lora_B.dtype))


def load_pgsam_checkpoint(
    checkpoint: dict,
    model: torch.nn.Module,
    sgn: torch.nn.Module,
    refiner: torch.nn.Module,
    full_res_refiner: torch.nn.Module,
    device: torch.device,
    lora_cls: type[torch.nn.Module],
) -> None:
    load_lora_state(model.image_encoder.trunk, checkpoint.get("lora", {}), device, lora_cls=lora_cls)
    if "sgn" in checkpoint:
        sgn.load_state_dict(checkpoint["sgn"], strict=False)
    if "mask_decoder" in checkpoint:
        model.sam_mask_decoder.load_state_dict(checkpoint["mask_decoder"], strict=False)
    if "refiner" in checkpoint:
        refiner.load_state_dict(checkpoint["refiner"], strict=False)
    if "full_res_refiner" in checkpoint:
        full_res_refiner.load_state_dict(checkpoint["full_res_refiner"], strict=False)


def collect_images(image: str | None, image_dir: str | None, limit: int | None) -> list[Path]:
    if image:
        paths = [Path(image)]
    else:
        root = Path(image_dir) if image_dir else DEFAULT_IMAGE_DIR
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory was not found: {root}")
        paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        paths = paths[: max(0, int(limit))]
    if not paths:
        raise RuntimeError("No input images were found.")
    for path in paths:
        require_file(path, "Input image")
    return paths


def load_image(path: Path, image_size: int, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    with Image.open(path) as image_in:
        original = image_in.convert("RGB")
    resized = original.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return tensor, original


def resize_mask(mask: torch.Tensor, original_size: tuple[int, int]) -> np.ndarray:
    mask = mask.detach().cpu().float()
    mask = F.interpolate(mask, size=(original_size[1], original_size[0]), mode="nearest")
    return ((mask[0, 0].numpy() > 0.5).astype(np.uint8) * 255)


def progress(paths: list[Path]):
    try:
        from tqdm import tqdm

        return tqdm(paths, desc="PG-SAM inference", ascii=True)
    except Exception:
        return paths


def should_use_bf16(device: torch.device, requested: bool) -> bool:
    """Return whether BF16 autocast can be used safely on the selected device."""
    if not requested or device.type != "cuda":
        return False
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        print("[warning] BF16 is not supported by this CUDA device; using FP32 instead.")
        return False
    return True


def main() -> None:
    args = parse_args()
    try:
        modules = load_project_modules()
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc))
    pgsam_checkpoint = Path(args.pgsam_checkpoint)
    sam2_checkpoint = Path(args.sam2_checkpoint)
    require_file(pgsam_checkpoint, "PG-SAM checkpoint")
    require_file(sam2_checkpoint, "SAM2 checkpoint")

    checkpoint = torch_load(pgsam_checkpoint, map_location="cpu")
    ckpt_args = get_ckpt_args(checkpoint)
    image_size = int(args.image_size or ckpt_args.get("image_size", 1024))
    model_args = build_model_args(args, ckpt_args)
    if model_args.prompt_source != "sgn_auto":
        raise ValueError(
            "This release supports the PG-SAM automatic SGN prompt path only. "
            f"Checkpoint prompt_source={model_args.prompt_source!r} is unsupported."
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = should_use_bf16(device, args.bf16)

    model, sgn, refiner, full_res_refiner = modules.build_trainable_modules(model_args, device)
    if image_size != int(model.image_size):
        raise ValueError(f"--image-size={image_size} does not match SAM2 model.image_size={model.image_size}.")
    load_pgsam_checkpoint(checkpoint, model, sgn, refiner, full_res_refiner, device, lora_cls=modules.LoRALinear)
    model.eval()
    sgn.eval()
    refiner.eval()
    full_res_refiner.eval()

    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    post_cfg = {
        "post_close_radius": int(pick(args.post_close_radius, ckpt_args, "post_close_radius", 3)),
        "post_min_area_ratio": float(pick(args.post_min_area_ratio, ckpt_args, "post_min_area_ratio", 0.001)),
        "post_boundary_band_radius": int(pick(args.post_boundary_band_radius, ckpt_args, "post_boundary_band_radius", 2)),
        "post_boundary_prob_threshold": float(
            pick(args.post_boundary_prob_threshold, ckpt_args, "post_boundary_prob_threshold", 0.62)
        ),
        "edge_search_radius": int(pick(args.edge_search_radius, ckpt_args, "edge_search_radius", 2)),
    }
    refine_iter = int(pick(args.refine_iter, ckpt_args, "refine_iter", 2))
    image_paths = collect_images(args.image, args.image_dir, args.limit)

    print(f"[PG-SAM] Running inference on {len(image_paths)} image(s).")

    for image_path in progress(image_paths):
        image_tensor, original = load_image(image_path, image_size=image_size, device=device)
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_bf16,
                dtype=torch.bfloat16,
            ):
                logits, _, _, _ = modules.forward_mapsam(
                    model=model,
                    sgn=sgn,
                    refiner=refiner,
                    full_res_refiner=full_res_refiner,
                    images=image_tensor,
                    refine_iter=refine_iter,
                    use_decoder_reconstruction=model_args.use_decoder_reconstruction,
                    highres_feature_mode=model_args.highres_feature_mode,
                    use_boundary_refinement=model_args.use_boundary_refinement,
                    use_fullres_refinement=model_args.use_fullres_refinement,
            )
            prob = torch.sigmoid(logits.float())
            post_mask = modules.postprocess_main_region(
                prob,
                threshold=float(args.threshold),
                close_radius=post_cfg["post_close_radius"],
                min_area_ratio=post_cfg["post_min_area_ratio"],
                boundary_band_radius=post_cfg["post_boundary_band_radius"],
                boundary_prob_threshold=post_cfg["post_boundary_prob_threshold"],
            )
            post_mask = modules.edge_snapping_postprocess(
                post_mask,
                image=image_tensor,
                edge_search_radius=post_cfg["edge_search_radius"],
            )

        stem = image_path.stem
        mask_path = mask_dir / f"{stem}_mask.png"
        mask_u8 = resize_mask(post_mask, original.size)
        Image.fromarray(mask_u8, mode="L").save(mask_path)

    print(f"[PG-SAM] done. Final masks saved to: {mask_dir.resolve()}")


if __name__ == "__main__":
    main()
