"""Inference-only PG-SAM V6.1 demo using the released model components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from map_sam2.geometry import edge_snapping_postprocess, postprocess_main_region  # noqa: E402
from map_sam2.inference import forward_pgsam  # noqa: E402
from map_sam2.modeling import LoRALinear, build_trainable_modules  # noqa: E402
from map_sam2.transforms import (  # noqa: E402
    LetterboxMeta,
    resize_pil_pair,
    restore_letterboxed_mask,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SAM2_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_PGSAM_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "pgsam_best_model.pt"
DEFAULT_IMAGE_DIR = PACKAGE_ROOT / "data" / "images"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs" / "inference_run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PG-SAM V6.1 inference demo")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=str, help="Path to one input image.")
    inputs.add_argument("--image-dir", type=str, help="Directory containing input images.")
    parser.add_argument("--config", type=str, default=None, help="Defaults to the value stored in the PGSAM checkpoint.")
    parser.add_argument("--sam2-checkpoint", type=str, default=str(DEFAULT_SAM2_CHECKPOINT),
                        help="Original SAM2.1 Hiera-L checkpoint; downloaded separately.")
    parser.add_argument("--pgsam-checkpoint", type=str, default=str(DEFAULT_PGSAM_CHECKPOINT),
                        help="Released PG-SAM V6.1 inference checkpoint.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to the value stored in the checkpoint.")
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"], default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-raw", action="store_true", help="Save raw masks before post-processing.")
    parser.add_argument("--save-prob", action="store_true", help="Save probability maps.")
    parser.add_argument("--save-overlay", action="store_true", help="Save simple mask overlays.")
    parser.add_argument("--bf16", dest="bf16", action="store_true")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)

    # Architecture defaults must match the trained PGSAM checkpoint.
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--refine-attn-dim", type=int, default=None)
    parser.add_argument("--refine-local-kernel", type=int, default=None)
    parser.add_argument("--refine-downsample-ratio", type=int, default=None)
    parser.add_argument("--refine-use-checkpoint", dest="refine_use_checkpoint", action="store_true")
    parser.add_argument("--no-refine-checkpoint", dest="refine_use_checkpoint", action="store_false")
    parser.set_defaults(refine_use_checkpoint=False)
    parser.add_argument("--refine-iter", type=int, default=None)
    parser.add_argument("--fullres-blocks", type=int, default=None)
    parser.add_argument("--fullres-hidden-dim", type=int, default=None)
    parser.add_argument("--fullres-block-type", choices=["standard", "depthwise"], default=None)
    parser.add_argument("--decoder-prompt-gate-floor", type=float, default=None)
    parser.add_argument("--multimask-output", dest="multimask_output", action="store_true")
    parser.add_argument("--single-mask-output", dest="multimask_output", action="store_false")
    parser.set_defaults(multimask_output=None)

    # Post-processing defaults match the paper evaluation protocol.
    parser.add_argument("--post-close-radius", type=int, default=None)
    parser.add_argument("--post-min-area-ratio", type=float, default=None)
    parser.add_argument("--post-boundary-band-radius", type=int, default=None)
    parser.add_argument("--post-boundary-prob-threshold", type=float, default=None)
    parser.add_argument("--edge-search-radius", type=int, default=None)
    return parser.parse_args()


def pick(cli_value, checkpoint_args: dict, key: str, default):
    return cli_value if cli_value is not None else checkpoint_args.get(key, default)


def make_model_args(args: argparse.Namespace, checkpoint_args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config or checkpoint_args.get("config") or "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint=args.sam2_checkpoint,
        use_lora=bool(checkpoint_args.get("use_lora", True)),
        lora_rank=int(pick(args.lora_rank, checkpoint_args, "lora_rank", 16)),
        lora_alpha=float(pick(args.lora_alpha, checkpoint_args, "lora_alpha", 32.0)),
        prompt_source=str(checkpoint_args.get("prompt_source", "sgn_auto")),
        highres_fusion_mode=str(checkpoint_args.get("highres_fusion_mode", "legacy_concat")),
        prompt_policy=str(checkpoint_args.get("prompt_policy", "legacy")),
        use_initial_implicit_prompt=checkpoint_args.get("use_initial_implicit_prompt"),
        use_final_implicit_prompt=checkpoint_args.get("use_final_implicit_prompt"),
        prompt_refinement_mode=checkpoint_args.get("prompt_refinement_mode", "global_local"),
        point_confidence_ratio=float(checkpoint_args.get("point_confidence_ratio", .6)),
        point_min_confidence=float(checkpoint_args.get("point_min_confidence", .5)),
        use_layout_prior=bool(checkpoint_args.get("use_layout_prior", False)),
        implicit_prompt_mode=str(checkpoint_args.get("implicit_prompt_mode", "residual")),
        refine_attn_dim=int(pick(args.refine_attn_dim, checkpoint_args, "refine_attn_dim", 64)),
        refine_local_kernel=int(pick(args.refine_local_kernel, checkpoint_args, "refine_local_kernel", 5)),
        refine_downsample_ratio=int(
            pick(args.refine_downsample_ratio, checkpoint_args, "refine_downsample_ratio", 2)
        ),
        refine_use_checkpoint=args.refine_use_checkpoint,
        fullres_blocks=int(pick(args.fullres_blocks, checkpoint_args, "fullres_blocks", 5)),
        fullres_hidden_dim=int(pick(args.fullres_hidden_dim, checkpoint_args, "fullres_hidden_dim", 48)),
        fullres_block_type=str(
            pick(args.fullres_block_type, checkpoint_args, "fullres_block_type", "standard")
        ),
        decoder_prompt_gate_floor=float(
            pick(args.decoder_prompt_gate_floor, checkpoint_args, "decoder_prompt_gate_floor", 0.0)
        ),
        sam2_normalize_inputs=bool(checkpoint_args.get("sam2_normalize_inputs", False)),
        use_decoder_reconstruction=bool(checkpoint_args.get("use_decoder_reconstruction", True)),
        highres_feature_mode=str(checkpoint_args.get("highres_feature_mode", "stage1_stage2")),
        use_boundary_refinement=bool(checkpoint_args.get("use_boundary_refinement", True)),
        use_fullres_refinement=bool(checkpoint_args.get("use_fullres_refinement", True)),
        num_prompt_points=int(checkpoint_args.get("num_prompt_points", 8)),
        global_prompt_points=int(checkpoint_args.get("global_prompt_points", 0)),
        prompt_constraint_mode=str(checkpoint_args.get("prompt_constraint_mode", "hard")),
        soft_constraint_floor=float(checkpoint_args.get("soft_constraint_floor", 0.25)),
        progressive_initial_weight=float(
            checkpoint_args.get("progressive_initial_weight", 0.5)
        ),
    )


def load_inference_checkpoint(
    checkpoint: dict,
    model: torch.nn.Module,
    sgn: torch.nn.Module,
    refiner: torch.nn.Module,
    full_res_refiner: torch.nn.Module,
    device: torch.device,
) -> None:
    """Load every trained component strictly; never silently use random weights."""
    expected = {"args", "lora", "sgn", "mask_decoder", "refiner", "full_res_refiner"}
    missing = expected.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Incomplete PG-SAM inference checkpoint: {sorted(missing)}")
    if not str(checkpoint.get("code_version", "")).startswith("PGSAM-v6.1"):
        raise RuntimeError("This demo requires a PG-SAM V6.1 checkpoint.")

    lora_state = checkpoint["lora"]
    for name, submodule in model.image_encoder.trunk.named_modules():
        if not isinstance(submodule, LoRALinear):
            continue
        for suffix in ("lora_A", "lora_B"):
            key = f"{name}.{suffix}"
            if key not in lora_state:
                raise RuntimeError(f"Missing LoRA parameter: {key}")
            parameter = getattr(submodule, suffix)
            parameter.data.copy_(lora_state[key].to(device=device, dtype=parameter.dtype))

    sgn.load_state_dict(checkpoint["sgn"], strict=True)
    model.sam_mask_decoder.load_state_dict(checkpoint["mask_decoder"], strict=True)
    refiner.load_state_dict(checkpoint["refiner"], strict=True)
    full_res_refiner.load_state_dict(checkpoint["full_res_refiner"], strict=True)


def collect_images(image: str | None, image_dir: str | None) -> List[Path]:
    if image:
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return [path]
    root = Path(image_dir) if image_dir else DEFAULT_IMAGE_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    images = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No supported images found in: {root}")
    return images


def load_image(
    path: Path,
    image_size: int,
    device: torch.device,
    resize_mode: str,
) -> Tuple[torch.Tensor, Image.Image, LetterboxMeta]:
    with Image.open(path) as image_in:
        original = image_in.convert("RGB")
    model_image, _, transform_meta = resize_pil_pair(
        original,
        target_size=image_size,
        resize_mode=resize_mode,
    )
    arr = np.asarray(model_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return tensor, original, transform_meta


def resize_mask(mask: torch.Tensor, transform_meta: LetterboxMeta) -> np.ndarray:
    mask = restore_letterboxed_mask(mask.detach().cpu(), transform_meta, mode="nearest")
    return ((mask[0, 0].numpy() > 0.5).astype(np.uint8) * 255)


def resize_prob(prob: torch.Tensor, transform_meta: LetterboxMeta) -> np.ndarray:
    prob = restore_letterboxed_mask(prob.detach().cpu(), transform_meta, mode="bilinear")
    return np.clip(prob[0, 0].numpy() * 255.0, 0.0, 255.0).astype(np.uint8)


def save_overlay(image: Image.Image, mask_u8: np.ndarray, path: Path) -> None:
    image_np = np.asarray(image, dtype=np.float32)
    color = np.array([255.0, 64.0, 64.0], dtype=np.float32)
    mask = (mask_u8 > 0)[..., None]
    overlay = np.where(mask, image_np * 0.55 + color * 0.45, image_np)
    Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8), mode="RGB").save(path)


def iter_progress(paths: Iterable[Path]) -> Iterable[Path]:
    try:
        from tqdm import tqdm

        return tqdm(list(paths), desc="infer")
    except Exception:
        return paths


def main() -> None:
    args = parse_args()
    pgsam_checkpoint_path = Path(args.pgsam_checkpoint)
    if not pgsam_checkpoint_path.is_file():
        raise FileNotFoundError(f"PGSAM checkpoint not found: {pgsam_checkpoint_path}")
    if not Path(args.sam2_checkpoint).is_file():
        raise FileNotFoundError(
            f"SAM2.1 base checkpoint not found: {args.sam2_checkpoint}\n"
            "Download it from https://huggingface.co/facebook/sam2.1-hiera-large"
        )
    checkpoint = torch.load(pgsam_checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("args"), dict):
        raise RuntimeError("The PG-SAM inference checkpoint is invalid.")
    checkpoint_args = checkpoint["args"]

    image_size = int(args.image_size or checkpoint_args.get("image_size", 1024))
    resize_mode = str(pick(args.resize_mode, checkpoint_args, "resize_mode", "letterbox"))
    refine_iter = int(pick(args.refine_iter, checkpoint_args, "refine_iter", 2))
    multimask_output = bool(
        pick(args.multimask_output, checkpoint_args, "multimask_output", False)
    )
    post_cfg = {
        "post_close_radius": int(pick(args.post_close_radius, checkpoint_args, "post_close_radius", 3)),
        "post_min_area_ratio": float(pick(args.post_min_area_ratio, checkpoint_args, "post_min_area_ratio", 0.001)),
        "post_boundary_band_radius": int(pick(args.post_boundary_band_radius, checkpoint_args, "post_boundary_band_radius", 0)),
        "post_boundary_prob_threshold": float(pick(args.post_boundary_prob_threshold, checkpoint_args, "post_boundary_prob_threshold", 0.62)),
        "edge_search_radius": int(pick(args.edge_search_radius, checkpoint_args, "edge_search_radius", 2)),
    }
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    raw_dir = output_dir / "raw_masks"
    prob_dir = output_dir / "probabilities"
    overlay_dir = output_dir / "overlays"
    meta_dir = output_dir / "metadata"
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
    if args.save_prob:
        prob_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        print("[warning] CPU inference is supported but slow.")
    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if args.bf16 and device.type == "cuda" and not use_bf16:
        print("[warning] BF16 is unsupported on this GPU; using FP32.")

    model_args = make_model_args(args, checkpoint_args)
    if model_args.prompt_source != "sgn_auto":
        raise ValueError(f"Unsupported prompt_source: {model_args.prompt_source}")
    model, sgn, refiner, full_res_refiner = build_trainable_modules(model_args, device)
    if image_size != int(model.image_size):
        raise ValueError(
            f"image_size={image_size} does not match SAM2 model.image_size={model.image_size}."
        )
    model.to(device)
    sgn.to(device)
    refiner.to(device)
    full_res_refiner.to(device)
    load_inference_checkpoint(
        checkpoint=checkpoint,
        model=model,
        sgn=sgn,
        refiner=refiner,
        full_res_refiner=full_res_refiner,
        device=device,
    )
    del checkpoint
    model.eval()
    sgn.eval()
    refiner.eval()
    full_res_refiner.eval()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    all_image_paths = collect_images(args.image, args.image_dir)
    image_paths = [
        path
        for index, path in enumerate(all_image_paths)
        if index % args.num_shards == args.shard_index
    ]
    print(
        f"[PGSAM] shard={args.shard_index}/{args.num_shards} "
        f"images={len(image_paths)}/{len(all_image_paths)}"
    )

    for image_path in iter_progress(image_paths):
        image_tensor, original, transform_meta = load_image(
            image_path,
            image_size,
            device,
            resize_mode=resize_mode,
        )
        valid = torch.zeros_like(image_tensor[:, :1])
        left, top = transform_meta.offset_xy
        width, height = transform_meta.resized_size
        valid[..., top:top+height, left:left+width] = 1
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_bf16,
                dtype=torch.bfloat16,
            ):
                forward_result = forward_pgsam(
                    model=model,
                    sgn=sgn,
                    refiner=refiner,
                    full_res_refiner=full_res_refiner,
                    images=image_tensor,
                    gt_masks=None,
                    refine_iter=refine_iter,
                    prompt_source=model_args.prompt_source,
                    valid_masks=valid,
                    use_decoder_reconstruction=model_args.use_decoder_reconstruction,
                    highres_feature_mode=model_args.highres_feature_mode,
                    use_boundary_refinement=model_args.use_boundary_refinement,
                    use_fullres_refinement=model_args.use_fullres_refinement,
                    multimask_output=multimask_output,
                    num_prompt_points=model_args.num_prompt_points,
                    global_prompt_points=model_args.global_prompt_points,
                    prompt_constraint_mode=model_args.prompt_constraint_mode,
                    soft_constraint_floor=model_args.soft_constraint_floor,
                    progressive_initial_weight=model_args.progressive_initial_weight,
                    return_details=True,
                )
                logits = forward_result.logits
                prompt_points = forward_result.prompt_points_xy
                prompt_heatmap = forward_result.prompt_heatmap
                if forward_result.prompt_point_labels is not None:
                    prompt_points = prompt_points[:, forward_result.prompt_point_labels[0] == 1]
            prob = torch.sigmoid(logits.float())
            raw_mask = (prob > float(args.threshold)).float()
            post_mask = postprocess_main_region(
                prob,
                threshold=args.threshold,
                close_radius=post_cfg["post_close_radius"],
                min_area_ratio=post_cfg["post_min_area_ratio"],
                boundary_band_radius=post_cfg["post_boundary_band_radius"],
                boundary_prob_threshold=post_cfg["post_boundary_prob_threshold"],
            )
            post_mask = edge_snapping_postprocess(
                post_mask,
                image=image_tensor,
                edge_search_radius=post_cfg["edge_search_radius"],
            )

        stem = image_path.stem
        mask_path = mask_dir / f"{stem}_mask.png"
        mask_u8 = resize_mask(post_mask, transform_meta)
        Image.fromarray(mask_u8, mode="L").save(mask_path)

        raw_path = None
        if args.save_raw:
            raw_path = raw_dir / f"{stem}_raw_mask.png"
            Image.fromarray(resize_mask(raw_mask, transform_meta), mode="L").save(raw_path)

        prob_path = None
        if args.save_prob:
            prob_path = prob_dir / f"{stem}_prob.png"
            Image.fromarray(resize_prob(prob, transform_meta), mode="L").save(prob_path)

        overlay_path = None
        if args.save_overlay:
            overlay_path = overlay_dir / f"{stem}_overlay.png"
            save_overlay(original, mask_u8, overlay_path)

        metadata = {
            "image": str(image_path),
            "mask": str(mask_path),
            "raw_mask": None if raw_path is None else str(raw_path),
            "probability": None if prob_path is None else str(prob_path),
            "overlay": None if overlay_path is None else str(overlay_path),
            "threshold": float(args.threshold),
            "postprocess": post_cfg,
            "model_input_size": image_size,
            "resize_mode": resize_mode,
            "multimask_output": multimask_output,
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "prompt_points_xy_model_space": [
                [float(x), float(y)]
                for x, y in prompt_points[0].detach().cpu().tolist()
            ],
            "prompt_heatmap_shape": None
            if prompt_heatmap is None
            else list(prompt_heatmap.shape),
        }
        with open(meta_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[PGSAM] done. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
