"""Strip training-only state from a trusted PG-SAM V6.1 checkpoint.

The output contains only architecture/inference settings and trained weights.
Do not use this script on untrusted pickle files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


INFERENCE_ARGS = (
    "config", "image_size", "resize_mode", "use_lora", "lora_rank", "lora_alpha",
    "prompt_source", "highres_fusion_mode", "prompt_policy",
    "use_initial_implicit_prompt", "use_final_implicit_prompt",
    "prompt_refinement_mode", "point_confidence_ratio", "point_min_confidence",
    "use_layout_prior", "implicit_prompt_mode", "refine_attn_dim",
    "refine_local_kernel", "refine_downsample_ratio", "fullres_blocks",
    "fullres_hidden_dim", "fullres_block_type", "decoder_prompt_gate_floor",
    "sam2_normalize_inputs", "use_decoder_reconstruction", "highres_feature_mode",
    "use_boundary_refinement", "use_fullres_refinement", "num_prompt_points",
    "global_prompt_points", "prompt_constraint_mode", "soft_constraint_floor",
    "progressive_initial_weight", "refine_iter", "multimask_output",
    "post_close_radius", "post_min_area_ratio", "post_boundary_band_radius",
    "post_boundary_prob_threshold", "edge_search_radius",
)
WEIGHT_KEYS = ("lora", "sgn", "mask_decoder", "refiner", "full_res_refiner")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path, help="Trusted training best_model.pt")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    if not str(source.get("code_version", "")).startswith("PGSAM-v6.1"):
        raise ValueError("Source checkpoint must be PG-SAM V6.1")
    original_args = source.get("args")
    if not isinstance(original_args, dict):
        raise ValueError("Source checkpoint has no argument dictionary")
    missing = [key for key in WEIGHT_KEYS if key not in source]
    if missing:
        raise ValueError(f"Missing model weights: {missing}")
    released = {
        "code_version": source["code_version"],
        "args": {key: original_args[key] for key in INFERENCE_ARGS if key in original_args},
        **{key: source[key] for key in WEIGHT_KEYS},
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(released, args.destination)
    print(f"Saved inference-only checkpoint: {args.destination}")
    print(f"Keys: {sorted(released)}")


if __name__ == "__main__":
    main()
