# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# this release folder. The file has been trimmed for PG-SAM image inference.

"""SAM2 model construction utilities required by PG-SAM inference."""

from __future__ import annotations

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2  # noqa: F401


def build_sam2(
    config_file: str,
    ckpt_path: str | None = None,
    device: str = "cuda",
    mode: str = "eval",
    hydra_overrides_extra: list[str] | None = None,
    apply_postprocessing: bool = True,
    **kwargs,
):
    """Build a SAM2 image model from a local Hydra config and checkpoint."""
    _ = kwargs
    overrides = list(hydra_overrides_extra or [])
    if apply_postprocessing:
        overrides.extend(
            [
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            ]
        )

    cfg = compose(config_name=config_file, overrides=overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str | None) -> None:
    if ckpt_path is None:
        return
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state)
    if missing_keys:
        logging.error("Missing SAM2 checkpoint keys: %s", missing_keys)
        raise RuntimeError("Failed to load SAM2 checkpoint because keys are missing.")
    if unexpected_keys:
        logging.error("Unexpected SAM2 checkpoint keys: %s", unexpected_keys)
        raise RuntimeError("Failed to load SAM2 checkpoint because keys are unexpected.")
    logging.info("Loaded SAM2 checkpoint successfully.")
