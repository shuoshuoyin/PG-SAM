"""Inference-only forward path for the PG-SAM release package."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import clean_binary_region, heatmap_to_points_xy, select_high_res_features


def forward_prompts(
    model: nn.Module,
    image_embeddings: torch.Tensor,
    high_res_features: list[torch.Tensor],
    points_xy: torch.Tensor,
    implicit_prompt_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode masks from image embeddings and automatically generated points."""
    if points_xy.dim() == 2:
        points_xy = points_xy.unsqueeze(1)
    point_coords = points_xy.to(dtype=image_embeddings.dtype)
    batch_size, num_points, _ = point_coords.shape
    point_labels = torch.ones((batch_size, num_points), device=points_xy.device, dtype=torch.int32)
    sparse_prompt_embeddings, dense_prompt_embeddings = model.sam_prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )
    low_res_logits, _, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
        implicit_prompt_map=implicit_prompt_map,
    )
    return low_res_logits


def forward_mapsam(
    model: nn.Module,
    sgn: nn.Module,
    refiner: nn.Module,
    full_res_refiner: nn.Module,
    images: torch.Tensor,
    refine_iter: int = 1,
    use_decoder_reconstruction: bool = True,
    highres_feature_mode: str = "stage1_stage2",
    use_boundary_refinement: bool = True,
    use_fullres_refinement: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the PG-SAM image inference path and return logits plus SGN/prompt geometry."""
    stage_feats = model.image_encoder.trunk(images)
    if not isinstance(stage_feats, (list, tuple)) or len(stage_feats) < 4:
        raise RuntimeError("Hiera trunk must return stage features [S1, S2, S3, S4].")
    stage4 = stage_feats[-1]

    neck_feats, _ = model.image_encoder.neck(stage_feats)
    if model.image_encoder.scalp > 0:
        neck_feats = neck_feats[: -model.image_encoder.scalp]
    if len(neck_feats) < 3:
        raise RuntimeError("Expected at least three neck feature levels.")

    image_embeddings = neck_feats[-1]
    high_res_features = select_high_res_features(
        neck_feats=neck_feats,
        enabled=use_decoder_reconstruction,
        mode=highres_feature_mode,
    )

    heatmap_64, pred_box_norm, pred_point_norm = sgn(stage4, return_box=True, return_point=True)
    img_size = float(model.image_size - 1)
    box_abs = pred_box_norm.clamp(0.0, 1.0) * img_size
    x1, y1, x2, y2 = box_abs.unbind(dim=1)
    min_span = 2.0
    x2 = torch.maximum(x2, x1 + min_span).clamp(max=img_size)
    y2 = torch.maximum(y2, y1 + min_span).clamp(max=img_size)
    box_abs = torch.stack([x1, y1, x2, y2], dim=1)
    point_abs = pred_point_norm.clamp(0.0, 1.0) * img_size
    stage1_points_xy = heatmap_to_points_xy(
        heatmap_64=heatmap_64,
        img_size=img_size,
        num_points=8,
        min_dist_px=4,
    )

    implicit_prompt_map = None
    if use_decoder_reconstruction:
        implicit_prompt_map = F.interpolate(
            heatmap_64,
            size=image_embeddings.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    stage1_low_res_logits = forward_prompts(
        model=model,
        image_embeddings=image_embeddings,
        high_res_features=high_res_features,
        points_xy=stage1_points_xy,
        implicit_prompt_map=implicit_prompt_map,
    )
    stage1_prob_64 = F.interpolate(
        torch.sigmoid(stage1_low_res_logits.float()),
        size=heatmap_64.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    stage1_mask_64 = F.interpolate(
        (stage1_prob_64 > 0.5).float(),
        size=heatmap_64.shape[-2:],
        mode="nearest",
    )
    stage1_region_64 = clean_binary_region(stage1_mask_64, erode_radius=1)
    high_conf_64 = (stage1_prob_64 >= 0.60).to(dtype=heatmap_64.dtype)
    stage1_high_conf_region_64 = clean_binary_region(stage1_region_64 * high_conf_64, erode_radius=0)
    min_stage2_area = 128
    has_high_conf_region = stage1_high_conf_region_64.flatten(1).sum(dim=1) >= min_stage2_area
    stage2_region_64 = torch.where(
        has_high_conf_region.view(-1, 1, 1, 1),
        stage1_high_conf_region_64,
        stage1_region_64,
    )
    stage2_heatmap_64 = heatmap_64 * stage2_region_64
    has_stage2_region = stage2_heatmap_64.flatten(1).amax(dim=1) > 0.0
    stage2_points_xy = heatmap_to_points_xy(
        heatmap_64=stage2_heatmap_64,
        img_size=img_size,
        num_points=8,
        min_dist_px=4,
        heatmap_min_conf_ratio=0.01,
        heatmap_topk_mult=32,
        point_min_conf_ratio=0.05,
    )
    points_prompt_xy = torch.where(
        has_stage2_region.view(-1, 1, 1),
        stage2_points_xy,
        stage1_points_xy,
    )

    low_res_logits = forward_prompts(
        model=model,
        image_embeddings=image_embeddings,
        high_res_features=high_res_features,
        points_xy=points_prompt_xy,
        implicit_prompt_map=implicit_prompt_map,
    )
    if use_boundary_refinement:
        refined_low_res_logits = refiner(low_res_logits, high_res_features[0])
    else:
        refined_low_res_logits = low_res_logits

    logits_1024 = F.interpolate(
        refined_low_res_logits,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).float()
    if use_fullres_refinement:
        image_f32 = images.float()
        for _ in range(max(1, int(refine_iter))):
            logits_1024 = full_res_refiner(logits_1024, image_f32)
    return logits_1024, box_abs, point_abs, points_prompt_xy
