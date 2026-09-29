"""Shared PGSAM forward path used by training, evaluation, and demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (
    clean_binary_region,
    heatmap_to_points_xy,
    select_high_res_features,
)
from .safe_prompts import select_safe_points, merge_point_sets


@dataclass
class PGSAMForwardDetails:
    logits: torch.Tensor
    prompt_points_xy: torch.Tensor
    quality_predictions: torch.Tensor
    quality_targets: Optional[torch.Tensor]
    prompt_heatmap: Optional[torch.Tensor]
    prompt_point_labels: Optional[torch.Tensor] = None


def normalize_sam2_images(images: torch.Tensor) -> torch.Tensor:
    """Apply the normalization used by the official SAM2 image predictor."""

    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std


def forward_prompts(
    model: nn.Module,
    image_embeddings: torch.Tensor,
    high_res_features: List[Optional[torch.Tensor]],
    points_xy: torch.Tensor,
    implicit_prompt_map: Optional[torch.Tensor] = None,
    multimask_output: bool = False,
    point_labels: Optional[torch.Tensor] = None,
    preserve_point_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one or more candidate masks from the supplied prompts."""

    if points_xy.dim() == 2:
        points_xy = points_xy.unsqueeze(1)
    if point_labels is not None:
        # Decode groups by actual point count. No repeated padding tokens alter attention.
        counts = (point_labels == 1).sum(1)
        masks, qualities, indices = [], [], []
        for count in counts.unique().tolist():
            idx = torch.where(counts == count)[0]
            pts = points_xy[idx][point_labels[idx] == 1].reshape(len(idx), count, 2)
            out, quality = forward_prompts(
                model, image_embeddings[idx],
                [x[idx] if x is not None else None for x in high_res_features], pts,
                implicit_prompt_map[idx] if implicit_prompt_map is not None else None,
                multimask_output, preserve_point_precision=True,
            )
            masks.append(out)
            qualities.append(quality)
            indices.append(idx)
        order = torch.cat(indices).argsort()
        return torch.cat(masks)[order], torch.cat(qualities)[order]
    point_coords = points_xy.to(dtype=torch.float32 if preserve_point_precision else image_embeddings.dtype)
    batch_size, num_points, _ = point_coords.shape
    point_labels = torch.ones(
        (batch_size, num_points),
        device=points_xy.device,
        dtype=torch.int32,
    )
    sparse_prompt_embeddings, dense_prompt_embeddings = model.sam_prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )
    low_res_logits, quality_predictions, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
        multimask_output=bool(multimask_output),
        repeat_image=False,
        high_res_features=high_res_features,
        implicit_prompt_map=implicit_prompt_map,
    )
    return low_res_logits, quality_predictions


def candidate_mask_iou_targets(
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute detached IoU targets for the SAM mask-quality head."""

    gt_low = F.interpolate(gt_masks.float(), size=mask_logits.shape[-2:], mode="nearest")
    pred = (torch.sigmoid(mask_logits.detach().float()) > float(threshold)).float()
    gt = (gt_low > 0.5).float().expand(-1, pred.shape[1], -1, -1)
    intersection = (pred * gt).sum(dim=(2, 3))
    union = ((pred + gt) > 0).float().sum(dim=(2, 3))
    return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.ones_like(union))


def select_mask_candidate(
    mask_logits: torch.Tensor,
    quality_predictions: torch.Tensor,
    gt_masks: Optional[torch.Tensor] = None,
    select_by_gt: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Select the oracle candidate during training and quality-ranked candidate otherwise."""

    if mask_logits.dim() != 4 or quality_predictions.dim() != 2:
        raise ValueError(
            "mask_logits and quality_predictions must be (B,M,H,W) and (B,M), "
            f"got {mask_logits.shape} and {quality_predictions.shape}"
        )
    if mask_logits.shape[:2] != quality_predictions.shape:
        raise ValueError("Mask candidate count must match quality prediction count.")

    quality_targets = None
    if gt_masks is not None:
        quality_targets = candidate_mask_iou_targets(mask_logits, gt_masks)
    if select_by_gt:
        if quality_targets is None:
            raise ValueError("gt_masks are required when select_by_gt=True")
        selected_indices = quality_targets.argmax(dim=1)
    else:
        selected_indices = quality_predictions.detach().float().argmax(dim=1)

    batch_indices = torch.arange(mask_logits.shape[0], device=mask_logits.device)
    selected_logits = mask_logits[batch_indices, selected_indices].unsqueeze(1)
    return selected_logits, quality_targets


def forward_pgsam(
    model: nn.Module,
    sgn: nn.Module,
    refiner: nn.Module,
    full_res_refiner: nn.Module,
    images: torch.Tensor,
    gt_masks: Optional[torch.Tensor] = None,
    refine_iter: int = 1,
    prompt_source: str = "sgn_auto",
    use_decoder_reconstruction: bool = True,
    highres_feature_mode: str = "stage1_stage2",
    use_boundary_refinement: bool = True,
    use_fullres_refinement: bool = True,
    multimask_output: bool = False,
    select_mask_by_gt: bool = False,
    return_details: bool = False,
    num_prompt_points: int = 8,
    global_prompt_points: int = 0,
    prompt_constraint_mode: str = "hard",
    soft_constraint_floor: float = 0.25,
    progressive_initial_weight: float = 0.5,
    valid_masks: Optional[torch.Tensor] = None,
):
    """Run the complete automatic-prompt PGSAM image path."""

    num_prompt_points = int(num_prompt_points)
    global_prompt_points = int(global_prompt_points)
    if num_prompt_points < 1:
        raise ValueError("num_prompt_points must be >= 1")
    if not 0 <= global_prompt_points <= num_prompt_points:
        raise ValueError("global_prompt_points must be in [0, num_prompt_points]")
    if prompt_constraint_mode not in {"hard", "soft"}:
        raise ValueError("prompt_constraint_mode must be 'hard' or 'soft'")
    if not 0.0 <= float(soft_constraint_floor) <= 1.0:
        raise ValueError("soft_constraint_floor must be in [0, 1]")
    if not 0.0 <= float(progressive_initial_weight) <= 1.0:
        raise ValueError("progressive_initial_weight must be in [0, 1]")

    encoder_images = (
        normalize_sam2_images(images)
        if bool(getattr(model, "pgsam_normalize_inputs", False))
        else images
    )
    stage_feats = model.image_encoder.trunk(encoder_images)
    if not isinstance(stage_feats, (list, tuple)) or len(stage_feats) < 4:
        raise RuntimeError("Hiera trunk must return stage features [S1,S2,S3,S4].")
    stage4 = stage_feats[-1]

    neck_feats, _ = model.image_encoder.neck(stage_feats)
    if model.image_encoder.scalp > 0:
        neck_feats = neck_feats[: -model.image_encoder.scalp]
    if len(neck_feats) < 3:
        raise RuntimeError("Expected at least 3 neck feature levels for skip-connections.")

    image_embeddings = neck_feats[-1]
    high_res_features = select_high_res_features(
        neck_feats=neck_feats,
        enabled=use_decoder_reconstruction,
        mode=highres_feature_mode,
        absent_as_none=(
            getattr(model.sam_mask_decoder, "highres_fusion_mode", "legacy_concat")
            == "residual_additive"
        ),
    )

    img_size = float(model.image_size - 1)
    prompt_heatmap = None
    point_labels = None
    safe = getattr(model, "pgsam_prompt_policy", "legacy") == "safe_v6"
    initial_prior_enabled = getattr(model, "pgsam_initial_implicit", None)
    final_prior_enabled = getattr(model, "pgsam_final_implicit", None)
    if initial_prior_enabled is None:
        initial_prior_enabled = use_decoder_reconstruction
    if final_prior_enabled is None:
        final_prior_enabled = True
    if prompt_source == "sgn_auto" and safe:
        heatmap_64 = sgn(stage4)
        prompt_heatmap = heatmap_64
        selection = dict(valid_mask=valid_masks, image_size=model.image_size,
                         confidence_ratio=getattr(model, "pgsam_point_conf_ratio", .6),
                         min_confidence=getattr(model, "pgsam_point_min_conf", .5))
        initial_points = select_safe_points(heatmap_64, num_points=num_prompt_points, **selection)
        valid64 = F.adaptive_avg_pool2d(valid_masks.float(), (64, 64)) >= 1.0
        initial_prior = heatmap_64.float() * valid64
        with torch.no_grad():
            candidates, quality = forward_prompts(
                model, image_embeddings, high_res_features, initial_points.coordinates,
                initial_prior if initial_prior_enabled else None,
                multimask_output, point_labels=initial_points.labels)
            initial_logits, _ = select_mask_candidate(candidates, quality)
            response = F.interpolate(torch.sigmoid(initial_logits.float()), (64, 64),
                                     mode="bilinear", align_corners=False) * valid64
            if getattr(model, "pgsam_prompt_refinement", "global_local") == "initial":
                points = initial_points
            else:
                global_points = select_safe_points(heatmap_64, num_points=global_prompt_points,
                                                   restrict_to_main_component=False, **selection)
                if prompt_constraint_mode == "hard":
                    local_map = heatmap_64.detach().float() * clean_binary_region((response > .5).float())
                else:
                    local_map = heatmap_64.detach().float() * (soft_constraint_floor + (1-soft_constraint_floor)*response)
                # Local confidence remains relative to the original SGN scale;
                # no relaxation outside supported locations just to fill a quota.
                local_points = select_safe_points(local_map, num_points=num_prompt_points-global_prompt_points,
                                                  existing=global_points, **selection)
                points = merge_point_sets(global_points, local_points)
        points_prompt_xy, point_labels = points.coordinates, points.labels
        implicit_prompt_map = ((1-progressive_initial_weight)*heatmap_64.float()
                               + progressive_initial_weight*response) * valid64
        if not final_prior_enabled:
            implicit_prompt_map = None
    elif prompt_source == "sgn_auto":
        heatmap_64 = sgn(stage4)
        prompt_heatmap = heatmap_64
        initial_points_xy = heatmap_to_points_xy(
            heatmap_64=heatmap_64,
            img_size=img_size,
            num_points=num_prompt_points,
            min_dist_px=4,
        )
        if global_prompt_points > 0:
            # These points intentionally see the complete SGN field, including
            # secondary components that the first SAM response may have missed.
            global_points_xy = heatmap_to_points_xy(
                heatmap_64=heatmap_64,
                img_size=img_size,
                num_points=global_prompt_points,
                min_dist_px=4,
                heatmap_min_conf_ratio=0.01,
                heatmap_topk_mult=128,
                point_min_conf_ratio=0.10,
                restrict_to_main_component=False,
            )
        else:
            global_points_xy = initial_points_xy[:, :0]
        implicit_prompt_map = None
        if initial_prior_enabled:
            implicit_prompt_map = F.interpolate(
                heatmap_64,
                size=image_embeddings.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        # The initial response spatially constrains the final automatic prompt set.
        with torch.no_grad():
            initial_candidates, initial_quality = forward_prompts(
                model=model,
                image_embeddings=image_embeddings,
                high_res_features=high_res_features,
                points_xy=initial_points_xy,
                implicit_prompt_map=implicit_prompt_map,
                multimask_output=multimask_output,
            )
            initial_logits, _ = select_mask_candidate(
                initial_candidates,
                initial_quality,
                select_by_gt=False,
            )
            initial_prob_64 = F.interpolate(
                torch.sigmoid(initial_logits.float()),
                size=heatmap_64.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            local_point_count = num_prompt_points - global_prompt_points
            if local_point_count > 0:
                if prompt_constraint_mode == "hard":
                    initial_region_64 = clean_binary_region(
                        (initial_prob_64 > 0.5).float(),
                        erode_radius=1,
                    )
                    high_confidence_region_64 = clean_binary_region(
                        initial_region_64
                        * (initial_prob_64 >= 0.60).to(dtype=heatmap_64.dtype),
                        erode_radius=0,
                    )
                    has_high_confidence_region = (
                        high_confidence_region_64.flatten(1).sum(dim=1) >= 128
                    )
                    constrained_region_64 = torch.where(
                        has_high_confidence_region.view(-1, 1, 1, 1),
                        high_confidence_region_64,
                        initial_region_64,
                    )
                    local_heatmap_64 = heatmap_64.detach() * constrained_region_64
                    has_local_region = local_heatmap_64.flatten(1).amax(dim=1) > 0.0
                else:
                    # A soft floor preserves exploration outside an incomplete
                    # first-pass mask and avoids reinforcing early under-segmentation.
                    floor = float(soft_constraint_floor)
                    local_heatmap_64 = heatmap_64.detach() * (
                        floor + (1.0 - floor) * initial_prob_64
                    )
                    has_local_region = torch.ones(
                        heatmap_64.shape[0], device=heatmap_64.device, dtype=torch.bool
                    )
                local_points_xy = heatmap_to_points_xy(
                    heatmap_64=local_heatmap_64,
                    img_size=img_size,
                    num_points=local_point_count,
                    min_dist_px=4,
                    heatmap_min_conf_ratio=0.01,
                    heatmap_topk_mult=32,
                    point_min_conf_ratio=0.05,
                )
                fallback_points = initial_points_xy[:, global_prompt_points:]
                local_points_xy = torch.where(
                    has_local_region.view(-1, 1, 1),
                    local_points_xy,
                    fallback_points,
                )
                points_prompt_xy = torch.cat(
                    [global_points_xy, local_points_xy], dim=1
                )
            else:
                points_prompt_xy = global_points_xy
        # Keep the SGN half of the progressive prompt differentiable. The
        # first-pass probability acts only as a stable self-correction target.
        initial_weight = float(progressive_initial_weight)
        progressive_prompt_64 = (
            (1.0 - initial_weight) * heatmap_64.float()
            + initial_weight * initial_prob_64.to(dtype=heatmap_64.dtype)
        )
        implicit_prompt_map = F.interpolate(
            progressive_prompt_64,
            size=image_embeddings.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(dtype=image_embeddings.dtype)
        if not final_prior_enabled:
            implicit_prompt_map = None
    elif prompt_source == "image_center_point":
        batch_size = images.shape[0]
        center_x = float(images.shape[-1] - 1) * 0.5
        center_y = float(images.shape[-2] - 1) * 0.5
        points_prompt_xy = torch.tensor(
            [center_x, center_y],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 2).repeat(batch_size, 1)
        implicit_prompt_map = None
    else:
        raise ValueError(f"Unsupported prompt_source: {prompt_source}")

    mask_candidates, quality_predictions = forward_prompts(
        model=model,
        image_embeddings=image_embeddings,
        high_res_features=high_res_features,
        points_xy=points_prompt_xy,
        implicit_prompt_map=implicit_prompt_map,
        multimask_output=multimask_output,
        point_labels=point_labels,
        preserve_point_precision=safe,
    )
    low_res_logits, quality_targets = select_mask_candidate(
        mask_candidates,
        quality_predictions,
        gt_masks=gt_masks,
        select_by_gt=select_mask_by_gt,
    )
    if use_boundary_refinement and high_res_features[0] is None:
        raise ValueError("Boundary refinement requires the Stage-1 high-resolution feature.")
    refined_low_res_logits = (
        refiner(low_res_logits, high_res_features[0])
        if use_boundary_refinement
        else low_res_logits
    )

    logits_full = F.interpolate(
        refined_low_res_logits,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).float()
    if use_fullres_refinement:
        image_f32 = images.float()
        image_context = None
        if hasattr(full_res_refiner, "prepare_image_context"):
            image_context = full_res_refiner.prepare_image_context(image_f32)
        for _ in range(max(1, int(refine_iter))):
            logits_full = full_res_refiner(
                logits_full,
                image_f32,
                image_context=image_context,
            )

    if return_details:
        return PGSAMForwardDetails(
            logits=logits_full,
            prompt_points_xy=points_prompt_xy,
            quality_predictions=quality_predictions,
            quality_targets=quality_targets,
            prompt_heatmap=prompt_heatmap,
            prompt_point_labels=point_labels,
        )
    return logits_full, points_prompt_xy, prompt_heatmap
