"""Versioned foreground-point selection with explicit validity and FP32 geometry."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import _largest_component_np


@dataclass
class PointSet:
    coordinates: torch.Tensor
    labels: torch.Tensor


def select_safe_points(heatmap, valid_mask, image_size, num_points=8,
                       min_distance=4.0, confidence_ratio=0.6,
                       min_confidence=0.5, restrict_to_main_component=True,
                       existing=None):
    """Return foreground points and -1 placeholders; never invent positive points.

    Spacing is in heatmap cells. Confidence never relaxes to fill the point budget.
    Existing valid points participate in spacing, but are not returned again.
    """
    if heatmap.ndim != 4 or heatmap.shape[1] != 1 or heatmap.shape[-2:] != (64, 64):
        raise ValueError('Expected Bx1x64x64 probability heatmap')
    if not torch.isfinite(heatmap).all():
        raise ValueError('Nonfinite prompt heatmap')
    if num_points < 0 or image_size < 2:
        raise ValueError('Invalid point count or image size')
    if not 0 <= confidence_ratio <= 1 or not 0 <= min_confidence <= 1:
        raise ValueError('Point confidence thresholds must be in [0,1]')
    if valid_mask is None or valid_mask.shape != (heatmap.shape[0], 1, image_size, image_size):
        raise ValueError('Safe prompting requires an image-space valid mask')
    if not torch.isfinite(valid_mask).all():
        raise ValueError('Nonfinite valid mask')
    b = heatmap.shape[0]
    result = torch.zeros(b, num_points, 2, device=heatmap.device, dtype=torch.float32)
    labels = torch.full((b, num_points), -1, device=heatmap.device, dtype=torch.int32)
    if num_points == 0:
        return PointSet(result, labels)
    # Conservative support, followed by exact checks at the actual image coordinates.
    small_valid = F.adaptive_avg_pool2d(valid_mask.float(), (64, 64)) >= 1.0
    hmaps = heatmap.detach().float().cpu().numpy()[:, 0]
    masks = small_valid.cpu().numpy()[:, 0]
    exact_masks = valid_mask.detach().cpu().numpy()[:, 0] > .5
    scale = float(image_size - 1) / 63.0
    yy, xx = np.mgrid[:64, :64]
    mapped_x = np.rint(xx * scale).astype(int)
    mapped_y = np.rint(yy * scale).astype(int)
    for index in range(b):
        field = np.clip(hmaps[index], 0, 1)
        valid = masks[index] & exact_masks[index][mapped_y, mapped_x]
        maximum = float(field[valid].max()) if valid.any() else 0.0
        threshold = max(float(min_confidence), maximum * float(confidence_ratio))
        support = valid & (field >= threshold) & (field > 0)
        if restrict_to_main_component and support.any():
            support = _largest_component_np(support)
        if support is None or not support.any():
            continue
        ys, xs = np.nonzero(support)
        candidates = np.stack([xs, ys], axis=1).astype(np.float32)
        weights = field[ys, xs]
        accepted = []
        if existing is not None:
            active = existing.labels[index] == 1
            accepted = (existing.coordinates[index, active].detach().cpu().numpy() / scale).tolist()
        chosen = []
        if not accepted:
            centroid = (candidates * weights[:, None]).sum(0) / weights.sum()
            cell = np.rint(centroid).astype(int).clip(0, 63)
            xy_image = np.rint(centroid * scale).astype(int).clip(0, image_size - 1)
            if support[cell[1], cell[0]] and exact_masks[index][xy_image[1], xy_image[0]]:
                anchor = centroid
            else:
                # Choose a real supported location, not an average of tied optima.
                anchor = candidates[((candidates-centroid)**2).sum(1).argmin()]
            chosen.append(anchor.tolist())
            accepted.append(anchor.tolist())
        while len(chosen) < num_points:
            if accepted:
                d2 = ((candidates[:, None] - np.asarray(accepted)[None])**2).sum(2).min(1)
            else:
                d2 = np.ones(len(candidates))
            eligible = d2 >= max(float(min_distance)**2, 1.0)
            if not eligible.any():
                break
            scores = np.where(eligible, d2 * weights, -1)
            point = candidates[int(scores.argmax())].tolist()
            chosen.append(point)
            accepted.append(point)
        if chosen:
            count = len(chosen)
            result[index, :count] = torch.as_tensor(chosen, device=heatmap.device, dtype=torch.float64) * scale
            labels[index, :count] = 1
    return PointSet(result, labels)


def merge_point_sets(first, second):
    return PointSet(torch.cat([first.coordinates, second.coordinates], dim=1),
                    torch.cat([first.labels, second.labels], dim=1))
