"""Geometry and post-processing helpers used by the PGSAM pipeline."""

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def binary_boundary(mask_bin: torch.Tensor) -> torch.Tensor:
    eroded = 1.0 - F.max_pool2d(1.0 - mask_bin, kernel_size=3, stride=1, padding=1)
    return torch.clamp(mask_bin - eroded, min=0.0, max=1.0)


def postprocess_main_region(
    pred_prob: torch.Tensor,
    threshold: float = 0.5,
    close_radius: int = 3,
    min_area_ratio: float = 0.001,
    boundary_band_radius: int = 2,
    boundary_prob_threshold: float = 0.62,
) -> torch.Tensor:
    pred_bin = pred_prob > float(threshold)
    try:
        import scipy.ndimage as ndimage  # type: ignore

        b, _, h, w = pred_bin.shape
        out = torch.zeros((b, 1, h, w), device=pred_prob.device, dtype=pred_prob.dtype)
        min_area = max(1, int(float(h * w) * float(min_area_ratio)))
        k = max(1, 2 * int(close_radius) + 1)
        structure = np.ones((k, k), dtype=np.uint8)
        bband = max(0, int(boundary_band_radius))

        for i in range(b):
            m = pred_bin[i, 0].detach().cpu().numpy().astype(bool)
            prob_i = pred_prob[i, 0].detach().cpu().numpy().astype(np.float32)
            if close_radius > 0:
                m = ndimage.binary_closing(m, structure=structure)
            labels, num = ndimage.label(m)
            if num <= 0:
                continue
            areas = np.bincount(labels.reshape(-1))
            if areas.shape[0] <= 1:
                continue
            areas_fg = areas[1:]
            best_rel = int(np.argmax(areas_fg))
            best_label = best_rel + 1
            if int(areas_fg[best_rel]) < min_area:
                continue

            largest = labels == best_label
            if bband > 0:
                inner = ndimage.binary_erosion(
                    largest,
                    structure=np.ones((3, 3), dtype=np.uint8),
                    iterations=bband,
                )
                band = largest & (~inner)
                keep_band = band & (prob_i >= float(boundary_prob_threshold))
                largest = inner | keep_band
                labels2, num2 = ndimage.label(largest)
                if num2 > 0:
                    areas2 = np.bincount(labels2.reshape(-1))
                    if areas2.shape[0] > 1:
                        best2 = 1 + int(np.argmax(areas2[1:]))
                        largest = labels2 == best2

            filled = ndimage.binary_fill_holes(largest)
            out[i, 0] = torch.from_numpy(filled.astype(np.float32)).to(device=out.device, dtype=out.dtype)
        return out
    except Exception:
        if cv2 is None:
            return pred_bin.float()

    pred_bin_f = pred_bin.float()
    if cv2 is None:
        return pred_bin_f

    b, _, h, w = pred_bin_f.shape
    out = torch.zeros_like(pred_bin_f)
    min_area = max(1, int(float(h * w) * float(min_area_ratio)))
    k = max(1, 2 * int(close_radius) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bband = max(0, int(boundary_band_radius))

    for i in range(b):
        m = (pred_bin_f[i, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
        prob_i = pred_prob[i, 0].detach().cpu().numpy().astype(np.float32)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num_labels <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_idx = 1 + int(np.argmax(areas))
        if int(stats[best_idx, cv2.CC_STAT_AREA]) < min_area:
            continue

        largest = (labels == best_idx).astype(np.uint8)
        if bband > 0:
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            inner = cv2.erode(largest, erode_k, iterations=bband)
            band = ((largest > 0) & (inner == 0))
            keep_band = band & (prob_i >= float(boundary_prob_threshold))
            largest = ((inner > 0) | keep_band).astype(np.uint8)
            num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
                largest * 255,
                connectivity=8,
            )
            if num_labels2 > 1:
                best2 = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
                largest = (labels2 == best2).astype(np.uint8)

        largest_u8 = largest.astype(np.uint8) * 255
        flood = largest_u8.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(largest_u8, holes)
        out[i, 0] = torch.from_numpy((filled > 0).astype(np.float32)).to(device=out.device)
    return out


def edge_snapping_postprocess(pred_bin: torch.Tensor, image: torch.Tensor, edge_search_radius: int = 2) -> torch.Tensor:
    if cv2 is None:
        return pred_bin
    r = max(0, int(edge_search_radius))
    if r == 0:
        return pred_bin

    b, _, _, _ = pred_bin.shape
    out = pred_bin.clone()
    for i in range(b):
        pm = (pred_bin[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        img = image[i].detach().cpu().numpy()
        gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).astype(np.float32)
        grad = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        gthr = float(np.percentile(grad, 80.0))
        edge = grad >= gthr
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        inner = cv2.erode(pm, k, iterations=1)
        band = (pm > 0) & (inner == 0)
        keep_band = band & edge
        snapped = (inner > 0) | keep_band
        out[i, 0] = torch.from_numpy(snapped.astype(np.float32)).to(device=out.device, dtype=out.dtype)
    return out


def heatmap_to_points_xy(
    heatmap_64: torch.Tensor,
    img_size: float,
    num_points: int = 8,
    min_dist_px: int = 4,
    eps: float = 1e-6,
    heatmap_min_conf_ratio: float = 0.05,
    heatmap_topk_mult: int = 6,
    point_min_conf_ratio: float = 0.3,
    restrict_to_main_component: bool = True,
) -> torch.Tensor:
    """Convert a non-differentiable SGN heatmap into well-spaced point prompts.

    Point selection intentionally runs from one small CPU snapshot of each 64x64
    heatmap.  The previous implementation repeatedly called ``.item()`` inside
    Python loops, forcing hundreds of CUDA synchronizations per batch.
    """
    if heatmap_64.dim() != 4 or heatmap_64.shape[1] != 1:
        raise ValueError(f"heatmap_64 must be (B,1,H,W), got {heatmap_64.shape}")
    b, _, h, w = heatmap_64.shape
    if h != 64 or w != 64:
        raise ValueError(f"heatmap_64 must be 64x64, got {(h, w)}")
    num_points = int(num_points)
    if num_points < 1:
        raise ValueError("num_points must be >= 1")

    weights_batch = heatmap_64.detach().float().cpu().numpy()[:, 0]
    points_np = np.zeros((b, num_points, 2), dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    min_d2 = float(min_dist_px * min_dist_px)
    extra_points = min(num_points - 1, h * w)
    candidate_count = min(h * w, max(1, extra_points * int(heatmap_topk_mult)))

    for batch_idx in range(b):
        weights = np.clip(weights_batch[batch_idx], 0.0, None)
        max_weight = float(weights.max())
        if restrict_to_main_component and max_weight > 0.0:
            component = _largest_component_np(
                weights >= max_weight * float(heatmap_min_conf_ratio)
            )
            if component is not None:
                weights = weights * component.astype(weights.dtype)

        denom = float(weights.sum())
        if denom <= eps * 10.0:
            centroid_x = (w - 1) / 2.0
            centroid_y = (h - 1) / 2.0
        else:
            centroid_x = float((weights * grid_x).sum() / (denom + eps))
            centroid_y = float((weights * grid_y).sum() / (denom + eps))

        points_np[batch_idx, :, 0] = centroid_x
        points_np[batch_idx, :, 1] = centroid_y
        if extra_points == 0 or max_weight < float(heatmap_min_conf_ratio):
            continue

        flat = weights.reshape(-1)
        sorted_indices = np.argsort(-flat, kind="stable")[:candidate_count]
        confidence_threshold = max_weight * float(point_min_conf_ratio)
        accepted = [(centroid_x, centroid_y)]
        current = 1
        for flat_idx in sorted_indices.tolist():
            if current >= num_points:
                break
            if float(flat[flat_idx]) < confidence_threshold:
                continue
            y, x = divmod(int(flat_idx), w)
            if min((x - ax) ** 2 + (y - ay) ** 2 for ax, ay in accepted) < min_d2:
                continue
            points_np[batch_idx, current] = (float(x), float(y))
            accepted.append((float(x), float(y)))
            current += 1

        positive_indices = np.flatnonzero(flat > 0.0)
        if current < num_points and positive_indices.size > 0:
            positive_y, positive_x = np.divmod(positive_indices, w)
            positive_xy = np.stack([positive_x, positive_y], axis=1).astype(np.float32)
            normalized_weights = flat[positive_indices] / (max_weight + eps)
            while current < num_points:
                accepted_np = np.asarray(accepted, dtype=np.float32)
                distances = ((positive_xy[:, None, :] - accepted_np[None, :, :]) ** 2).sum(axis=2)
                nearest_d2 = distances.min(axis=1)
                scores = nearest_d2 * (0.25 + 0.75 * normalized_weights)
                scores[nearest_d2 < 1.0] = -1.0
                best_position = int(np.argmax(scores))
                if float(scores[best_position]) < 0.0:
                    break
                x, y = positive_xy[best_position]
                points_np[batch_idx, current] = (x, y)
                accepted.append((float(x), float(y)))
                current += 1

    points_np[..., 0] = points_np[..., 0] / float(w - 1) * float(img_size)
    points_np[..., 1] = points_np[..., 1] / float(h - 1) * float(img_size)
    return torch.from_numpy(points_np).to(
        device=heatmap_64.device,
        dtype=heatmap_64.dtype,
    ).clamp(0.0, float(img_size))


def largest_component_mask(weights: torch.Tensor, min_conf_ratio: float = 0.05) -> torch.Tensor:
    """Return the largest connected positive region for each spatial weight map."""
    if weights.dim() != 3:
        raise ValueError(f"weights must be (B,H,W), got {weights.shape}")

    b, _, _ = weights.shape
    out = torch.zeros_like(weights, dtype=torch.float32)
    weights_cpu = weights.detach().float().cpu().numpy()

    for i in range(b):
        max_w = float(weights_cpu[i].max())
        if max_w <= 0.0:
            out[i].fill_(1.0)
            continue
        mask = weights_cpu[i] >= max_w * float(min_conf_ratio)
        component = _largest_component_np(mask)
        if component is None:
            component = mask
        out[i] = torch.from_numpy(component.astype(np.float32)).to(device=weights.device)
    return out.to(dtype=weights.dtype)


def clean_binary_region(mask: torch.Tensor, erode_radius: int = 1) -> torch.Tensor:
    """Keep the largest filled component and optionally shrink it away from boundaries."""
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be (B,1,H,W), got {mask.shape}")

    b, _, h, w = mask.shape
    out = torch.zeros((b, 1, h, w), device=mask.device, dtype=torch.float32)
    mask_cpu = mask.detach().float().cpu().numpy()[:, 0] > 0.5

    for i in range(b):
        component = _largest_filled_component_np(mask_cpu[i])
        if component is None:
            continue
        eroded = _erode_np(component, radius=int(erode_radius))
        if eroded is not None and int(eroded.sum()) >= 8:
            component = eroded
        out[i, 0] = torch.from_numpy(component.astype(np.float32)).to(device=mask.device)
    return out.to(dtype=mask.dtype)


def _largest_component_np(mask: np.ndarray) -> np.ndarray | None:
    mask_u8 = mask.astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return None

    try:
        import scipy.ndimage as ndimage  # type: ignore

        labels, num = ndimage.label(mask_u8.astype(bool))
        if num <= 0:
            return None
        areas = np.bincount(labels.reshape(-1))
        if areas.shape[0] <= 1:
            return None
        best_label = 1 + int(np.argmax(areas[1:]))
        return labels == best_label
    except Exception:
        pass

    if cv2 is None:
        return mask_u8.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return None
    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best_label


def _largest_filled_component_np(mask: np.ndarray) -> np.ndarray | None:
    component = _largest_component_np(mask)
    if component is None:
        return None
    try:
        import scipy.ndimage as ndimage  # type: ignore

        return ndimage.binary_fill_holes(component)
    except Exception:
        pass

    if cv2 is None:
        return component
    component_u8 = component.astype(np.uint8) * 255
    flood = component_u8.copy()
    ff_mask = np.zeros((component_u8.shape[0] + 2, component_u8.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(component_u8, holes) > 0


def _erode_np(mask: np.ndarray, radius: int) -> np.ndarray | None:
    if radius <= 0:
        return mask
    try:
        import scipy.ndimage as ndimage  # type: ignore

        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        return ndimage.binary_erosion(mask, structure=structure)
    except Exception:
        pass

    if cv2 is None:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def select_high_res_features(
    neck_feats: Sequence[torch.Tensor],
    enabled: bool,
    mode: str,
    absent_as_none: bool = False,
) -> List[Optional[torch.Tensor]]:
    s0 = neck_feats[0]
    s1 = neck_feats[1]
    z0 = None if absent_as_none else torch.zeros_like(s0)
    z1 = None if absent_as_none else torch.zeros_like(s1)
    if not enabled or mode == "none":
        return [z0, z1]
    if mode == "stage1":
        return [s0, z1]
    if mode == "stage2":
        return [z0, s1]
    if mode == "stage1_stage2":
        return [s0, s1]
    raise ValueError(f"Unsupported high_res feature mode: {mode}")


def masks_to_interior_points_xy(mask: torch.Tensor, work_size: int = 64) -> torch.Tensor:
    """Return a foreground point near the maximum interior distance of each mask."""

    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be (B,1,H,W), got {mask.shape}")
    work_size = max(8, int(work_size))
    batch_size, _, height, width = mask.shape
    small = F.interpolate(mask.float(), size=(work_size, work_size), mode="nearest")
    small_np = (small.detach().cpu().numpy()[:, 0] > 0.5).astype(np.uint8)
    points = np.zeros((batch_size, 2), dtype=np.float32)

    for batch_idx, binary in enumerate(small_np):
        if int(binary.sum()) == 0:
            x_small = (work_size - 1) * 0.5
            y_small = (work_size - 1) * 0.5
        else:
            distance = None
            try:
                import scipy.ndimage as ndimage  # type: ignore

                distance = ndimage.distance_transform_edt(binary.astype(bool))
            except Exception:
                if cv2 is not None:
                    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            if distance is None:
                ys, xs = np.nonzero(binary)
                x_small = float(xs.mean())
                y_small = float(ys.mean())
            else:
                max_distance = float(distance.max())
                deepest_y, deepest_x = np.nonzero(
                    distance >= max_distance - max(1e-6, max_distance * 1e-6)
                )
                x_small = float(deepest_x.mean())
                y_small = float(deepest_y.mean())

        points[batch_idx, 0] = x_small / float(work_size - 1) * float(width - 1)
        points[batch_idx, 1] = y_small / float(work_size - 1) * float(height - 1)

    return torch.from_numpy(points).to(device=mask.device, dtype=mask.dtype)
