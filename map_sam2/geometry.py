"""Geometry and post-processing helpers used by the MMA-SAM2 pipeline."""

from typing import List, Sequence

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
    if heatmap_64.dim() != 4 or heatmap_64.shape[1] != 1:
        raise ValueError(f"heatmap_64 must be (B,1,H,W), got {heatmap_64.shape}")
    b, _, h, w = heatmap_64.shape
    if h != 64 or w != 64:
        raise ValueError(f"heatmap_64 must be 64x64, got {(h, w)}")

    weights = heatmap_64.squeeze(1).clamp(min=0.0)
    if restrict_to_main_component:
        weights = weights * largest_component_mask(weights, min_conf_ratio=heatmap_min_conf_ratio)
    denom = weights.sum(dim=(1, 2)) + eps
    max_w = weights.amax(dim=(1, 2))
    conf_ok = max_w >= float(heatmap_min_conf_ratio)

    grid_y = torch.arange(h, device=weights.device, dtype=weights.dtype).view(h, 1).expand(h, w)
    grid_x = torch.arange(w, device=weights.device, dtype=weights.dtype).view(1, w).expand(h, w)

    centroid_x = (weights * grid_x).sum(dim=(1, 2)) / denom
    centroid_y = (weights * grid_y).sum(dim=(1, 2)) / denom
    small = denom <= eps * 10.0
    centroid_x = torch.where(small, torch.full_like(centroid_x, (w - 1) / 2.0), centroid_x)
    centroid_y = torch.where(small, torch.full_like(centroid_y, (h - 1) / 2.0), centroid_y)

    x0_abs = centroid_x / (w - 1) * float(img_size)
    y0_abs = centroid_y / (h - 1) * float(img_size)

    points = torch.zeros((b, int(num_points), 2), device=heatmap_64.device, dtype=heatmap_64.dtype)
    points[:, 0, 0] = x0_abs
    points[:, 0, 1] = y0_abs

    k = max(0, min(int(num_points) - 1, h * w))
    if k == 0:
        return points.clamp(0.0, float(img_size))

    cand_k = min(h * w, max(1, int(k) * int(heatmap_topk_mult)))
    flat = weights.reshape(b, -1)
    topk_idx_all = torch.topk(flat, k=cand_k, dim=1).indices

    min_d2 = float(min_dist_px * min_dist_px)
    for i in range(b):
        if not bool(conf_ok[i].item()):
            points[i, :, 0] = x0_abs[i]
            points[i, :, 1] = y0_abs[i]
            continue
        accepted = [(float(centroid_x[i].detach().cpu().item()), float(centroid_y[i].detach().cpu().item()))]
        cur = 1
        conf_thresh = float(max_w[i].detach().cpu().item()) * float(point_min_conf_ratio)
        for j in range(int(topk_idx_all.shape[1])):
            if cur >= int(num_points):
                break
            idx = topk_idx_all[i, j].detach().cpu().item()
            if float(flat[i, idx].detach().cpu().item()) < conf_thresh:
                continue
            y = float(idx // w)
            x = float(idx % w)
            ok = True
            for ax, ay in accepted:
                dx = x - ax
                dy = y - ay
                if dx * dx + dy * dy < min_d2:
                    ok = False
                    break
            if not ok:
                continue
            points[i, cur, 0] = x / (w - 1) * float(img_size)
            points[i, cur, 1] = y / (h - 1) * float(img_size)
            accepted.append((x, y))
            cur += 1

        if cur < int(num_points):
            positive_idx = torch.nonzero(flat[i] > 0, as_tuple=False).flatten()
            positive_idx_cpu = positive_idx.detach().cpu().tolist()
            max_val = float(max_w[i].detach().cpu().item()) + eps
            while cur < int(num_points) and positive_idx_cpu:
                best_idx = None
                best_score = -1.0
                for idx in positive_idx_cpu:
                    y = float(idx // w)
                    x = float(idx % w)
                    min_d2_to_accepted = min((x - ax) * (x - ax) + (y - ay) * (y - ay) for ax, ay in accepted)
                    if min_d2_to_accepted < 1.0:
                        continue
                    weight = float(flat[i, idx].detach().cpu().item()) / max_val
                    score = min_d2_to_accepted * (0.25 + 0.75 * weight)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                y = float(best_idx // w)
                x = float(best_idx % w)
                points[i, cur, 0] = x / (w - 1) * float(img_size)
                points[i, cur, 1] = y / (h - 1) * float(img_size)
                accepted.append((x, y))
                cur += 1

        if cur < int(num_points):
            points[i, cur:, 0] = x0_abs[i]
            points[i, cur:, 1] = y0_abs[i]
    return points.clamp(0.0, float(img_size))


def largest_component_mask(weights: torch.Tensor, min_conf_ratio: float = 0.05) -> torch.Tensor:
    """Return the largest connected positive region for each spatial weight map."""
    if weights.dim() != 3:
        raise ValueError(f"weights must be (B,H,W), got {weights.shape}")

    b, h, w = weights.shape
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
    mask_cpu = (mask.detach().float().cpu().numpy()[:, 0] > 0.5)

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
    return (cv2.bitwise_or(component_u8, holes) > 0)


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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def select_high_res_features(neck_feats: Sequence[torch.Tensor], enabled: bool, mode: str) -> List[torch.Tensor]:
    s0 = neck_feats[0]
    s1 = neck_feats[1]
    z0 = torch.zeros_like(s0)
    z1 = torch.zeros_like(s1)
    if not enabled or mode == "none":
        return [z0, z1]
    if mode == "stage1":
        return [s0, z1]
    if mode == "stage2":
        return [z0, s1]
    if mode == "stage1_stage2":
        return [s0, s1]
    raise ValueError(f"Unsupported high_res feature mode: {mode}")


def masks_to_boxes_xyxy(mask: torch.Tensor) -> torch.Tensor:
    b, _, h, w = mask.shape
    boxes = torch.zeros((b, 4), device=mask.device, dtype=torch.float32)
    m = (mask > 0.5).squeeze(1)
    for i in range(b):
        ys, xs = torch.where(m[i])
        if ys.numel() == 0:
            boxes[i] = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], device=mask.device)
            continue
        boxes[i] = torch.stack([xs.min().float(), ys.min().float(), xs.max().float(), ys.max().float()], dim=0)
    return boxes


def boxes_to_centers_xy(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    cx = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
    cy = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3])
    return torch.stack([cx, cy], dim=1)


def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x1 = torch.maximum(box1[:, 0], box2[:, 0])
    y1 = torch.maximum(box1[:, 1], box2[:, 1])
    x2 = torch.minimum(box1[:, 2], box2[:, 2])
    y2 = torch.minimum(box1[:, 3], box2[:, 3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return (inter + eps) / (union + eps)


def generalized_box_iou_loss(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x1 = torch.maximum(box1[:, 0], box2[:, 0])
    y1 = torch.maximum(box1[:, 1], box2[:, 1])
    x2 = torch.minimum(box1[:, 2], box2[:, 2])
    y2 = torch.minimum(box1[:, 3], box2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    iou = (inter + eps) / (union + eps)

    cx1 = torch.minimum(box1[:, 0], box2[:, 0])
    cy1 = torch.minimum(box1[:, 1], box2[:, 1])
    cx2 = torch.maximum(box1[:, 2], box2[:, 2])
    cy2 = torch.maximum(box1[:, 3], box2[:, 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
    giou = iou - (c_area - union) / (c_area + eps)
    return (1.0 - giou).mean()
