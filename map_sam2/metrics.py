"""Metric helpers for validation-time reporting and model selection."""

import numpy as np
import torch

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def compute_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor, bf1_tol: int = 2) -> dict:
    # bf1_tol is kept for CLI/backward compatibility even though the current
    # implementation reports pixel-level mask metrics only.
    _ = bf1_tol
    eps = 1e-6
    pred = (pred_mask > 0.5).float()
    gt = (gt_mask > 0.5).float()

    inter = (pred * gt).sum(dim=(1, 2, 3))
    union = ((pred + gt) > 0).float().sum(dim=(1, 2, 3))
    obj_iou = torch.where(union > 0, inter / (union + eps), torch.ones_like(union))

    pred_pos = pred.sum(dim=(1, 2, 3))
    gt_pos = gt.sum(dim=(1, 2, 3))
    total = torch.tensor(float(pred[0].numel()), device=pred.device, dtype=pred.dtype)
    fp = pred_pos - inter
    fn = gt_pos - inter
    tn = total - inter - fp - fn
    obj_recall = torch.where(gt_pos > 0, inter / (gt_pos + eps), torch.zeros_like(gt_pos))
    precision = torch.where(pred_pos > 0, inter / (pred_pos + eps), torch.ones_like(pred_pos))
    recall = torch.where(gt_pos > 0, inter / (gt_pos + eps), torch.zeros_like(gt_pos))
    specificity = torch.where((tn + fp) > 0, tn / (tn + fp + eps), torch.ones_like(tn))
    accuracy = (inter + tn) / (total + eps)
    macc_all = 0.5 * (recall + specificity)
    fg_f1_all = torch.where(
        (precision + recall) > 0,
        2.0 * precision * recall / (precision + recall + eps),
        torch.zeros_like(precision),
    )

    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    lcc_list = []
    hole_list = []

    structure = np.ones((3, 3), dtype=np.uint8)
    use_scipy = False
    ndimage = None
    try:
        import scipy.ndimage as _ndimage  # type: ignore

        ndimage = _ndimage
        use_scipy = True
    except Exception:
        use_scipy = False

    for i in range(pred_np.shape[0]):
        pm = pred_np[i, 0].astype(bool)
        total_area = float(pm.sum())
        if total_area <= 0.0:
            lcc_ratio = 0.0
        else:
            if use_scipy:
                labels, num = ndimage.label(pm, structure=structure)
                if num <= 0:
                    lcc_ratio = 0.0
                else:
                    areas = np.bincount(labels.reshape(-1))
                    lcc_ratio = float(areas[1:].max() / (total_area + eps))
            elif cv2 is None:
                lcc_ratio = 0.0
            else:
                m = (pm.astype(np.uint8) * 255).copy()
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                if num_labels <= 1:
                    lcc_ratio = 0.0
                else:
                    lcc_ratio = float(stats[1:, cv2.CC_STAT_AREA].astype(np.float32).max() / (total_area + eps))

        hole_ratio = 0.0
        if total_area > 0.0:
            if use_scipy:
                labels, num = ndimage.label(pm, structure=structure)
                if num > 0:
                    areas = np.bincount(labels.reshape(-1))
                    best_label = int(np.argmax(areas[1:])) + 1
                    largest = labels == best_label
                    largest_area = float(largest.sum())
                    if largest_area > 0:
                        filled = ndimage.binary_fill_holes(largest, structure=structure)
                        holes = filled & (~largest)
                        hole_ratio = float(holes.sum()) / (largest_area + eps)
            elif cv2 is not None:
                m = (pm.astype(np.uint8) * 255).copy()
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                if num_labels > 1:
                    best_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
                    largest = labels == best_label
                    largest_area = float(largest.sum())
                    if largest_area > 0:
                        largest_u8 = largest.astype(np.uint8) * 255
                        flood = largest_u8.copy()
                        ff_mask = np.zeros((pm.shape[0] + 2, pm.shape[1] + 2), np.uint8)
                        cv2.floodFill(flood, ff_mask, (0, 0), 255)
                        holes = cv2.bitwise_not(flood) > 0
                        hole_ratio = float((holes & (largest == 0)).sum()) / (largest_area + eps)

        lcc_list.append(float(lcc_ratio))
        hole_list.append(float(hole_ratio))

    lcc_ratio_t = torch.tensor(lcc_list, device=pred.device, dtype=pred.dtype)
    hole_ratio_t = torch.tensor(hole_list, device=pred.device, dtype=pred.dtype)
    gt_non_empty = gt.sum(dim=(1, 2, 3)) > 0
    if gt_non_empty.any():
        obj_iou_fg = obj_iou[gt_non_empty].mean()
        obj_recall_fg = obj_recall[gt_non_empty].mean()
        fg_f1 = fg_f1_all[gt_non_empty].mean()
        fg_precision = precision[gt_non_empty].mean()
        fg_recall = recall[gt_non_empty].mean()
        accuracy_fg = accuracy[gt_non_empty].mean()
        macc_fg = macc_all[gt_non_empty].mean()
        lcc_ratio_fg = lcc_ratio_t[gt_non_empty].mean()
        hole_ratio_fg = hole_ratio_t[gt_non_empty].mean()
    else:
        obj_iou_fg = torch.tensor(0.0, device=pred.device)
        obj_recall_fg = torch.tensor(0.0, device=pred.device)
        fg_f1 = torch.tensor(0.0, device=pred.device)
        fg_precision = torch.tensor(0.0, device=pred.device)
        fg_recall = torch.tensor(0.0, device=pred.device)
        accuracy_fg = torch.tensor(0.0, device=pred.device)
        macc_fg = torch.tensor(0.0, device=pred.device)
        lcc_ratio_fg = torch.tensor(0.0, device=pred.device)
        hole_ratio_fg = torch.tensor(0.0, device=pred.device)

    return {
        "iou": float(obj_iou_fg.detach().cpu()),
        "precision": float(fg_precision.detach().cpu()),
        "recall": float(fg_recall.detach().cpu()),
        "f1": float(fg_f1.detach().cpu()),
        "accuracy": float(accuracy_fg.detach().cpu()),
        "macc": float(macc_fg.detach().cpu()),
        "tp": float(inter.sum().detach().cpu()),
        "fp": float(fp.sum().detach().cpu()),
        "fn": float(fn.sum().detach().cpu()),
        "tn": float(tn.sum().detach().cpu()),
        "pred_pos_total": float(pred_pos.sum().detach().cpu()),
        "gt_pos_total": float(gt_pos.sum().detach().cpu()),
        "intersection_total": float(inter.sum().detach().cpu()),
        "union_total": float(union.sum().detach().cpu()),
        "obj_iou": float(obj_iou_fg.detach().cpu()),
        "obj_recall": float(obj_recall_fg.detach().cpu()),
        "fg_f1": float(fg_f1.detach().cpu()),
        "fg_precision": float(fg_precision.detach().cpu()),
        "fg_recall": float(fg_recall.detach().cpu()),
        "lcc_ratio": float(lcc_ratio_fg.detach().cpu()),
        "hole_ratio": float(hole_ratio_fg.detach().cpu()),
        "gt_non_empty_ratio": float(gt_non_empty.float().mean().detach().cpu()),
    }


def metrics_from_pixel_counts(tp: float, fp: float, fn: float, tn: float = 0.0) -> dict:
    eps = 1e-6
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    macc = 0.5 * (recall + specificity)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "macc": float(macc),
        "obj_iou": float(iou),
        "obj_recall": float(recall),
        "fg_f1": float(f1),
        "fg_precision": float(precision),
        "fg_recall": float(recall),
    }
