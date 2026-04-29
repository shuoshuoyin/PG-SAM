# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# this release folder. The file has been trimmed for PG-SAM image inference.

"""Small SAM2 utility subset required by PG-SAM inference."""

from __future__ import annotations

import torch


def mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """Compute bounding boxes for masks with shape ``[B, 1, H, W]``."""
    batch_size, _, height, width = masks.shape
    device = masks.device
    xs = torch.arange(width, device=device, dtype=torch.int32)
    ys = torch.arange(height, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(batch_size, 1, height, width)
    grid_ys = grid_ys[None, None, ...].expand(batch_size, 1, height, width)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, width).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, height).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    return torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)
