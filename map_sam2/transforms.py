"""Shared image geometry transforms for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry needed to map a square model output back to the source image."""

    original_size: Tuple[int, int]
    resized_size: Tuple[int, int]
    offset_xy: Tuple[int, int]
    target_size: int


def resize_and_pad_pil(
    image: Image.Image,
    target_size: int,
    mask: Optional[Image.Image] = None,
) -> tuple[Image.Image, Optional[Image.Image], LetterboxMeta]:
    """Resize without changing aspect ratio, then center-pad to a square."""

    target_size = int(target_size)
    if target_size <= 0:
        raise ValueError("target_size must be > 0")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")
    if mask is not None and mask.size != image.size:
        raise ValueError(
            f"Image and mask sizes must match, got image={image.size}, mask={mask.size}"
        )

    scale = min(target_size / float(width), target_size / float(height))
    new_width = max(1, min(target_size, int(round(width * scale))))
    new_height = max(1, min(target_size, int(round(height * scale))))
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2

    image_resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    image_out = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    image_out.paste(image_resized, (left, top))

    mask_out: Optional[Image.Image] = None
    if mask is not None:
        mask_resized = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        mask_out = Image.new("L", (target_size, target_size), 0)
        mask_out.paste(mask_resized, (left, top))

    meta = LetterboxMeta(
        original_size=(width, height),
        resized_size=(new_width, new_height),
        offset_xy=(left, top),
        target_size=target_size,
    )
    return image_out, mask_out, meta


def resize_pil_pair(
    image: Image.Image,
    target_size: int,
    mask: Optional[Image.Image] = None,
    resize_mode: str = "letterbox",
) -> tuple[Image.Image, Optional[Image.Image], LetterboxMeta]:
    """Resize an image/mask pair under an explicit, checkpointable protocol."""

    if resize_mode == "letterbox":
        return resize_and_pad_pil(image=image, target_size=target_size, mask=mask)
    if resize_mode != "stretch":
        raise ValueError("resize_mode must be 'stretch' or 'letterbox'")
    if mask is not None and mask.size != image.size:
        raise ValueError(
            f"Image and mask sizes must match, got image={image.size}, mask={mask.size}"
        )

    target_size = int(target_size)
    if target_size <= 0:
        raise ValueError("target_size must be > 0")
    original_size = image.size
    image_out = image.resize(
        (target_size, target_size),
        Image.Resampling.BILINEAR,
    )
    mask_out = None
    if mask is not None:
        mask_out = mask.resize(
            (target_size, target_size),
            Image.Resampling.NEAREST,
        )
    meta = LetterboxMeta(
        original_size=original_size,
        resized_size=(target_size, target_size),
        offset_xy=(0, 0),
        target_size=target_size,
    )
    return image_out, mask_out, meta


def restore_letterboxed_mask(
    mask: torch.Tensor,
    meta: LetterboxMeta,
    mode: str = "nearest",
) -> torch.Tensor:
    """Remove model-space padding and resize a mask to the source dimensions."""

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if mask.dim() != 4:
        raise ValueError(f"mask must have 2, 3, or 4 dimensions, got {mask.shape}")

    if mask.shape[-2:] != (meta.target_size, meta.target_size):
        interpolate_kwargs = {}
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            interpolate_kwargs["align_corners"] = False
        mask = F.interpolate(
            mask.float(),
            size=(meta.target_size, meta.target_size),
            mode=mode,
            **interpolate_kwargs,
        )

    left, top = meta.offset_xy
    resized_width, resized_height = meta.resized_size
    cropped = mask[..., top : top + resized_height, left : left + resized_width]
    original_width, original_height = meta.original_size
    interpolate_kwargs = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interpolate_kwargs["align_corners"] = False
    return F.interpolate(
        cropped.float(),
        size=(original_height, original_width),
        mode=mode,
        **interpolate_kwargs,
    )
