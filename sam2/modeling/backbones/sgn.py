from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(num_channels: int) -> nn.Module:
    """
    GroupNorm tends to be more stable than BatchNorm for small batch sizes.
    """
    # Use 32 groups when possible; fall back to 1 group (=InstanceNorm-like).
    num_groups = 32
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class BoxRegressor(nn.Module):
    """
    Predict normalized xyxy box coordinates from SGN bottleneck features.
    """

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        mid_channels = max(32, hidden_channels // 2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, mid_channels),
            nn.LayerNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 4),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raw_box = self.head(feats)  # (B,4)
        norm_box = torch.sigmoid(raw_box)
        x1 = torch.minimum(norm_box[:, 0], norm_box[:, 2])
        y1 = torch.minimum(norm_box[:, 1], norm_box[:, 3])
        x2 = torch.maximum(norm_box[:, 0], norm_box[:, 2])
        y2 = torch.maximum(norm_box[:, 1], norm_box[:, 3])
        return torch.stack([x1, y1, x2, y2], dim=1).clamp(0.0, 1.0)


class PointRegressor(nn.Module):
    """
    Predict normalized center point (x, y) from SGN bottleneck features.
    """

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        mid_channels = max(32, hidden_channels // 2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, mid_channels),
            nn.LayerNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 2),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(feats)).clamp(0.0, 1.0)


class _SobelTextureEnergy(nn.Module):
    """
    Lightweight, parameter-free texture energy using Sobel gradients.
    Output is a single-channel "edge/texture energy" map per sample.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            energy: (B, 1, H, W), non-negative
        """
        # Convert feature tensor into a pseudo-gray image.
        x_gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(x_gray, self.sobel_x.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(x_gray, self.sobel_y.to(dtype=x.dtype), padding=1)
        energy = torch.sqrt(gx * gx + gy * gy + self.eps)
        return energy


class _TextureDensityAnalyzer(nn.Module):
    """
    Produces a soft spatial mask that emphasizes high texture density regions.
    """

    def __init__(
        self,
        alpha_init: float = 5.0,
        bias_init: float = 0.0,
        mask_power: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.texture_energy = _SobelTextureEnergy(eps=eps)

        # Learnable affine transform applied to standardized energy.
        self.texture_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.texture_bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))
        self.mask_power = float(mask_power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            mask: (B, 1, H, W) in [0, 1]
        """
        energy = self.texture_energy(x)  # (B,1,H,W)
        mean = energy.mean(dim=(2, 3), keepdim=True)
        std = energy.std(dim=(2, 3), keepdim=True) + self.eps
        z = (energy - mean) / std

        mask = torch.sigmoid(self.texture_alpha * z + self.texture_bias)
        if self.mask_power != 1.0:
            mask = mask.pow(self.mask_power)
        return mask


class SemanticGuidanceNeck(nn.Module):
    """
    SemanticGuidanceNeck

    - Input: Stage-4 feature map from Hiera (1/32 resolution), as (B, C, H, W).
      (For convenience, forward also accepts a list/tuple of stage features and
      uses the last element as Stage-4.)
    - ASPP-style multi-scale context extraction.
    - Texture density analyzer to emphasize high-density map areas over
      low-density legend boxes.
    - Output:
      1) a single-channel 64x64 heatmap (values in [0,1]) for main map region.
      2) optionally, a normalized bounding box (xyxy in [0,1]) and a center point
         (x,y in [0,1]) for auxiliary supervision.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Tuple[int, ...] = (2, 4, 6),
        aspp_branch_channels: Optional[int] = None,
        dropout: float = 0.0,
        point_box_mix: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.atrous_rates = tuple(int(r) for r in atrous_rates)
        self.dropout = float(dropout)
        self.point_box_mix = float(point_box_mix)
        if not (0.0 <= self.point_box_mix <= 1.0):
            raise ValueError("point_box_mix must be in [0, 1].")

        if aspp_branch_channels is None:
            # With N dilated branches + 1x1 + global pooling = (len(rates)+2) branches.
            # Choose a branch width so concatenation can be projected back to out_channels.
            aspp_branch_channels = max(32, out_channels // 4)
        self.aspp_branch_channels = int(aspp_branch_channels)

        # 1x1 conv branch
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.aspp_branch_channels, kernel_size=1, bias=False),
            _make_norm(self.aspp_branch_channels),
            nn.ReLU(inplace=True),
        )

        # Dilated 3x3 conv branches
        self.branches_dilated = nn.ModuleList()
        for r in self.atrous_rates:
            self.branches_dilated.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        self.aspp_branch_channels,
                        kernel_size=3,
                        padding=r,
                        dilation=r,
                        bias=False,
                    ),
                    _make_norm(self.aspp_branch_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global pooling branch
        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, self.aspp_branch_channels, kernel_size=1, bias=False),
            _make_norm(self.aspp_branch_channels),
            nn.ReLU(inplace=True),
        )

        concat_channels = self.aspp_branch_channels * (len(self.atrous_rates) + 2)
        self.aspp_fuse = nn.Sequential(
            nn.Conv2d(concat_channels, self.out_channels, kernel_size=1, bias=False),
            _make_norm(self.out_channels),
            nn.ReLU(inplace=True),
        )
        if self.dropout > 0:
            self.aspp_fuse.add_module("dropout2d", nn.Dropout2d(p=self.dropout))

        # Texture density analyzer (map texture vs legend boxes)
        self.texture_analyzer = _TextureDensityAnalyzer()

        # Predict an attention logits map, then gate with texture density.
        self.attention_head = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels // 2, kernel_size=1, bias=False),
            _make_norm(self.out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels // 2, 1, kernel_size=1, bias=True),
        )
        # BoxRegressor head over SGN bottleneck features.
        self.box_regressor = BoxRegressor(
            in_channels=self.out_channels,
            hidden_channels=max(32, self.out_channels // 2),
        )
        self.point_regressor = PointRegressor(
            in_channels=self.out_channels,
            hidden_channels=max(32, self.out_channels // 2),
        )

    def _get_stage4(self, x: Union[torch.Tensor, Iterable[torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("Received empty list/tuple for stage features.")
            return x[-1]
        # If it's some other iterable, take the last element.
        x_list = list(x)
        if len(x_list) == 0:
            raise ValueError("Received empty iterable for stage features.")
        return x_list[-1]

    def forward(
        self,
        stage4: Union[torch.Tensor, Iterable[torch.Tensor]],
        return_density_mask: bool = False,
        return_box: bool = False,
        return_point: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Args:
            stage4: (B, C, H, W) or a stage feature list/tuple, whose last element is Stage-4.
            return_density_mask: if True, also return (B, 1, H, W) density mask.
        Returns:
            spatial_attention_map: (B, 1, 64, 64), values in [0, 1]
        """
        x = self._get_stage4(stage4)
        if x.dim() != 4:
            raise ValueError(f"stage4 must be a 4D tensor (B,C,H,W), got {x.shape}")

        _, _, H, W = x.shape

        # ASPP multi-scale context (keeps spatial size).
        feats_1x1 = self.branch_1x1(x)
        feats_dilated = [b(x) for b in self.branches_dilated]

        feats_global = self.branch_global(x)  # (B, Cb, 1, 1)
        feats_global = F.interpolate(
            feats_global, size=(H, W), mode="bilinear", align_corners=False
        )

        feats = torch.cat([feats_1x1, *feats_dilated, feats_global], dim=1)
        feats = self.aspp_fuse(feats)  # (B, 256, H, W)

        # Texture density mask emphasizes map areas with higher texture density.
        density_mask = self.texture_analyzer(x)  # (B,1,H,W)

        # Produce spatial attention and gate by texture density.
        attn_logits = self.attention_head(feats)  # (B,1,H,W)
        spatial_attention = torch.sigmoid(attn_logits) * density_mask
        spatial_attention = spatial_attention.clamp(0.0, 1.0)

        # Predict optional geometry for auxiliary losses.
        norm_box = None
        norm_point = None
        if return_box or return_point:
            if return_box:
                norm_box = self.box_regressor(feats)
            if return_point:
                norm_point_raw = self.point_regressor(feats)
                if return_box:
                    box_center = torch.stack(
                        [
                            0.5 * (norm_box[:, 0] + norm_box[:, 2]),
                            0.5 * (norm_box[:, 1] + norm_box[:, 3]),
                        ],
                        dim=1,
                    )
                    # Couple point and box predictions to keep them consistent.
                    norm_point = (
                        (1.0 - self.point_box_mix) * norm_point_raw
                        + self.point_box_mix * box_center
                    )
                else:
                    norm_point = norm_point_raw
                norm_point = norm_point.clamp(0.0, 1.0)

        # Standardize to a fixed 64x64 attention map for downstream decoder use.
        spatial_attention = F.interpolate(
            spatial_attention, size=(64, 64), mode="bilinear", align_corners=False
        )

        if return_density_mask and return_box and return_point:
            return spatial_attention, density_mask, norm_box, norm_point
        if return_density_mask and return_box:
            return spatial_attention, density_mask, norm_box
        if return_density_mask:
            return spatial_attention, density_mask
        if return_box and return_point:
            return spatial_attention, norm_box, norm_point
        if return_box:
            return spatial_attention, norm_box
        if return_point:
            return spatial_attention, norm_point
        return spatial_attention

