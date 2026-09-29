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
    - Output: a single-channel 64x64 main-map-area prior in [0, 1].

    The optional layout branch is specific to page-level main-map extraction:
    it encodes normalized location and distance to the page boundary instead of
    regressing an object box. This keeps PGSAM's prompt generation box-free.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Tuple[int, ...] = (2, 4, 6),
        aspp_branch_channels: Optional[int] = None,
        dropout: float = 0.0,
        use_layout_prior: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.atrous_rates = tuple(int(r) for r in atrous_rates)
        self.dropout = float(dropout)
        self.use_layout_prior = bool(use_layout_prior)

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

        self.layout_branch = None
        if self.use_layout_prior:
            self.layout_branch = nn.Sequential(
                nn.Conv2d(3, self.aspp_branch_channels, kernel_size=1, bias=False),
                _make_norm(self.aspp_branch_channels),
                nn.ReLU(inplace=True),
            )

        concat_channels = self.aspp_branch_channels * (
            len(self.atrous_rates) + 2 + int(self.use_layout_prior)
        )
        self.aspp_fuse = nn.Sequential(
            nn.Conv2d(concat_channels, self.out_channels, kernel_size=1, bias=False),
            _make_norm(self.out_channels),
            nn.ReLU(inplace=True),
        )
        if self.dropout > 0:
            self.aspp_fuse.add_module("dropout2d", nn.Dropout2d(p=self.dropout))

        # Texture density analyzer (map texture vs legend boxes)
        self.texture_analyzer = _TextureDensityAnalyzer()
        # The former hard product `semantic * density` could nearly erase the
        # whole prior before SGN had learned anything, especially with 10-shot
        # data. Keep texture as a learnable residual cue initialized to an
        # identity mapping so the semantic branch always receives gradients.
        self.texture_logit_scale = nn.Parameter(torch.tensor(0.0))

        # Predict an attention logits map, then gate with texture density.
        self.attention_head = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels // 2, kernel_size=1, bias=False),
            _make_norm(self.out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels // 2, 1, kernel_size=1, bias=True),
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

        branches = [feats_1x1, *feats_dilated, feats_global]
        if self.layout_branch is not None:
            y = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
            x_coord = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
            edge_distance = 1.0 - torch.maximum(xx.abs(), yy.abs())
            layout = torch.stack([xx, yy, edge_distance], dim=0).unsqueeze(0)
            layout = layout.expand(x.shape[0], -1, -1, -1)
            branches.append(self.layout_branch(layout))

        feats = torch.cat(branches, dim=1)
        feats = self.aspp_fuse(feats)  # (B, 256, H, W)

        # Texture density mask emphasizes map areas with higher texture density.
        density_mask = self.texture_analyzer(x)  # (B,1,H,W)

        # Fuse semantic attention and texture density in logit space. A zero
        # scale is exactly the semantic prediction; training can learn whether
        # texture should enhance or suppress a location.
        attn_logits = self.attention_head(feats)  # (B,1,H,W)
        texture_residual = 2.0 * density_mask - 1.0
        spatial_attention = torch.sigmoid(
            attn_logits + torch.tanh(self.texture_logit_scale) * texture_residual
        )

        # Standardize to a fixed 64x64 attention map for downstream decoder use.
        spatial_attention = F.interpolate(
            spatial_attention, size=(64, 64), mode="bilinear", align_corners=False
        )

        if return_density_mask:
            return spatial_attention, density_mask
        return spatial_attention
