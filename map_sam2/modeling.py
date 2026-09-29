"""Model-side building blocks and trainable-module assembly for PGSAM."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from sam2.build_sam import build_sam2
from sam2.modeling.backbones.sgn import SemanticGuidanceNeck


class BoundaryRefinementModule(nn.Module):
    def __init__(
        self,
        stage1_in_channels: int,
        attn_dim: int = 64,
        local_kernel: int = 5,
        downsample_ratio: int = 2,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if downsample_ratio < 1:
            raise ValueError("downsample_ratio must be >= 1.")
        if local_kernel < 1 or local_kernel % 2 == 0:
            raise ValueError("local_kernel must be a positive odd integer.")
        self.attn_dim = int(attn_dim)
        self.local_kernel = int(local_kernel)
        self.downsample_ratio = int(downsample_ratio)
        self.use_checkpoint = bool(use_checkpoint)

        self.stage1_proj = nn.Sequential(
            nn.Conv2d(stage1_in_channels, self.attn_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.attn_dim),
            nn.ReLU(inplace=True),
        )
        self.mask_query = nn.Conv2d(1, self.attn_dim, kernel_size=3, padding=1, bias=False)
        self.atrous1 = nn.Sequential(
            nn.Conv2d(self.attn_dim, self.attn_dim, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(1, self.attn_dim),
            nn.ReLU(inplace=True),
        )
        self.atrous2 = nn.Sequential(
            nn.Conv2d(self.attn_dim, self.attn_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(1, self.attn_dim),
            nn.ReLU(inplace=True),
        )
        self.atrous3 = nn.Sequential(
            nn.Conv2d(self.attn_dim, self.attn_dim, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.GroupNorm(1, self.attn_dim),
            nn.ReLU(inplace=True),
        )
        self.atrous_fuse = nn.Sequential(
            nn.Conv2d(self.attn_dim * 3, self.attn_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.attn_dim),
            nn.ReLU(inplace=True),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(self.attn_dim, self.attn_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, self.attn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.attn_dim, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)
        lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap_kernel, persistent=False)
        self.last_debug: Dict[str, torch.Tensor] = {}

    def _residual_ms_block(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.atrous1(x)
        b2 = self.atrous2(x)
        b3 = self.atrous3(x)
        y = self.atrous_fuse(torch.cat([b1, b2, b3], dim=1))
        return x + y

    def _boundary_focus(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        edge = F.conv2d(probs, self.lap_kernel.to(dtype=logits.dtype), padding=1).abs()
        edge = F.avg_pool2d(
            edge,
            kernel_size=self.local_kernel,
            stride=1,
            padding=self.local_kernel // 2,
        )
        e_min = edge.amin(dim=(2, 3), keepdim=True)
        e_max = edge.amax(dim=(2, 3), keepdim=True)
        return (edge - e_min) / (e_max - e_min + 1e-6)

    def _boundary_mask(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # A differentiable, symmetric uncertainty band lets the refiner move
        # the contour both inward and outward. The previous hard 0.5 mask
        # blocked gradients and could not recover regions omitted upstream.
        uncertainty = 4.0 * probs * (1.0 - probs)
        local_edge = F.max_pool2d(probs, 3, stride=1, padding=1) - (
            1.0 - F.max_pool2d(1.0 - probs, 3, stride=1, padding=1)
        )
        boundary = torch.maximum(uncertainty, local_edge.clamp(0.0, 1.0))
        return F.max_pool2d(
            boundary,
            kernel_size=self.local_kernel,
            stride=1,
            padding=self.local_kernel // 2,
        )

    def forward(self, low_res_logits: torch.Tensor, stage1_feat: torch.Tensor) -> torch.Tensor:
        target_hw = low_res_logits.shape[-2:]
        if stage1_feat.shape[-2:] != target_hw:
            stage1_feat = F.interpolate(stage1_feat, size=target_hw, mode="bilinear", align_corners=False)

        if self.downsample_ratio > 1:
            low_res_small = F.interpolate(
                low_res_logits,
                scale_factor=1.0 / float(self.downsample_ratio),
                mode="bilinear",
                align_corners=False,
            )
            stage1_small = F.interpolate(
                stage1_feat,
                size=low_res_small.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            low_res_small = low_res_logits
            stage1_small = stage1_feat

        boundary_focus = self._boundary_focus(low_res_small)
        boundary_mask = self._boundary_mask(low_res_small)
        q = torch.sigmoid(self.mask_query(low_res_small))
        s1 = self.stage1_proj(stage1_small)
        local_ctx = s1 * q * boundary_mask
        if self.use_checkpoint and self.training:
            local_ctx = torch_checkpoint(self._residual_ms_block, local_ctx, use_reentrant=False)
        else:
            local_ctx = self._residual_ms_block(local_ctx)
        local_ctx = local_ctx * boundary_focus * boundary_mask
        delta = self.refine_head(local_ctx) * boundary_mask
        if self.downsample_ratio > 1:
            delta = F.interpolate(delta, size=target_hw, mode="bilinear", align_corners=False)
            boundary_mask = F.interpolate(
                boundary_mask, size=target_hw, mode="bilinear", align_corners=False
            )
        delta = delta * boundary_mask
        self.last_debug = {
            "refiner_boundary_focus_mean": boundary_focus.detach().mean(),
            "refiner_boundary_focus_max": boundary_focus.detach().max(),
            "refiner_boundary_mask_ratio": boundary_mask.detach().mean(),
            "refiner_delta_abs_mean": delta.detach().abs().mean(),
            "refiner_delta_abs_max": delta.detach().abs().max(),
        }
        return low_res_logits + delta


class GeometricResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.act(x + y)


class DepthwiseGeometricResidualBlock(nn.Module):
    """Full-resolution residual block with substantially lower memory and FLOPs."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, channels)
        self.depthwise2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pointwise1(self.depthwise1(x))
        y = self.act(self.norm1(y))
        y = self.pointwise2(self.depthwise2(y))
        y = self.norm2(y)
        return self.act(x + y)


class FullResGeometricAligner(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 48,
        num_blocks: int = 5,
        block_type: str = "standard",
    ):
        super().__init__()
        if block_type not in {"standard", "depthwise"}:
            raise ValueError("block_type must be 'standard' or 'depthwise'.")
        self.block_type = block_type
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        block_cls = (
            DepthwiseGeometricResidualBlock
            if block_type == "depthwise"
            else GeometricResidualBlock
        )
        self.blocks = nn.Sequential(*[block_cls(hidden_channels) for _ in range(int(num_blocks))])
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
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

    @staticmethod
    def _sobel_rgb(image: torch.Tensor) -> torch.Tensor:
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(image, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        gy = F.conv2d(image, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return mag.amax(dim=1, keepdim=True)

    @staticmethod
    def _local_std(image: torch.Tensor, k: int = 5) -> torch.Tensor:
        mean = F.avg_pool2d(image, kernel_size=k, stride=1, padding=k // 2)
        mean2 = F.avg_pool2d(image * image, kernel_size=k, stride=1, padding=k // 2)
        var = torch.clamp(mean2 - mean * mean, min=0.0)
        std = torch.sqrt(var + 1e-6)
        return std.mean(dim=1, keepdim=True)

    def prepare_image_context(self, image_1024: torch.Tensor) -> torch.Tensor:
        sobel_x = self.sobel_x.to(dtype=image_1024.dtype)
        sobel_y = self.sobel_y.to(dtype=image_1024.dtype)
        gx = F.conv2d(image_1024, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        gy = F.conv2d(image_1024, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        sobel = torch.sqrt(gx * gx + gy * gy + 1e-6).amax(dim=1, keepdim=True)
        high_frequency = F.max_pool2d(sobel, kernel_size=3, stride=1, padding=1)
        local_std = self._local_std(image_1024, k=5)
        return torch.cat([image_1024, high_frequency, local_std], dim=1)

    def forward(
        self,
        coarse_logits_1024: torch.Tensor,
        image_1024: torch.Tensor,
        image_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prob = torch.sigmoid(coarse_logits_1024)
        if image_context is None:
            image_context = self.prepare_image_context(image_1024)
        x = torch.cat([coarse_logits_1024, prob, image_context], dim=1)
        f = self.stem(x)
        f = self.blocks(f)
        delta = self.head(f)
        return coarse_logits_1024 + delta


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dev = base_layer.weight.device
        self.lora_A = nn.Parameter(torch.zeros(self.r, in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.lora_A.device != x.device:
            self.lora_A.data = self.lora_A.data.to(device=x.device)
            self.lora_B.data = self.lora_B.data.to(device=x.device)
        lora_intermediate = F.linear(x, self.lora_A.to(dtype=x.dtype))
        lora_out = F.linear(lora_intermediate, self.lora_B.to(dtype=x.dtype))
        return base_out + lora_out * self.scaling


def apply_lora_to_hiera_qkv(trunk: nn.Module, r: int = 16, alpha: float = 32.0) -> int:
    wrapped = 0
    for blk in trunk.blocks:
        if hasattr(blk, "attn") and hasattr(blk.attn, "qkv"):
            qkv = blk.attn.qkv
            if isinstance(qkv, nn.Linear):
                blk.attn.qkv = LoRALinear(qkv, r=r, alpha=alpha)
                wrapped += 1
    if wrapped == 0:
        raise RuntimeError("No qkv linear layers were wrapped with LoRA.")
    return wrapped


def get_lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for name, submodule in module.named_modules():
        if isinstance(submodule, LoRALinear):
            state[f"{name}.lora_A"] = submodule.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = submodule.lora_B.detach().cpu()
            state[f"{name}.alpha"] = torch.tensor(submodule.alpha, dtype=torch.float32)
            state[f"{name}.r"] = torch.tensor(submodule.r, dtype=torch.int32)
    return state


def freeze_all_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def _load_checkpoint_non_strict(model: nn.Module, ckpt_path: str) -> None:
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed = tuple('sam_mask_decoder.' + name for name in (
        'dc1.', 'dc2.', 'ln1.', 'fuse_s0.', 'fuse_s1.',
        'implicit_prompt_residual_scale', 'highres_gate_logit_'))
    required_missing = [name for name in missing if not name.startswith(allowed)]
    if required_missing or unexpected:
        raise RuntimeError(f'Incompatible pretrained checkpoint: missing={required_missing}, unexpected={unexpected}')
    print(
        f"[checkpoint] loaded non-strictly from {ckpt_path} | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def build_trainable_modules(args, device: torch.device) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """Build the trainable SAM2 extension modules without changing the method definition."""
    config_name = str(args.config)
    if config_name.startswith("sam2/configs/"):
        config_name = config_name[len("sam2/") :]
    if config_name.startswith("./"):
        config_name = config_name[2:]

    model = build_sam2(
        config_file=config_name,
        ckpt_path=None,
        device=device.type,
        mode="train",
        apply_postprocessing=False,
        hydra_overrides_extra=[],
    )
    _load_checkpoint_non_strict(model, args.checkpoint)

    freeze_all_params(model)
    freeze_all_params(model.image_encoder.trunk)
    freeze_all_params(model.image_encoder.neck)

    if args.use_lora:
        wrapped = apply_lora_to_hiera_qkv(model.image_encoder.trunk, r=args.lora_rank, alpha=args.lora_alpha)
        print(f"[LoRA] wrapped {wrapped} Hiera qkv layers (r={args.lora_rank}, alpha={args.lora_alpha})")
    else:
        print("[LoRA] disabled for this ablation preset")

    unfreeze_params(model.sam_mask_decoder)
    highres_fusion_mode = str(
        getattr(args, "highres_fusion_mode", "legacy_concat")
    )
    model.sam_mask_decoder.highres_fusion_mode = highres_fusion_mode
    model.sam_mask_decoder.implicit_prompt_gate_floor = float(
        getattr(args, "decoder_prompt_gate_floor", 0.0)
    )
    model.sam_mask_decoder.implicit_prompt_mode = str(
        getattr(args, "implicit_prompt_mode", "multiplicative")
    )
    model.pgsam_normalize_inputs = bool(
        getattr(args, "sam2_normalize_inputs", False)
    )
    model.pgsam_prompt_policy = getattr(args, "prompt_policy", "legacy")
    model.pgsam_initial_implicit = getattr(args, "use_initial_implicit_prompt", None)
    model.pgsam_final_implicit = getattr(args, "use_final_implicit_prompt", None)
    model.pgsam_prompt_refinement = getattr(args, "prompt_refinement_mode", "global_local")
    model.pgsam_point_conf_ratio = getattr(args, "point_confidence_ratio", .6)
    model.pgsam_point_min_conf = getattr(args, "point_min_confidence", .5)

    stage4_channels = model.image_encoder.trunk.channel_list[0]
    sgn = SemanticGuidanceNeck(
        in_channels=stage4_channels,
        out_channels=model.sam_prompt_embed_dim,
        use_layout_prior=bool(getattr(args, "use_layout_prior", False)),
    ).to(device)
    if args.prompt_source == "sgn_auto":
        unfreeze_params(sgn)
    else:
        freeze_all_params(sgn)

    if hasattr(model.sam_mask_decoder, "conv_s0"):
        fpn_channels = model.sam_prompt_embed_dim
        if model.sam_mask_decoder.conv_s0.in_channels != fpn_channels:
            model.sam_mask_decoder.conv_s0 = nn.Conv2d(
                fpn_channels,
                model.sam_prompt_embed_dim // 8,
                kernel_size=1,
                stride=1,
            ).to(device)
        if model.sam_mask_decoder.conv_s1.in_channels != fpn_channels:
            model.sam_mask_decoder.conv_s1 = nn.Conv2d(
                fpn_channels,
                model.sam_prompt_embed_dim // 4,
                kernel_size=1,
                stride=1,
            ).to(device)

        # Both implementations remain instantiated so archived v4 checkpoints
        # can still be loaded.  Only the branch used by the selected version is
        # trainable, which also keeps optimizer diagnostics unambiguous.
        if highres_fusion_mode == "residual_additive":
            for legacy_name in ("dc1", "ln1", "dc2", "fuse_s0", "fuse_s1"):
                legacy_module = getattr(model.sam_mask_decoder, legacy_name, None)
                if legacy_module is not None:
                    freeze_all_params(legacy_module)
            active_s0 = args.highres_feature_mode in {"stage1", "stage1_stage2"}
            active_s1 = args.highres_feature_mode in {"stage2", "stage1_stage2"}
            if not active_s0:
                freeze_all_params(model.sam_mask_decoder.conv_s0)
                model.sam_mask_decoder.highres_gate_logit_s0.requires_grad = False
            if not active_s1:
                freeze_all_params(model.sam_mask_decoder.conv_s1)
                model.sam_mask_decoder.highres_gate_logit_s1.requires_grad = False
        elif highres_fusion_mode == "legacy_concat":
            freeze_all_params(model.sam_mask_decoder.output_upscaling)
            model.sam_mask_decoder.highres_gate_logit_s0.requires_grad = False
            model.sam_mask_decoder.highres_gate_logit_s1.requires_grad = False
        else:
            raise ValueError(f"Unsupported high-resolution fusion mode: {highres_fusion_mode}")

    refiner = BoundaryRefinementModule(
        stage1_in_channels=model.sam_prompt_embed_dim,
        attn_dim=args.refine_attn_dim,
        local_kernel=args.refine_local_kernel,
        downsample_ratio=args.refine_downsample_ratio,
        use_checkpoint=args.refine_use_checkpoint,
    ).to(device)
    if args.use_boundary_refinement:
        unfreeze_params(refiner)
    else:
        freeze_all_params(refiner)

    full_res_refiner = FullResGeometricAligner(
        in_channels=7,
        hidden_channels=args.fullres_hidden_dim,
        num_blocks=args.fullres_blocks,
        block_type=getattr(args, "fullres_block_type", "standard"),
    ).to(device)
    if args.use_fullres_refinement:
        unfreeze_params(full_res_refiner)
    else:
        freeze_all_params(full_res_refiner)

    return model, sgn, refiner, full_res_refiner


def collect_hybrid_trainable_params(
    backbone: nn.Module,
    sgn: nn.Module,
    decoder: nn.Module,
    refiner: nn.Module,
    full_res_refiner: Optional[nn.Module] = None,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Collect LoRA parameters and all currently enabled non-backbone parameters."""
    lora_params: List[nn.Parameter] = []
    for module in backbone.modules():
        if isinstance(module, LoRALinear) and module.lora_A.requires_grad:
            lora_params.extend([module.lora_A, module.lora_B])

    lora_param_ids = {id(p) for p in lora_params}
    for _, p in backbone.named_parameters():
        if p.requires_grad and id(p) not in lora_param_ids:
            p.requires_grad = False

    module_params: List[nn.Parameter] = []
    for module in (sgn, decoder, refiner, full_res_refiner):
        if module is None:
            continue
        module_params.extend([p for p in module.parameters() if p.requires_grad])
    return lora_params, module_params
