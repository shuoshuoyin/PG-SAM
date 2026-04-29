# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        # If you pass raw Hiera stage features to `high_res_features`, set these so we
        # can project them to the internal mask-decoder channel sizes.
        stage1_in_channels: Optional[int] = None,  # 1/4 resolution feature channels
        stage2_in_channels: Optional[int] = None,  # 1/8 resolution feature channels
        implicit_prompt_in_dim: Optional[int] = None,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            stage1_in_channels = (
                transformer_dim if stage1_in_channels is None else stage1_in_channels
            )
            stage2_in_channels = (
                transformer_dim if stage2_in_channels is None else stage2_in_channels
            )
            self.conv_s0 = nn.Conv2d(
                stage1_in_channels, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                stage2_in_channels, transformer_dim // 4, kernel_size=1, stride=1
            )
            # When concatenating skip features, fuse them back to match the
            # channel size expected by the next upsampling stage.
            self.fuse_s1 = nn.Sequential(
                nn.Conv2d(
                    transformer_dim // 2,
                    transformer_dim // 4,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(transformer_dim // 4),
                activation(),
            )
            self.fuse_s0 = nn.Sequential(
                nn.Conv2d(
                    transformer_dim // 4,
                    transformer_dim // 8,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(transformer_dim // 8),
                activation(),
            )

            # Rebuild the upscaling path so we can concatenate skip features
            # after each intermediate upsampling step.
            self.output_upscaling = None
            self.dc1 = nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            )
            self.ln1 = LayerNorm2d(transformer_dim // 4)
            self.act1 = activation()
            self.dc2 = nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            )
            self.act2 = activation()

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        # Optional projection for SGN implicit prompt maps.
        if implicit_prompt_in_dim is None or implicit_prompt_in_dim == transformer_dim:
            self.implicit_prompt_proj = nn.Identity()
        else:
            self.implicit_prompt_proj = nn.Conv2d(
                implicit_prompt_in_dim, transformer_dim, kernel_size=1, bias=False
            )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        implicit_prompt_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        spatial_attn = None
        if implicit_prompt_map is not None:
            if implicit_prompt_map.dim() != 4:
                raise ValueError(
                    f"implicit_prompt_map must be (B,C,H,W), got {implicit_prompt_map.shape}"
                )
            # SGN now outputs a spatial attention map (ideally Bx1x64x64).
            # If multiple channels are given, collapse to one attention channel.
            if implicit_prompt_map.shape[1] != 1:
                implicit_prompt_map = implicit_prompt_map.mean(dim=1, keepdim=True)
            spatial_attn = implicit_prompt_map.to(
                dtype=image_embeddings.dtype, device=image_embeddings.device
            ).clamp(0.0, 1.0)

        # Spatial modulation before transformer (Stage 3 / 1/16 features).
        if spatial_attn is not None:
            attn_s3 = F.interpolate(
                spatial_attn,
                size=image_embeddings.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            image_embeddings = image_embeddings * attn_s3

            # Gated skip-connections:
            # use the same SGN spatial prior to filter Stage-1/2 high-res features
            # before concatenation in the decoder upsampling path.
            if high_res_features is not None and len(high_res_features) == 2:
                feat_s0, feat_s1 = high_res_features
                gate_s0 = F.interpolate(
                    spatial_attn,
                    size=feat_s0.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                gate_s1 = F.interpolate(
                    spatial_attn,
                    size=feat_s1.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                high_res_features = [feat_s0 * gate_s0, feat_s1 * gate_s1]

        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Always output a single mask token (single high-resolution binary mask in downstream usage).
        masks = masks[:, 0:1, :, :]
        iou_pred = iou_pred[:, 0:1]
        sam_tokens_out = mask_tokens_out[:, 0:1]  # (B,1,C)

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            if high_res_features is None or len(high_res_features) != 2:
                raise ValueError(
                    "When use_high_res_features=True, high_res_features must be [feat_s0, feat_s1]."
                )
            feat_s0, feat_s1 = high_res_features

            # Stage-2 (1/8 res): fuse at the intermediate upsampling step.
            # Support both raw (transformer_dim channels) and pre-projected features.
            if feat_s0.shape[1] == self.conv_s0.in_channels:
                feat_s0 = self.conv_s0(feat_s0)
            if feat_s1.shape[1] == self.conv_s1.in_channels:
                feat_s1 = self.conv_s1(feat_s1)

            up1 = self.dc1(src)  # (B, C/4, 2H, 2W)
            up1 = self.act1(self.ln1(up1))
            up1 = self.fuse_s1(torch.cat([up1, feat_s1], dim=1))

            # Stage-1 (1/4 res): fuse at the final upsampling step.
            up2 = self.dc2(up1)  # (B, C/8, 4H, 4W)
            up2 = self.act2(up2)
            upscaled_embedding = self.fuse_s0(torch.cat([up2, feat_s0], dim=1))

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
