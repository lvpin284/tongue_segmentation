# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tokenize import Double
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d

import matplotlib.pyplot as plt
import cv2


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)


        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Prior input channels: 4 shape prototypes + 1 temporal (previous-frame) mask.
        # Cardiac callers still pass 4 channels; the forward pass zero-pads the missing
        # temporal channel so this branch stays backward compatible.
        self.prior_in_channels = 5
        self.cls_sim_avg_label_enc1 = CBR(self.prior_in_channels, 32)
        self.cls_sim_avg_label_enc2 = CBR(32, 64)


        self.cls_sim_avg_label_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cls_sim_avg_label_bottleneck = CBR(64, 128)
        self.cls_sim_avg_label_upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.cls_sim_avg_label_dec2 = CBR(128, 64)
        self.cls_sim_avg_label_upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.cls_sim_avg_label_dec1 = CBR(64, 32)
        self.cls_sim_avg_label_conv_last = nn.Conv2d(32, 1, kernel_size=1)

        # SPM: a lightweight statistical-prior branch built from the prior input
        # (4 shape prototypes + 1 temporal mask channel).
        self.spm_encoder = nn.Sequential(
            nn.Conv2d(self.prior_in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.spm_shape_head = nn.Conv2d(32, 1, kernel_size=1)
        # Predict per-pixel fusion weight alpha in [0, 1] for coarse-vs-prior blending.
        self.spm_fusion_gate = nn.Conv2d(2, 1, kernel_size=1)


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_cls(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        cls_repeated: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight

        point_embedding[:, 1, :] = cls_repeated[:, 0, :]

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        # cls_sim: Optional[torch.Tensor],
        cls_sim_avg_label_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates (b N_points 2)
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed (b 4)
          masks (torch.Tensor or none): masks to embed (b 1 h w)

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if cls_sim_avg_label_input is None:
            # Keep compatibility with vanilla SAM-style calls that provide no prototype prior.
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
            zero_logit = torch.zeros(
                (bs, 1, self.mask_input_size[0] // 2, self.mask_input_size[1] // 2),
                device=self._get_device(),
                dtype=dense_embeddings.dtype,
            )
            zero_alpha = torch.zeros_like(zero_logit)
            return sparse_embeddings, dense_embeddings, zero_logit, zero_logit, zero_alpha

        # Accept either 4 channels (shape prototypes only, e.g. cardiac callers) or
        # 5 channels (4 prototypes + 1 temporal previous-frame mask). Zero-pad the
        # temporal channel when it is absent so the branch weights stay compatible.
        if cls_sim_avg_label_input.shape[1] == self.prior_in_channels - 1:
            temporal_pad = torch.zeros_like(cls_sim_avg_label_input[:, :1])
            cls_sim_avg_label_input = torch.cat([cls_sim_avg_label_input, temporal_pad], dim=1)

        enc1 = self.cls_sim_avg_label_enc1(cls_sim_avg_label_input)
        enc2 = self.cls_sim_avg_label_enc2(self.cls_sim_avg_label_pool(enc1))
        bottleneck = self.cls_sim_avg_label_bottleneck(self.cls_sim_avg_label_pool(enc2))
        dec2 = self.cls_sim_avg_label_upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.cls_sim_avg_label_dec2(dec2)
        dec1 = self.cls_sim_avg_label_upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.cls_sim_avg_label_dec1(dec1)

        coarse_mask_logit = self.cls_sim_avg_label_conv_last(dec1)
        low_coarse_mask_logit = F.interpolate(coarse_mask_logit, (128, 128), mode="bilinear", align_corners=False)
        low_coarse_mask_prob = torch.sigmoid(low_coarse_mask_logit)

        spm_feat = self.spm_encoder(cls_sim_avg_label_input)
        spm_shape_logit = self.spm_shape_head(spm_feat)
        spm_shape_logit = F.interpolate(spm_shape_logit, (128, 128), mode="bilinear", align_corners=False)
        spm_shape_prob = torch.sigmoid(spm_shape_logit)

        # Shape prototype prior comes only from the 4 prototype channels; the 5th
        # (temporal) channel is excluded here since it already flows through the
        # coarse U-Net and SPM encoders above.
        prototype_prior = F.interpolate(
            cls_sim_avg_label_input[:, :4].mean(dim=1, keepdim=True),
            (128, 128),
            mode="bilinear",
            align_corners=False,
        )
        prototype_min = prototype_prior.amin(dim=(-2, -1), keepdim=True)
        prototype_max = prototype_prior.amax(dim=(-2, -1), keepdim=True)
        prototype_prior = (prototype_prior - prototype_min) / (prototype_max - prototype_min + 1e-6)

        hybrid_prior = (0.5 * prototype_prior + 0.5 * spm_shape_prob).clamp(1e-4, 1 - 1e-4)
        alpha = torch.sigmoid(self.spm_fusion_gate(torch.cat([low_coarse_mask_prob, hybrid_prior], dim=1)))
        refined_coarse_prob = (alpha * low_coarse_mask_prob + (1.0 - alpha) * hybrid_prior).clamp(1e-4, 1 - 1e-4)
        refined_coarse_logit = torch.logit(refined_coarse_prob)

        if coarse_mask_logit is not None:
            dense_embeddings = self._embed_masks(refined_coarse_prob)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings, refined_coarse_logit, spm_shape_logit, alpha


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float32))  # B x N x C
