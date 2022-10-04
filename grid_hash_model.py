from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gridencoder import GridEncoder
from misc import MLP


class GridCLIPModel(nn.Module):
    def __init__(
        self,
        max_coords: Optional[torch.Tensor] = None,
        min_coords: Optional[torch.Tensor] = None,
        mlp_depth: int = 2,
        mlp_width: int = 256,
        batchnorm: bool = False,
        num_levels: int = 16,
        level_dim: int = 8,
        log2_hashmap_size: int = 24,
        per_level_scale: float = 2.0,
        device: str = "cuda",
        image_rep_size: int = 512,
        text_rep_size: int = 512,
        bounds: float = 10.0,
    ):
        super().__init__()

        self._grid_model = GridEncoder(
            input_dim=3,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=16,
            log2_hashmap_size=log2_hashmap_size,
            per_level_scale=per_level_scale,
            desired_resolution=None,
            gridtype="hash",
            align_corners=False,
        )
        # Now convert the output with an MLP
        self._post_grid = MLP(
            input_dim=num_levels * level_dim,
            hidden_dim=mlp_width,
            hidden_depth=mlp_depth,
            output_dim=image_rep_size + text_rep_size,
            batchnorm=batchnorm,
        )
        # Mini MLP for extra storage for image loss
        self._image_head = nn.Identity()
        # Magic value adviced by @imisra
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        self._image_rep_size = image_rep_size
        self._text_rep_size = text_rep_size

        if not (max_coords is not None and min_coords is not None):
            self._max_bounds, self._min_bounds = (
                torch.ones(3) * bounds,
                torch.ones(3) * -bounds,
            )
        else:
            assert len(max_coords) == len(min_coords)
            self._max_bounds, self._min_bounds = max_coords, min_coords

        self._grid_model = self._grid_model.to(device)
        self._post_grid = self._post_grid.to(device)
        self._image_head = self._image_head.to(device)
        self.temperature.data = self.temperature.data.to(device)
        self._max_bounds = self._max_bounds.to(device)
        self._min_bounds = self._min_bounds.to(device)

    def forward(self, x: torch.Tensor, bounds: Optional[float] = None):
        if bounds is None:
            max_bounds, min_bounds = self._max_bounds.to(x.device), self._min_bounds.to(
                x.device
            )
        else:
            max_bounds, min_bounds = (
                torch.ones(3, device=x.device) * bounds,
                torch.ones(3, device=x.device) * -bounds,
            )
        bounded_x = (x - min_bounds) / (max_bounds - min_bounds)
        grid_hash = self._grid_model(bounded_x, bound=1.0)
        result = self._post_grid(grid_hash)
        # label_latent, image_latent = torch.chunk(result, chunks=2, dim=-1)
        label_latent, image_latent = (
            result[..., : self._text_rep_size],
            result[
                ..., self._text_rep_size : self._text_rep_size + self._image_rep_size
            ],
        )
        image_latent = self._image_head(image_latent)
        return label_latent, image_latent

    def to(self, device):
        self._grid_model = self._grid_model.to(device)
        self._post_grid = self._post_grid.to(device)
        self._image_head = self._image_head.to(device)
        self._max_bounds = self._max_bounds.to(device)
        self._min_bounds = self._min_bounds.to(device)
        self.temperature.data = self.temperature.data.to(device)
        return self

    def compute_loss(
        self, predicted_latents, actual_latents, label_mask=None, weights=None
    ):
        normalized_predicted_latents = F.normalize(predicted_latents, p=2, dim=-1)
        normalized_actual_latents = F.normalize(actual_latents, p=2, dim=-1)
        temp = torch.exp(self.temperature)
        sim = (
            torch.einsum(
                "i d, j d -> i j",
                normalized_predicted_latents,
                normalized_actual_latents,
            )
            * temp
        )
        # Zero out the cells where the labels are same.
        if label_mask is not None:
            sim = sim * label_mask
            del label_mask
        labels = torch.arange(len(predicted_latents), device=predicted_latents.device)
        if weights is None:
            loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        else:
            loss = (
                F.cross_entropy(sim, labels, reduction="none")
                + F.cross_entropy(sim.t(), labels, reduction="none")
            ) / 2
            loss = (loss * weights).mean()
        return loss
