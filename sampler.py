import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        n_rays = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device

        # Step 1: Create Bin Boundary 
        bin_boundaries = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.n_pts_per_ray + 1,
            device=device
        )

        # Step 2: Random Offset from each line
        random_offsets = torch.rand(n_rays, self.n_pts_per_ray, device=device)

        # Step 3: Bin Boundary
        bin_width = (self.max_depth - self.min_depth) / self.n_pts_per_ray
        z_vals = bin_boundaries[:-1] + random_offsets * bin_width

        

        # TODO (Q1.4): Sample points from z values
        sample_points = (
            ray_bundle.origins[:, None, :]
            + ray_bundle.directions[:, None, :] * z_vals[..., None]
        )

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals.unsqueeze(-1).expand_as(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}