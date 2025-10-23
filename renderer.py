import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # --- Step 1: Calculate alpha (opacity) for each segment ---
        # alpha = 1 - exp(-sigma * delta)
        alpha = 1.0 - torch.exp(-rays_density * deltas)

        # --- Step 2: Calculate Transmittance (T) for each point ---
        # T_i = Product of (1 - alpha_j) for all points j *before* point i.
        # This is how much light makes it *to* point i without being absorbed yet.
        alpha_shifted = torch.cat(
            [torch.ones_like(alpha[:, :1]), 1.0 - alpha + eps], dim=1
        ) # Shape: (N_rays, N_pts + 1, 1)

        # `torch.cumprod` calculates the cumulative product along the points dimension (dim=1).
        transmittance = torch.cumprod(
            alpha_shifted, dim=1
        )[:, :-1] # Shape: (N_rays, N_pts, 1) 

        # --- Step 3: Calculate the final weights ---
        # Weight_i = Transmittance_i * Alpha_i
        # How much light reached the point * how much that point absorbed.
        weights = transmittance * alpha # Shape: (N_rays, N_pts, 1)

        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=-2)
        return feature

    def forward(
        self,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            # cur_ray_bundle = sampler(cur_ray_bundle)
            if cur_ray_bundle.sample_points is None:
                raise ValueError("RayBundle must have sample points before rendering.")
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            # Use the _aggregate function to get the final color.
            # Reshape feature to match expected input shape (N_rays, N_pts, N_features)
            feature_reshaped = feature.view(-1, n_pts, feature.shape[-1])
            rendered_feature = self._aggregate(weights, feature_reshaped)

            # TODO (1.5): Render depth map
            depth_reshaped = depth_values.view(-1, n_pts, 1)
            rendered_depth = self._aggregate(weights, depth_reshaped)

            # Return
            cur_out = {
                'feature': rendered_feature,
                'depth': rendered_depth,
                'weights': weights.view(-1, n_pts) 
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters
    
    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        # Initialize points to ray origins
        points = origins
        # Keep track of rays that haven't hit yet (initially all rays)
        mask_unfinished = torch.ones_like(origins[..., 0], dtype=torch.bool)
        # Accumulate distance traveled along each ray
        dist_acc = torch.zeros_like(origins[..., 0])
        # Initialize hit mask (initially no hits)
        mask_hit = torch.zeros_like(origins[..., 0], dtype=torch.bool)

        # Main sphere tracing loop
        for _ in range(self.max_iters):
            # Only process rays that haven't finished yet
            points_unfinished = points[mask_unfinished]

            # --- Intuition: Tap the magic stick ---
            # Query the SDF function to get the distance to the nearest surface
            # for the currently active points.
            distances = implicit_fn(points_unfinished) # Shape: (N_unfinished, 1)

            # --- Intuition: Check if we've hit the wall ---
            # Check if the distance is very small (close to zero).
            # We use a small threshold (e.g., 1e-4 or 1e-5) for numerical stability.
            hit_threshold = 1e-5
            mask_curr_hit = distances.abs() < hit_threshold
            # --- Intuition: Check if we've gone too far ---
            mask_missed = dist_acc[mask_unfinished] > self.far

            # --- Update the unfinished mask ---
            # Original problematic line:
            # mask_unfinished[mask_unfinished] = ~mask_curr_hit.view(-1) & ~mask_missed

            # --- FIX: Calculate the new state directly ---
            # Identify which of the *currently unfinished* rays have now finished (either hit or missed).
            currently_unfinished_indices = torch.where(mask_unfinished)[0]
            finished_in_this_step = mask_curr_hit.view(-1) | mask_missed
            still_unfinished_in_this_step = ~finished_in_this_step
            
            # Update mask_hit for rays that hit in this step
            mask_hit[currently_unfinished_indices[mask_curr_hit.view(-1)]] = True

            # Update the main mask_unfinished tensor at the correct indices.
            mask_unfinished[currently_unfinished_indices[finished_in_this_step]] = False

            # --- Intuition: Take a safe step ---
            # Advance the points along their ray directions by the SDF distance.
            # Only update points that are STILL unfinished (didn't hit or miss).
            step = distances[still_unfinished_in_this_step] * directions[currently_unfinished_indices[still_unfinished_in_this_step]]
            points = points.clone() # Ensure we modify a copy
            points[currently_unfinished_indices[still_unfinished_in_this_step]] += step

            # Update the accumulated distance for rays that are still unfinished
            dist_acc = dist_acc.clone()
            dist_acc[currently_unfinished_indices[still_unfinished_in_this_step]] += distances[still_unfinished_in_this_step].view(-1)

            # --- Early exit if all rays are finished ---
            if not mask_unfinished.any():
                break

        # Return the final points and the mask indicating which rays hit something.
        return points, mask_hit.unsqueeze(-1) # Add channel dimension to mask

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q7): Convert signed distance to density with alpha, beta parameters
    pass

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = None # TODO (Q7): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}
