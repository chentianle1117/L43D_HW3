import os
import warnings

import matplotlib.pyplot as plt

import hydra
import numpy as np
import torch
import tqdm
import imageio

from omegaconf import DictConfig
from PIL import Image
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
import matplotlib.pyplot as plt

from implicit import implicit_dict
from sampler import sampler_dict
from renderer import renderer_dict
from ray_utils import (
    sample_images_at_xy,
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_rays_from_pixels,
    RayBundle
)
from data_utils import (
    dataset_from_config,
    create_surround_cameras,
    vis_grid,
    vis_rays,
)
from dataset import (
    get_nerf_datasets,
    trivial_collate,
)

from render_functions import render_points_with_save

# Model class containing:
#   1) Implicit volume defining the scene
#   2) Sampling scheme which generates sample points along rays
#   3) Renderer which can render an implicit volume given a sampling scheme

class Model(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        # # Get implicit function from config
        # self.implicit_fn = implicit_dict[cfg.implicit_function.type](
        #     cfg.implicit_function
        # )

        # # Point sampling (raymarching) scheme
        # self.sampler = sampler_dict[cfg.sampler.type](
        #     cfg.sampler
        # )

        # # Initialize volume renderer
        # self.renderer = renderer_dict[cfg.renderer.type](
        #     cfg.renderer
        # )

        # --- Coarse Network ---
        # Load the implicit function (e.g., NeRF) using the main config section
        self.implicit_fn_coarse = implicit_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )

        # --- Fine Network ---
        # Load a second implicit function. Often uses the same architecture.
        # Check if a specific config section exists for the fine network, otherwise reuse the coarse one.
        fine_cfg_section = cfg.implicit_function_fine if 'implicit_function_fine' in cfg else cfg.implicit_function
        self.implicit_fn_fine = implicit_dict[cfg.implicit_function.type](
            fine_cfg_section
        )

        # --- Coarse Sampler ---
        # Uses the 'stratified' sampler type and settings from `cfg.sampler_coarse`
        # (Make sure your config file has a `sampler_coarse` section)
        self.sampler_coarse = sampler_dict['stratified'](
            cfg.sampler_coarse
        )

        # --- Fine Sampler ---
        # Uses the 'pdf' sampler type and settings from `cfg.sampler_fine`
        # (Make sure your config file has a `sampler_fine` section defining 'pdf' type)
        self.sampler_fine = sampler_dict['pdf'](
             cfg.sampler_fine
        )

        # --- Renderer ---
        # We use the same VolumeRenderer instance for both passes.
        self.renderer = renderer_dict[cfg.renderer.type](
            cfg.renderer
        )

    # def forward(
    #     self,
    #     ray_bundle
    # ):
    #     # Call renderer with
    #     #  a) Implicit volume
    #     #  b) Sampling routine

    #     return self.renderer(
    #         self.sampler,
    #         self.implicit_fn,
    #         ray_bundle
    #     )
    def forward(self, ray_bundle):
        # --- 1. Coarse Pass ---
        # Sample points using the coarse (stratified) sampler.
        ray_bundle_coarse = self.sampler_coarse(ray_bundle)

        # Render using the coarse network and coarse samples.
        # We need to modify the renderer's forward method slightly to accept 'coarse_pass=True'
        # and to ensure it uses the correct implicit function and sampler internally.
        # For simplicity here, let's assume the renderer's forward directly takes these:
        rendered_coarse_output = self.renderer(
            implicit_fn=self.implicit_fn_coarse, # Pass the coarse network
            ray_bundle=ray_bundle_coarse # Pass the coarse samples
        )
        # Store the weights from the coarse pass, needed for fine sampling.
        coarse_weights = rendered_coarse_output['weights'] # Shape: (N_rays, N_coarse_pts)

        # --- 2. Fine Sampling (Importance Sampling) ---
        # Use the PDF sampler, feeding it the original ray bundle and the coarse weights.
        # The PDF sampler calculates new sample points biased towards important regions.
        ray_bundle_fine = self.sampler_fine(
            ray_bundle_coarse, coarse_weights # Pass coarse bundle (for z_vals) and weights
        )

        # --- 3. Fine Pass ---
        # Render using the fine network and the combined (coarse + fine) samples.
        rendered_fine_output = self.renderer(
            implicit_fn=self.implicit_fn_fine, # Pass the fine network
            ray_bundle=ray_bundle_fine # Pass the fine samples
        )

        # Return results from both passes (needed for the combined loss).
        return {
            'coarse': rendered_coarse_output,
            'fine': rendered_fine_output,
        }


def render_images(
    model,
    cameras,
    image_size,
    save=False,
    file_prefix=''
):
    all_images = []
    device = list(model.parameters())[0].device

    for cam_idx, camera in enumerate(cameras):
        print(f'Rendering image {cam_idx}')

        torch.cuda.empty_cache()
        camera = camera.to(device)
        # TODO (Q1.3): implement in ray_utils.py
        xy_grid = get_pixels_from_image(image_size, camera)
        # TODO (Q1.3): implement in ray_utils.py
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)

        # TODO (Q1.3): Visualize xy grid using vis_grid
        if cam_idx == 0 and file_prefix == '':
            # 1. Call vis_grid to get the image data as a NumPy array.
            #    Remember to reshape the grid for the function.
            grid_vis_array = vis_grid(
                xy_grid.view(image_size[1], image_size[0], 2),
                image_size
            )
            # 2. Use plt.imsave to save that array to a file.
            plt.imsave("images/grid_visualization.png", grid_vis_array)

        # TODO (Q1.3): Visualize rays using vis_rays
        if cam_idx == 0 and file_prefix == '':
            # 1. Call vis_rays to get the ray visualization data.
            rays_vis_array = vis_rays(ray_bundle, image_size)
            # 2. Save that array to a file.
            plt.imsave("images/rays_visualization.png", rays_vis_array)

        # TODO (Q1.4): Implement point sampling along rays in sampler.py
        pass

        # TODO (Q1.4): Visualize sample points as point cloud
        if cam_idx == 0 and file_prefix == '':
            # 1. Explicitly call the sampler to generate sample points
            sampler = model.sampler
            #    Make a temporary copy of the ray_bundle 
            vis_ray_bundle = RayBundle(
                origins=ray_bundle.origins.clone(),
                directions=ray_bundle.directions.clone(),
                sample_points=None,
                sample_lengths=None,
            )
            vis_ray_bundle = sampler(vis_ray_bundle)

            # 2. Reshape the points for the rendering function 
            points_to_render = vis_ray_bundle.sample_points.view(1, -1, 3)

            # 3. Call render_points_with_save
            render_points_with_save(
                points=points_to_render,
                cameras=[camera], # Pass the current camera in a list
                image_size=image_size,
                save=True, # Tell the function to save the image
                file_prefix="images/point_cloud_visualization" # Base filename
            )

        # TODO (Q1.5): Implement rendering in renderer.py
        out = model(ray_bundle)

        # Return rendered features (colors)
        image = np.array(
            out['fine']['feature'].view(
                image_size[1], image_size[0], 3
            ).detach().cpu()
        )
        all_images.append(image)

        # TODO (Q1.5): Visualize depth
        if cam_idx == 2 and file_prefix == '':
            # 1. Get the depth tensor from the renderer's output.
            #    The shape is likely (H*W, 1).
            depth_tensor = out['fine']['depth']

            # 2. Reshape the depth tensor to the image dimensions (H, W).
            depth_map = depth_tensor.view(image_size[1], image_size[0])

            # 3. Move tensor to CPU and convert to NumPy array for saving.
            depth_map_np = depth_map.detach().cpu().numpy()

            # 4. Normalize the depth map to the [0, 1] range.
            #    This makes closer pixels darker and farther pixels brighter (or vice versa depending on cmap).
            depth_map_normalized = (depth_map_np - depth_map_np.min()) / (depth_map_np.max() - depth_map_np.min())

            # 5. Save the normalized depth map as an image using matplotlib.
            #    `cmap='viridis'` gives a color map similar to the example, 'gray' gives grayscale.
            plt.imsave("images/depth_visualization.png", depth_map_normalized, cmap='viridis')

        # Save
        if save:
            plt.imsave(
                f'{file_prefix}_{cam_idx}.png',
                image
            )

    return all_images


def render(
    cfg,
):
    # Create model
    model = Model(cfg)
    model = model.cuda()
    model.eval()

    # Render spiral
    cameras = create_surround_cameras(5.0, n_poses=20)
    all_images = render_images(
        model, cameras, cfg.data.image_size
    )
    imageio.mimsave('images/part_1.gif',
                    [np.uint8(im * 255) for im in all_images], loop=0)


def train(
    cfg
):
    # Create model
    model = Model(cfg)
    model = model.cuda()
    model.train()

    # Create dataset
    train_dataset = dataset_from_config(cfg.data)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    image_size = cfg.data.image_size

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr
    )

    # Render images before training
    cameras = [item['camera'] for item in train_dataset]
    render_images(
        model, cameras, image_size,
        save=True, file_prefix='images/part_2_before_training'
    )

    # Train
    t_range = tqdm.tqdm(range(cfg.training.num_epochs))

    for epoch in t_range:
        for iteration, batch in enumerate(train_dataloader):
            image, camera, camera_idx = batch[0].values()
            image = image.cuda()
            camera = camera.cuda()

            # Sample rays
            # TODO (Q2.1): implement in ray_utils.py
            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, image_size, camera)
            ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Run model forward
            out = model(ray_bundle)

            # # TODO (Q2.2): Calculate loss (normal version)
            # loss = torch.nn.functional.mse_loss(out['feature'], rgb_gt)

            # TODO (Q4.2 / Q3.1): Calculate combined loss
            # Calculate MSE loss for the coarse prediction
            loss_coarse = torch.nn.functional.mse_loss(
                out['coarse']['feature'], rgb_gt
            )
            # Calculate MSE loss for the fine prediction
            loss_fine = torch.nn.functional.mse_loss(
                out['fine']['feature'], rgb_gt
            )
            # The total loss is the sum of both
            loss = loss_coarse + loss_fine

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % 10) == 0:
            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

    # Print center and side lengths
    print("Box center:", tuple(
        np.array(model.implicit_fn.sdf.center.data.detach().cpu()).tolist()[0]))
    print("Box side lengths:", tuple(
        np.array(model.implicit_fn.sdf.side_lengths.data.detach().cpu()).tolist()[0]))

    # Render images after training
    render_images(
        model, cameras, image_size,
        save=True, file_prefix='images/part_2_after_training'
    )
    all_images = render_images(
        model, create_surround_cameras(3.0, n_poses=20), image_size, file_prefix='part_2'
    )
    imageio.mimsave('images/part_2.gif',
                    [np.uint8(im * 255) for im in all_images], loop=0)


def create_model(cfg):
    # Create model
    model = Model(cfg)
    model.cuda()
    model.train()

    # Load checkpoints
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.training.checkpoint_path
    )

    if len(cfg.training.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.training.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return cfg.training.lr_scheduler_gamma ** (
            epoch / cfg.training.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    return model, optimizer, lr_scheduler, start_epoch, checkpoint_path


def train_nerf(
    cfg
):
    # Create model
    model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(
        cfg)

    # Load the training/validation data.
    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            image, camera, camera_idx = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, cfg.data.image_size, camera
            )
            ray_bundle = get_rays_from_pixels(
                xy_grid, cfg.data.image_size, camera
            )
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Run model forward
            out = model(ray_bundle)

            # TODO (Q3.1): Calculate loss
            # loss = torch.nn.functional.mse_loss(out['feature'], rgb_gt)

            # Calculate MSE loss for the coarse prediction
            loss_coarse = torch.nn.functional.mse_loss(
                out['coarse']['feature'], rgb_gt # Access coarse feature
            )
            # Calculate MSE loss for the fine prediction
            loss_fine = torch.nn.functional.mse_loss(
                out['fine']['feature'], rgb_gt   # Access fine feature
            )
            # The total loss is the sum of both
            loss = loss_coarse + loss_fine

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

        # Adjust the learning rate.
        lr_scheduler.step()

        # Checkpoint.
        if (
            epoch % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        # Render
        if (
            epoch % cfg.training.render_interval == 0
            and epoch > 0
        ):
            with torch.no_grad():
                test_images = render_images(
                    model, create_surround_cameras(
                        4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
                    cfg.data.image_size, file_prefix='nerf'
                )
                imageio.mimsave(
                    'images/part_3.gif', [np.uint8(im * 255) for im in test_images], loop=0)


@hydra.main(config_path='./configs', config_name='sphere')
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.type == 'render':
        render(cfg)
    elif cfg.type == 'train':
        train(cfg)
    elif cfg.type == 'train_nerf':
        train_nerf(cfg)


if __name__ == "__main__":
    main()
