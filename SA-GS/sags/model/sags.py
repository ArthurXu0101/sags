# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from pathlib import Path
import csv
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat.cuda_legacy._wrapper import num_sh_bases
from pytorch_msssim import SSIM
from torch.nn import Parameter
from nerfstudio.data.scene_box import OrientedBox

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    if image.dim() == 3:
        return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)
    else:
        return tf.conv2d(image[None, None, ...], weight, stride=d).squeeze(0).squeeze(0)
        


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class SagsConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: Sags)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    packed: bool = False
    """Whether to use packed mode which is more memory efficient but might or might not be as fast. Default is True"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    


class Sags(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SagsConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        self.gaussian_ids = None # for evaluation
        self.flatten_ids = None # for evaluation
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        load_gs = True
        if load_gs:
            means = torch.nn.Parameter(self.seed_points[0]["means"]) 
            self.xys_grad_norm = None
            self.max_2Dsize = None
            scales = torch.nn.Parameter(self.seed_points[0]["scales"].permute(1, 0))
            quats = torch.nn.Parameter(self.seed_points[0]["quats"].permute(1, 0))
            features_dc = torch.nn.Parameter(self.seed_points[0]["feature_dc"].permute(1, 0))
            features_rest = torch.nn.Parameter(self.seed_points[0]["feature_rest"].permute(1, 0).flip(dims=[1]))
            opacities = torch.nn.Parameter(self.seed_points[0]["opacities"].permute(1, 0))
            features_rest = features_rest.reshape(means.shape[0], num_sh_bases(self.config.sh_degree)-1 , 3)
        else:
            if self.seed_points is not None and not self.config.random_init:
                means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
            else:
                means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
            self.xys_grad_norm = None
            self.max_2Dsize = None
            distances, _ = self.k_nearest_sklearn(means.data, 3)
            distances = torch.from_numpy(distances)
            # find the average of the three nearest neighbors for each point and use that as the scale
            avg_dist = distances.mean(dim=-1, keepdim=True)
            scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
            num_points = means.shape[0] #shape (n, x, y, z)
            quats = torch.nn.Parameter(random_quat_tensor(num_points))
            dim_sh = num_sh_bases(self.config.sh_degree)

            if (
                self.seed_points is not None
                and not self.config.random_init
                # We can have colors without points.
                and self.seed_points[1].shape[0] > 0
            ):
                shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
                if self.config.sh_degree > 0:
                    shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                    shs[:, 1:, 3:] = 0.0
                else:
                    CONSOLE.log("use color only optimization with sigmoid activation")
                    shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
                features_dc = torch.nn.Parameter(shs[:, 0, :])
                features_rest = torch.nn.Parameter(shs[:, 1:, :])
            else:
                features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
                features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

            opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.absgrad[visible_mask].norm(dim=-1) if self.config.packed else self.xys.absgrad[0][visible_mask].norm(dim=-1) # type: ignore
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.Means2D.shape[0], device=self.device, dtype=torch.float32) if self.config.packed else torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.Means2D.shape[0], device=self.device, dtype=torch.float32) if self.config.packed else torch.ones(self.num_points, device=self.device, dtype=torch.float32)
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads
            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                if self.max_2Dsize is not None:
                    toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None
        
        if crop_ids is not None:
            print(f"crop_ids: {crop_ids}, sum: {crop_ids.sum()}")
            if crop_ids.sum() == 0:
                print("Warning: No points within the crop box, returning empty outputs.")
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None
            
        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=self.config.packed,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        # if "gaussian_ids" not in info or info["gaussian_ids"] is None:
        #     raise ValueError("The rasterization function did not produce valid 'gaussian_ids'.")
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"] if self.config.packed else info["radii"][0]  # [N]
        self.gaussian_ids = info["gaussian_ids"]
        self.flatten_ids = info["flatten_ids"]
        self.Means2D = info["means2d"]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def _get_intensity_matrix(self, depth_img:torch.Tensor) -> torch.Tensor:
        """ Convert the depth img in outputs['depth'] to an intensity img to enable loss calculation.
        
        Args:
            depth_img: quried by outputs['depth'] from the returned dict of method 'get_outputs()', a tensor object of shape (H,W).

        Outputs:
            intensity_matrix: a tensor object in shape of (H,W).
        
        """
        depth_img_shape = depth_img.shape
        assert len(depth_img_shape) == 2, "depth img should be in the shape of (H,W)" 

        depth_img_flat = np.asarray(depth_img)
        min_depth = np.min(depth_img_flat)
        max_depth = np.max(depth_img_flat)
        normalized_depth_img_flat = (depth_img_flat - min_depth) / (max_depth - min_depth)

        intensity_img_flat = (normalized_depth_img_flat * 255).astype(np.uint8)
        intensity_img = torch.tensor(intensity_img_flat.reshape((depth_img_shape)))


        return intensity_img
    
    def _get_real_shape(self) -> Dict:
        ''' Establish a dictionary where gaussian_ids are keys to query real shapes in format (label, x/y, x/z) and shape (3,)

        Args: 
            None
        
        Outputs:
            real_shape_dict: a dictionary where gaussian_ids are keys and self.scales[gaussian_ids] are values.
        '''
        assert self.gaussian_ids != None, "Empty gaussian_ids during evaluation."

        real_shape_dict = {}

        for gs_id in self.gaussian_ids:
            gs_key = gs_id.item()
            if 0 <= gs_key < len(self.flatten_ids):
                flatten_id = self.flatten_ids[gs_id]
                if 0 <= flatten_id.item() < len(self.scales):
                    real_shape_dict[gs_key] = self.scales[flatten_id] # is it in format (label, x/y, x/z)? Perhaps need modification here.
                else:
                    real_shape_dict[gs_key] = torch.zeros(3, dtype=torch.float, device='cuda')
            else:
                real_shape_dict[gs_key] = torch.zeros(3, dtype=torch.float, device='cuda')
        return real_shape_dict
    
    def _get_expected_shape(self, mask_with_shape:torch.tensor) -> Dict:
        ''' Establish a dictionary where gaussian_ids are keys to query their expected shapes in format (label, x/y, x/z)
        Args: 
            mask_with_shape: a tensor obj of (H, W, 3).
        Outputs:
            expected_shape_dict: a dictionary where gaussian_ids are keys and their shapes of (label, x/y, x/z) are values.
        '''
        assert self.gaussian_ids is not None, "Empty gaussian_ids during evaluation."
        expected_shape_dict = {}

        for gs_id in self.gaussian_ids:
            gs_key = gs_id.item()
            # Ensure gs_id is within the correct range
            if 0 <= gs_key < len(self.flatten_ids):
                flatten_id = self.flatten_ids[gs_id]
                # Ensure flatten_id is within bounds for self.xys
                if 0 <= flatten_id.item() < len(self.xys):
                    means2d_id = self.xys[flatten_id].long()  # Ensure tensor is of long type for indexing
            
                    # Ensure means2d_id is within bounds for mask_with_shape
                    if 0 <= means2d_id[0].item() < mask_with_shape.shape[0] and 0 <= means2d_id[1].item() < mask_with_shape.shape[1]:
                        # Convert means2d_id to integer indices
                        row_idx = means2d_id[0].item()
                        col_idx = means2d_id[1].item()
                        # Index into mask_with_shape
                        expected_shape_dict[gs_key] = mask_with_shape[row_idx, col_idx]
                    else:
                        expected_shape_dict[gs_key] = torch.zeros(3, dtype=torch.float, device='cuda')
                else:
                    expected_shape_dict[gs_key] = torch.zeros(3, dtype=torch.float, device='cuda')
            else:
                expected_shape_dict[gs_key] = torch.zeros(3, dtype=torch.float, device='cuda')

        return expected_shape_dict

    def _get_mask_with_shape(self, mask:torch.tensor) -> torch.Tensor:
        ''' Create a (H, W, 3) mask where "3" implies label, x/y, x/z accordingly. Used for evaluation.
    
        Args:
            mask: a tensor object in shape of (H, W, 1).
    
        Outputs: 
            mask_with_shape: a tensor object in shape of (H, W, 3).
        '''
        # Initialize the mask_with_shape tensor
        mask_with_shape = torch.zeros((mask.shape[0], mask.shape[1], 3), dtype=torch.float, device=mask.device)
        mask_with_shape[:, :, 0] = mask.squeeze(-1)  # Copy labels in mask to the new mask
    
        # Flatten the H, W indices to create a 1D index tensor
        H, W = mask.shape[:2]
        indices = torch.arange(H * W, device=mask.device).view(H, W)
    
        # Ensure means tensor is properly expanded or truncated to match H * W if necessary
        means_padded = torch.zeros((H * W, 3), device=self.means.device, dtype=self.means.dtype)
    
        if self.means.shape[0] >= H * W:
            means_padded = self.means[:H * W]  
        else:
            means_padded[:self.means.shape[0]] = self.means  
    
        # Extract x, y, z values from means tensor
        x_values = means_padded[:, 0].view(H, W)
        y_values = means_padded[:, 1].view(H, W)
        z_values = means_padded[:, 2].view(H, W)
    
        # Compute x/y and x/z and assign them to the mask_with_shape tensor
        mask_with_shape[:, :, 1] = x_values / (y_values + 1e-8)  
        mask_with_shape[:, :, 2] = x_values / (z_values + 1e-8)

        return mask_with_shape
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        mask = self._downscale_if_required(batch["semantic_masks"])
        mask = mask.to(self.device)

        

        Ll1 = torch.abs(gt_img - pred_img).mean()
        if Ll1.dim() == 0:
            Ll1 = Ll1.unsqueeze(0)
        # calculate the intensity img from depth img, the intensity is used for loss2

        # calculate loss2
        Ll2 = ((gt_img - pred_img)** 2).mean()

        # expected_shape_dict = self._get_expected_shape(self._get_mask_with_shape(mask))
        # real_shape_dict = self._get_real_shape()


        # total_diff = torch.zeros((3,), dtype=torch.float, device=self.device) # shape(3,)
        # if len(self.gaussian_ids) > 0:
        #     for gs_id in self.gaussian_ids:
        #         gs_key = gs_id.item()
        #         one_diff = torch.abs(expected_shape_dict[gs_key] - real_shape_dict[gs_key])
        #         total_diff += one_diff

        #     Ll2 = total_diff / len(self.gaussian_ids)
        # else:
        #     Ll2 = torch.zeros((3,), dtype=torch.float, device=self.device)
        # not used: 
        # depth_img = outputs["depth"]
        # intensity_img = self._get_intensity_matrix(depth_img)
        
        
        # calculate ssim in main loss = (1-lambda)*Ll1 + lambda*simloss
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0, device=self.device).unsqueeze(0)
            
        ##################### Shape Regularization #######################
        perplexity = self.form_perplexity_tensor(self.config.perplexity_path)
        ##################### Shape Regularization #######################
            
        ############################################################################################
        # At this position we will calculate the semantic-wise regularization term. Given Means2D  #
        ############################################################################################
        # try:
        #     shape_loss = self.shape_constrain_loss(perplexity, mask, self.Means2D, scalings=self.scales)
        # except:
        #     print("shape loss out of bound")
        shape_loss = self.shape_constrain_loss(perplexity, mask, self.Means2D, self.scales)
        ############################################################################################
        # At this position we will calculate the semantic-wise regularization term. Given Means2D  #
        ############################################################################################
        
        loss_dict = {
            "loss1": Ll1,
            "loss2": Ll2,
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss + self.config.ssim_lambda * shape_loss,
            "scale_reg": scale_reg,
        }


        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
    
    def form_perplexity_tensor(self, perplexity_path: str) -> torch.Tensor:
    
    # Open the file
        with open(perplexity_path, mode='r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)

            # Skip the header if it exists
            next(csv_reader)

            # Fetch the second row
            second_row = next(csv_reader)  # This assumes the first call of next() skipped the header
            second_row = [float(item) for item in second_row]

            perplexity = torch.tensor(second_row)
            return perplexity
        
        
    def shape_constrain_loss(self, perplexity: torch.Tensor, semantic_mask: torch.Tensor, Means2D: torch.Tensor, scalings: torch.Tensor):
        '''
        Calculate a shape constraint loss based on the description provided.
        
        Arguments:
        - perplexity: Tensor containing perplexity values.
        - semantic_mask: Tensor of semantic masks.
        - Means2D: Tensor containing means, assumed to be of shape [n, 2] where each row is [x, y].
        - scalings: Tensor of scalings for Gaussian models.
        - k1, k2: Scaling factors for the loss components.
        '''
        scalings = scalings[(self.radii > 0).flatten()] # Visible Mask
        # Ensure all tensors are on the same device, ideally 'cuda' for GPU acceleration
        k1=3
        k2=1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perplexity = perplexity.to(device)
        semantic_mask = semantic_mask.to(device)
        Means2D = Means2D.squeeze(dim=0)[(self.radii > 0).flatten()].to(device)
        scalings = scalings.to(device)
        # values = torch.tensor([0.0, 1.0, 2.0, -1.0], dtype=torch.float32)
        # shape = (499, 751)

        # # 随机生成包含指定值的浮点张量
        # semantic_mask = values[torch.randint(0, len(values), shape)].to(torch.float32).to(device)

        # Filter valid Gaussians, assuming Means2D is [n, 2] and all are valid if not exactly [0,0] (0, 0) is the initial value
        # Calculate valid_indices considering [0,0] at the center of semantic_mask
        valid_indices = ~(torch.all(Means2D == torch.tensor([0.0, 0.0], device=device), dim=1)) & \
                        (Means2D[:, 0] >= 0) & (Means2D[:, 0] < semantic_mask.shape[0]) & \
                        (Means2D[:, 1] >= 0) & (Means2D[:, 1] < semantic_mask.shape[1])

        valid_means = Means2D[valid_indices]
        # Convert mean coordinates to integer indices, ensuring they are within bounds
        x = valid_means[:, 0].long()
        y = valid_means[:, 1].long()
        # Get masks and filter out -1 (indicating no calculation needed)
        # assert x.min() >= 0 and x.max() < semantic_mask.shape[0], "x indices are out of bounds"
        # assert y.min() >= 0 and y.max() < semantic_mask.shape[1], "y indices are out of bounds"

        mask = semantic_mask[x, y]  # Note the order of indices [x,y]
        valid_mask_indices = mask != -1
        mask = mask[valid_mask_indices].long()

        # Get valid perplexities using mask
        # Filter valid scalings based on the mask
        valid_scalings = scalings[valid_indices]
        valid_scalings = valid_scalings[valid_mask_indices]
        
        a1 = valid_scalings[:, 0] / valid_scalings[:, 1]
        a2 = valid_scalings[:, 0] / valid_scalings[:, 2]
        # Compute the loss for each valid Gaussian
        mask = mask - 1
        p = perplexity[mask]
        
        loss_per_gaussian = torch.sigmoid(k1 / p - a1) + torch.sigmoid(k2 / p - a2)

        # Return the average loss or handle cases where there are no valid Gaussians
        return loss_per_gaussian.mean() if loss_per_gaussian.numel() > 0 else torch.tensor(0.0, device=device)
    # Example usage:
    # Assume you have initialized the tensors `perplexity`, `semantic_mask`, `Means2D`, and `scalings` appropriately.
    # Call shape_constrain_loss(None, perplexity_tensor, semantic_mask_tensor, means2D_tensor, scalings_tensor)