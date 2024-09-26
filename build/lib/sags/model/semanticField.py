"""
Semantic Feature Lifted into Gaussian Splatting
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch

try:
    from feature_splat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from pytorch_msssim import SSIM
from torch.nn import Parameter, CosineEmbeddingLoss

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from pathlib import Path
'''
    Question: how to do visualization? 
'''




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
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


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
class SemanticFieldConfig(ModelConfig):
    """We fix alpha, xyz, and scale, we only train for features"""
    _target: Type = field(default_factory=lambda: SemanticField)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black" 
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    feature_len = 768  # not fix
    """This is the clip feature dimension"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    pretrained_gaussian_location: Path = Path()
    """Pretrained Gaussian Splatting Position"""   
    sh_degree: int = 0
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    ssim_lambda: float = 0.2
    """Used for loss control"""
    feature_color_ratio: float = 20

    
    # It usually set as False, so that we do not get depth,we can also set it to 
    # True to get some depth
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Where they get this antialiased method? 
    
    
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""


class SemanticField(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SemanticFieldConfig

    def __init__(
        self,
        *args,
        seed_points, # This is a tuple, first one is a dictionary, second one is nothing
        **kwargs,
    ):
        # directly load Gaussians 
        # xyz, quaternion, scale, alpha and the content 

        self.initialized_gaussians:dict = seed_points[0]
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        assert self.initialized_gaussians != None, "In semantic Field, we assume one must give a pretrained Gaussian model"
        ### Semantic Field Gaussian 
        self.xys_grad_norm = None # I don't think we beed this, let's see
        self.max_2Dsize = None
        # find the average of the three nearest neighbors for each point and use that as the scale
        self.means = torch.tensor(self.initialized_gaussians["means"]) # This parameter should not be optimized
        self.quats = torch.tensor(self.initialized_gaussians["quats"]).T
        self.scales = torch.tensor(self.initialized_gaussians["scales"]).T
        self.opacities = torch.tensor(self.initialized_gaussians["opacities"]).squeeze()
        
        # Then we need to initilize the features we need
        # for each Gaussian, I have a parameters
        features = torch.nn.Parameter(torch.rand((len(self.means), self.config.feature_len)))
        colors = torch.nn.Parameter(torch.rand((len(self.means), 3)))

        self.gauss_params = torch.nn.ParameterDict(
            {
                "features": features, # This is a (points_number, feature_dimension)
                "colors": colors # This is a (points number, 3)
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        # shall we use psnr? 

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.feature_loss = CosineEmbeddingLoss()
        
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)


    def to(self, device):

        super(SemanticField, self).to(device)

        self.means = self.means.to(device)
        self.quats = self.quats.to(device)
        self.scales = self.scales.to(device)
        self.opacities = self.opacities.to(device)


    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    # This is differentialble btw
    def features(self):
        return self.gauss_params["features"]

    @property
    def colors(self):
        return self.gauss_params["colors"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in [ "features"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)
        
    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            # It looks like its a momentum thing
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # type: ignore
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)
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

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["features", "colors"]
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
    
    def _get_project_output(self, cameras: Cameras) -> Tuple[torch.Tensor]:
        '''
            Take in camera and return the following:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [C, N]
        '''

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
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features = self.features[crop_ids]
            colors = self.colors[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features = self.features
            colors = self.colors
            scales_crop = self.scales
            quats_crop = self.quats

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world).cuda()
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

        '''
            We use both semantics feature and color to train
            Notice that color is not real RGB color, it is only used 
            for viser to do the visualization.

            We might need to design different loss here, and we also need to filter
            Out Gaussians !!! Too memory consuming

            Wht not normalize features? 
        '''
        color_and_features = torch.cat([features, colors], dim=1) # (B, C+3) B is the number of points
        # C is the feature channel, 3 is the colors channel. Color is range from 0 to 1
    
        ######################################################################
        # Why we need to use sigmoid? Why not just ignore activation function? 
        color_and_features_activatet = torch.sigmoid(color_and_features)

        # Why we need to use sigmoid? Why not just ignore activation function?
        ######################################################################
        sh_degree_to_use = None

        
        ### We need to modfiy the backend in gsplat to render features
        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=color_and_features_activatet, # color is N,D (number of points, 512)
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        # No idea, what the fuck is this
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2] 
        # No idea, what the fuck is this
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        # The rendered result should be something like (H,W, C+3)
        feature_rendered = render[:, ..., :self.config.feature_len]

        background = self._get_background_color()
        rgb = render[:, ..., self.config.feature_len:self.config.feature_len+3] + (1 - alpha) * background 
        # This means the 2D plane has an overall alpha, and we need to overlay it with our bg
        rgb = torch.clamp(rgb, 0.0, 1.0)

        
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., self.config.feature_len+3:self.config.feature_len+4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)
        return {
            "features": feature_rendered.squeeze(0),
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
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["colors_image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_features = self.get_gt_img(batch["features_image"])
        gt_colors = self.get_gt_img(batch["colors_image"])

        predicted_features = outputs["features"]
        prediced_colors = outputs["rgb"]

        cosine_loss = self.feature_loss(gt_features.view(-1, gt_features.shape[-1]), 
                                        predicted_features.view(-1, gt_features.shape[-1]), 
                                        torch.ones(gt_features.view(-1, gt_features.shape[-1]).shape[0]).to(self.device)).mean()

        color_Ll1 = torch.abs(gt_colors - prediced_colors).mean()
        color_simloss = 1 - self.ssim(gt_colors.permute(2, 0, 1)[None, ...], prediced_colors.permute(2, 0, 1)[None, ...])
        color_loss = (1 - self.config.ssim_lambda) * color_Ll1 + self.config.ssim_lambda * color_simloss

        loss_dict = {
            "feature_loss": cosine_loss*self.config.feature_color_ratio,
            "color_loss": color_loss,
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        del batch["features_image"]
        del batch["colors_image"]
        
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
        gt_features = batch["features_image"]
        gt_colors = batch["colors_image"]
        predicted_features = outputs["features"]
        prediced_colors = outputs["rgb"]

        cosine_loss = self.feature_loss(gt_features.view(-1, gt_features.shape[-1]), 
                                        predicted_features.view(-1, gt_features.shape[-1]), 
                                        gt_features.view(-1, gt_features.shape[-1]).shape[0]).mean()

        
        gt_colors = gt_colors.permute(2,0,1).unsqueeze()
        prediced_colors = gt_colors.permute(2,0,1).unsqueeze()
        
        psnr = self.psnr(gt_colors, prediced_colors)
        ssim = self.ssim(gt_colors, prediced_colors)
        

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "consine": float(cosine_loss)}  # type: ignore

        images_dict = {"features": predicted_features, "colors": prediced_colors}

        del batch["features_image"]
        del batch["colors_image"]

        return metrics_dict, images_dict
