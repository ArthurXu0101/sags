# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Data manager that outputs cameras / images instead of raybundles

Good for things like gaussian splatting which require full cameras instead of the standard ray
paradigm
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import cv2
import fpsample
import numpy as np
import torch
from rich.progress import track
from torch.nn import Parameter
from typing_extensions import assert_never

from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from semanticField.data.SemanticFieldDataParser import SemanticDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import IterableWrapper, get_orig_class

'''
    What we expect is that the input data is a semantic latent
    or simply a semantic masks plus each different latent. 
    
    And then we need to load them into cpu, and we need to accosiate it with 
    camera index. 
    
    The question here is the original image size and mask are not the same. 

    And even if our mask shape and image shape are the same, what does this
    undistor function means for the image? Shall we apply it to our features?
    
    After carefully examine, here is the process, first undistort our images, 
    and then get the features. And this state, we do not use the _image_undistortion
'''


@dataclass
class SemanticDataManagerConfig(FullImageDatamanagerConfig):
    '''
        For this data manager,we only load semantic map without loading images
        The reason why we need to have a image or camera in the data parser is because we need to get the name of images
    '''
    _target: Type = field(default_factory=lambda: SemanticDatamanager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=SemanticDataParserConfig)
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    cache_semantics: Literal["cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device."""
    cache_images_type: Literal["uint8", "float32"] = "float32"
    """The image type returned from manager, caching images in uint8 saves memory"""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images. If None, uses all available threads."""
    train_cameras_sampling_strategy: Literal["random", "fps"] = "random"
    """Specifies which sampling strategy is used to generate train cameras, 'random' means sampling 
    uniformly random without replacement, 'fps' means farthest point sampling which is helpful to reduce the artifacts 
    due to oversampling subsets of cameras that are very close to each other."""
    train_cameras_sampling_seed: int = 42
    """Random seed for sampling train cameras. Fixing seed may help reduce variance of trained models across 
    different runs."""
    fps_reset_every: int = 100
    """The number of iterations before one resets fps sampler repeatly, which is essentially drawing fps_reset_every
    samples from the pool of all training cameras without replacement before a new round of sampling starts."""

class SemanticDatamanager(DataManager, Generic[TDataset]):
    """
    We need to generate a feature according to masks and features
    """
    config: SemanticDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
        self,
        config: FullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        
        # at the begining, the config file has not been instentiate
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        
        self.dataparser = self.dataparser_config.setup() 
        # over here. self.data parser config has been initialized
        
        
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        
        ## This will create trainding_data_parser_output
        ## To be specific, it will call _generate_dataparser_output
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        
        ## Notice that we do not need to cache any images, what we want to cache is the
        ## Is the semantic representation
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"
        
        ## I have no idea, wtf is this
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = self.sample_train_cameras()
        ## This should only return a list of index
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"

        super().__init__()

    def sample_train_cameras(self):
        """Return a list of camera indices sampled using the strategy specified by
        self.config.train_cameras_sampling_strategy"""
        num_train_cameras = len(self.train_dataset)
        if self.config.train_cameras_sampling_strategy == "random":
            if not hasattr(self, "random_generator"):
                self.random_generator = random.Random(self.config.train_cameras_sampling_seed)
            indices = list(range(num_train_cameras))
            self.random_generator.shuffle(indices)
            return indices
        elif self.config.train_cameras_sampling_strategy == "fps":
            if not hasattr(self, "train_unsampled_epoch_count"):
                np.random.seed(self.config.train_cameras_sampling_seed)  # fix random seed of fpsample
                self.train_unsampled_epoch_count = np.zeros(num_train_cameras)
            camera_origins = self.train_dataset.cameras.camera_to_worlds[..., 3].numpy()
            # We concatenate camera origins with weighted train_unsampled_epoch_count because we want to
            # increase the chance to sample camera that hasn't been sampled in consecutive epochs previously.
            # We assume the camera origins are also rescaled, so the weight 0.1 is relative to the scale of scene
            data = np.concatenate(
                (camera_origins, 0.1 * np.expand_dims(self.train_unsampled_epoch_count, axis=-1)), axis=-1
            )
            n = self.config.fps_reset_every
            if num_train_cameras < n:
                CONSOLE.log(
                    f"num_train_cameras={num_train_cameras} is smaller than fps_reset_ever={n}, the behavior of "
                    "camera sampler will be very similar to sampling random without replacement (default setting)."
                )
                n = num_train_cameras
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(data, n, h=3)

            self.train_unsampled_epoch_count += 1
            self.train_unsampled_epoch_count[kdline_fps_samples_idx] = 0
            return kdline_fps_samples_idx.tolist()
        else:
            raise ValueError(f"Unknown train camera sampling strategy: {self.config.train_cameras_sampling_strategy}")

    # The cached property is a decorator that can help us to use it only once
    @cached_property
    def cached_train(self) -> List[Dict[str, torch.Tensor]]:
        """Get the training images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        # need to change this function to _load_images
        return self._load_images("train", cache_images_device=self.config.cache_images)

    @cached_property
    def cached_eval(self) -> List[Dict[str, torch.Tensor]]:
        """Get the eval images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        # don't know if we need to eval
        return self._load_images("eval", cache_images_device=self.config.cache_images)


    #### we need to load features according to the name of images
    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        """We requrie the image is undistorted"""
        undistorted_features: List[Dict[str, torch.Tensor]] = []
        # Initialize the image we want to use

        # Which dataset?
        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)


        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_sags_data(idx) # retured a dictionary of data
            # this inlude masks, features, colors, and so on
            camera = dataset.cameras[idx].reshape(())

            # masks in the shape of B,H,W
            assert data["masks"].shape[2] == camera.width.item() and data["masks"].shape[1] == camera.height.item(), (
                f'The size of image ({data["masks"].shape[2]}, {data["masks"].shape[1]}) loaded '
                f'does not match the camera parameters ({camera.width.item(), camera.height.item()})'
            )

            # Check if the distortion is effectively zero
            distortion_threshold = 1e-6  # Define a small threshold
            assert torch.all(torch.abs(camera.distortion_params) <= distortion_threshold), \
                "Detected significant distortion parameters, image undistortion may be required."

            # If distortion is negligible, return the data without undistortion
            return data

        # Load images with potential multi-threading
        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )


        if cache_images_device == "gpu":
            """features are too large, really not recommended to put it into gpu"""
            raise NotImplementedError
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["masks"] = cache["masks"].pin_memory()
                cache["features"] = cache["features"].pin_memory()
                cache["colors"] = cache["colors"].pin_memory()
                self.train_cameras = self.train_dataset.cameras
        else: 
            assert_never(cache_images_device)

        return undistorted_images

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[SemanticDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is SemanticDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is SemanticDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is SemanticDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default


    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        """Sets up the data loaders for training"""

    def setup_eval(self):
        """Sets up the data loader for evaluation"""

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval)
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            cameras.append(_cameras[i : i + 1])
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def get_train_rays_per_batch(self):
        # TODO: fix this to be the resolution of the last image rendered
        return 800 * 800


    def _get_feature_map(self, mask_array:torch.Tensor, feature_array: torch.Tensor):
        # Convert inputs to torch tensors if they aren't already
        mask_tensor = mask_array.to('cuda')
        feature_tensor = feature_array.to('cuda')

        _, c = feature_tensor.shape 
        _, h, w = mask_tensor.shape

        # Extend the mask array
        mask_extended = mask_tensor.unsqueeze(-1)  # Now shape is (B, H, W, 1)
        mask_extended = mask_extended.expand(-1, -1, -1, c)  # Now shape is (B, H, W, c)

        # Extend the feature array
        feature_extended = feature_tensor.unsqueeze(1).unsqueeze(1)  # Now shape is (B, 1, 1, c)
        feature_extended = feature_extended.expand(-1, h, w, -1)  # Now shape is (B, H, W, c)

        # Element-wise multiplication
        weighted_feature = mask_extended * feature_extended  # Shape is (B, H, W, c)
        result = weighted_feature.sum(0)  # Sum over the batch dimension, shape is (H, W, c)

        norms = torch.norm(result, dim=2, keepdim=True)
        norms = norms.clamp(min=1)  # To avoid division by zero

        normalized_result = result / norms  # Normalize

        # Move the tensor back to CPU if necessary for further processing

        return normalized_result

    def _get_color_map(self, mask_array:torch.Tensor, color_array: torch.Tensor):
        # Convert inputs to torch tensors and move to GPU
        mask_tensor = mask_array.to('cuda')
        color_tensor = color_array.to('cuda')

        # Extend mask tensor to (B, H, W, 3)
        mask_extended = mask_tensor.unsqueeze(-1).expand(-1, -1, -1, 3)

        # Extend color tensor to (B, H, W, 3)
        color_extended = color_tensor.unsqueeze(1).unsqueeze(1).expand(-1, mask_array.shape[1], mask_array.shape[2], -1)

        # Element-wise multiplication of mask_extended and color_extended
        weighted_color = mask_extended * color_extended

        # Sum across the first axis (batch dimension)
        result = weighted_color.sum(0)

        # Sum the mask along the batch dimension to compute the mask sum
        mask_sum = mask_tensor.sum(0)

        # Extend mask_sum to (H, W, 3)
        mask_sum_extended = mask_sum.unsqueeze(-1).expand(-1, -1, 3)

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        final_result = result / (mask_sum_extended + epsilon)

        # Optionally move the tensor back to CPU if further non-GPU processing is needed

        return final_result


    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch, Returns a Camera"""
        image_idx = self.train_unseen_cameras.pop(0)
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            # after pop out the last one of the training camera, we need to random sample
            # again
            self.train_unseen_cameras = self.sample_train_cameras()

        
        data = self.cached_train[image_idx]
        
        feature_image = self._get_feature_map(data['masks'], data['features'])
        color_image = self._get_color_map(data['masks'], data['colors'])
        data['features_image'] = feature_image
        data['colors_image'] = color_image

        assert len(self.train_cameras.shape) == 1, "Assumes single batch dimension"
        # Now idea why our batchsize is 1, but, let us leave it here
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        # we might need to exam the meta data here
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        return self.next_eval_image(step=step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        
        feature_image = self._get_feature_map(data['masks'], data['features'])
        color_image = self._get_color_map(data['masks'], data['colors'])
        
        data['features_image'] = feature_image
        data['colors_image'] = color_image
        
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data



if __name__ == "__main__":
    data_parser_config = SemanticDataParserConfig(
        data=Path('/home/butian/workspace/10F')
    )
    from semanticField.data.SemanticFieldDataset import SemanticFieldDataset
    data_manager_config = SemanticDataManagerConfig(
        _target=SemanticDatamanager[SemanticFieldDataset],
        dataparser=data_parser_config
    )

    data_manager:SemanticDatamanager = data_manager_config.setup()
    
    
    