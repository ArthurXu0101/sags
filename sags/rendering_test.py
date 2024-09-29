'''
    In the current script we will use data parser and manager to implement a rendering method to obtain the 
    2D covariance, means 2D, and rendering order sequence for each Gaussian Splatting

    According to Gaussian Splatting, we maintained a weighted sum, and a weighted sum of ground truth for each Gaussian
    Instead of use Gaussian Splatting for full pipeline, we only need to do the projection
'''

from nerfstudio.cameras.cameras import Cameras
from semanticField.data.SemanticFieldDataManager import SemanticDataManagerConfig, SemanticDatamanager
from semanticField.data.SemanticFieldDataParser import SemanticDataParserConfig
from semanticField.data.SemanticFieldDataset import SemanticFieldDataset
from semanticField.model.semanticField import SemanticFieldConfig, SemanticField, get_viewmat
from semanticField.rasterization import rasterization
from nerfstudio.utils.rich_utils import CONSOLE, get_progress
import torch
from dataclasses import dataclass, field
from typing import Type, Dict, Union, List, Literal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
'''
    Here is the thing we need to obtained. Which Gaussian is projected to which pixel. 
    And we also need to know the Z-buffer. 
    To be specific we have these two classes
    gaussian_optimizer
    rasterized_result
'''
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@dataclass

class gaussian_optimizer:
    def __init__(self, gaussian_len:int, feature_len: int, optimize_function) -> None:
        
        self.gaussian_len = gaussian_len
        self.feature_len = feature_len
        
        # Optimize result is target_feature_sum / weighted_sum
        self.optimize_function = optimize_function
        if optimize_function == 'polynomial':
            self.type = torch.float64
        else:
            self.type = torch.float32
        self.target_feature_sum = torch.zeros((self.gaussian_len, self.feature_len), dtype=self.type)
        self.weight_sum = torch.zeros(self.gaussian_len, dtype=self.type)

        
        # we need to weight_sum and feature_sum every iteration through each iteration


    def __len__(self)-> int:
        return self.gaussian_len


    def to(self, device):
        self.target_feature_sum = self.target_feature_sum.to(device=device)
        self.weight_sum = self.weight_sum.to(device=device)


    def _polynomial_optimal(self, weight, pixel_index, gaussian_index, gt_feature) -> None:
        # Define batch size
        batch_size = 1024**2

        # Number of batches
        num_batches = (weight.shape[0] + batch_size - 1) // batch_size


        for batch in range(num_batches):
             # Determine the start and end of the batch
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, weight.shape[0])

            # Process the batch
            batch_weight = weight[start_idx:end_idx]
            batch_pixel_index = pixel_index[start_idx:end_idx]
            batch_gaussian_index = gaussian_index[start_idx:end_idx]

            
            # Extract features for the current batch
            batch_features = gt_feature[batch_pixel_index[:, 0], batch_pixel_index[:, 1]]
            batch_weight = batch_weight.unsqueeze(1)

            batch_weighted_features = batch_features * batch_weight
            
            # Accumulate the results
            ### double addition decrease the speed by half

            self.weight_sum.index_add_(0, batch_gaussian_index, batch_weight.squeeze().to(torch.float64))
            self.target_feature_sum.index_add_(0, batch_gaussian_index, batch_weighted_features.to(torch.float64))


    def _best_fit(self, weight, pixel_index, gaussian_index, gt_feature):
        '''
            After create 
        '''
        # Define batch size
        batch_size = 1024**2

        # Number of batches
        num_batches = (weight.shape[0] + batch_size - 1) // batch_size

        buffer_feature = torch.zeros_like(self.target_feature_sum)
        buffer_weight = torch.zeros_like(self.weight_sum)

        for batch in range(num_batches):
            # Determine the start and end of the batch
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, weight.shape[0])

            # Process the batch
            batch_weight = weight[start_idx:end_idx]
            batch_pixel_index = pixel_index[start_idx:end_idx]
            batch_gaussian_index = gaussian_index[start_idx:end_idx]

            # Extract features for the current batch
            batch_features = gt_feature[batch_pixel_index[:, 0], batch_pixel_index[:, 1]]
            batch_weight = batch_weight.unsqueeze(1)
            batch_weighted_features = batch_features * batch_weight



            buffer_weight.index_add_(0, batch_gaussian_index, batch_weight.squeeze())
            buffer_feature.index_add_(0, batch_gaussian_index, batch_weighted_features)

            
        should_update = buffer_weight > self.weight_sum
        # Apply the condition to update self.weight_sum and self.target_feature_sum

        self.weight_sum = torch.where(should_update, buffer_weight, self.weight_sum)
        self.target_feature_sum = torch.where(should_update.unsqueeze(1), buffer_feature, self.target_feature_sum)



    def iteration_update(self, rasterize_output:Dict, gt_feature: torch.Tensor) -> None:
        '''
            This will update target sum and weight 

            Where the sum will be update according to the rasterize_output in the following way:
            Iterate through rasterize_output. Since it is ranked, we need to process it in linear
            time. 

            Args: 
                rasterize_output: it store a sorted result for blending and weight calculation
            Output:
                It will update self.target_feature_sum and self.weight_sum.
        '''
        weight:torch.Tensor = rasterize_output['Weight']
        pixel_index:torch.Tensor = rasterize_output['Pxiel IDs']
        gaussian_index:torch.Tensor = rasterize_output['Gaussian IDs']


        if self.optimize_function == 'polynomial':
            return self._polynomial_optimal(weight=weight, pixel_index=pixel_index, gaussian_index=gaussian_index, gt_feature=gt_feature)
        elif self.optimize_function == 'best_fit':
            return self._best_fit(weight=weight, pixel_index=pixel_index,gaussian_index=gaussian_index,gt_feature=gt_feature)
        

    def optimize_feature(self)->torch.Tensor:
        '''
            directly return the optimized result
        '''
        self.weight_sum = self.weight_sum+1e-5
        double_presision_result = self.target_feature_sum/self.weight_sum.unsqueeze(1)
        assert torch.isfinite(double_presision_result).all(), f"there are nan for double precision result"

        single_result = double_presision_result.to(torch.float32)
        assert torch.isfinite(single_result).all(), f"there are nan for single precision result"

        return single_result


class SemanticFieldAnalyticalConfig(SemanticFieldConfig):
    """We fix alpha, xyz, and scale, we only train for features"""
    _target: Type = field(default_factory=lambda: SemanticFieldAnalytical)


class SemanticFieldAnalytical(SemanticField):
    def __init__(self, *args, seed_points, **kwargs):
        super().__init__(*args, seed_points=seed_points, **kwargs)
    
    def get_outputs_simple(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
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
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            scales_crop = self.scales
            quats_crop = self.quats
        
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        viewmat = get_viewmat(optimized_camera_to_world).cuda()

        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        render_mode = "RGB"
        # no need to use color and features 
        # psudo rasterization, try to implemente this

        gaussian_ids, pixelIds, weight = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            rasterize_mode=self.config.rasterize_mode,
        )

        return {
                "Gaussian IDs":gaussian_ids, 
                "Pxiel IDs": pixelIds, 
                "Weight": weight
                }


class Trainer:
    def __init__(self, dataManagerConfig: SemanticDataManagerConfig, modelConfig: SemanticFieldConfig, device = 'cuda', optimal_function='polynomial') -> None:
        "set up the data manager, and take out the Gaussian model"
        self.dataManager:SemanticDatamanager = dataManagerConfig.setup()
        pts = self.dataManager.train_dataparser_outputs.metadata["points3D_xyz"]
        pts_rgb = self.dataManager.train_dataparser_outputs.metadata["points3D_rgb"]
        seed_pts = (pts, pts_rgb)
        self.dataManager.to(device)
        self.model = SemanticFieldAnalytical(
            config = modelConfig,
            scene_box=self.dataManager.train_dataset.scene_box,
            num_train_data=len(self.dataManager.train_dataset),
            metadata=self.dataManager.train_dataset.metadata,
            device=device,
            seed_points=seed_pts,
        )
        self.model.to(device)
        self.model.training = False

        self.optimizer = gaussian_optimizer(
            self.model.num_points,
            feature_len=768,
            optimize_function=optimal_function
        )

        self.optimizer.to(device=device)


    def assert_feature_stable(self, gt_feature:torch.Tensor, text_embedding:torch.Tensor, step, name):
        
        H, W, C = gt_feature.shape
        B,C = text_embedding.shape


        # Reshape HWC_result to match the shape (H * W, C) for matrix multiplication
        HWC_flattened = gt_feature.view(-1, C)  # (H*W, C)

        # Perform matrix multiplication to get the attention map (b, H*W)
        attention_map = torch.matmul(text_embedding.to(torch.float32), HWC_flattened.t())  # (b, H*W)

        # Reshape attention map to (b, H, W)
        attention_map = attention_map.view(B, H, W)

        # Normalize the attention map to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Convert to CPU and detach from the graph for visualization
        attention_map = attention_map.cpu().detach().numpy()

        # Generate heatmap and save as PNG
        for i in range(B):  # Loop through each batch element
            plt.imshow(attention_map[i], cmap='viridis')  # Change cmap if needed
            plt.colorbar()
            plt.title(f'Attention Map for Batch {i}')
            plt.savefig(f'/home/butian/workspace/outputs/debug/{name}_{step}_{i}.png')
            plt.close()

    def train_iteration(self, step:int, text_embedding):
        '''
            For this part, we need to use the rasterization part of Gaussian
        '''

        camera_position, batch = self.dataManager.next_train(step)
        gt_feature = batch['features_image']
                

        outputs = self.model.get_outputs_simple(camera_position)
        # we need to update gaussian ids expected result according to weight, and pixel IDs

        self.optimizer.iteration_update(rasterize_output=outputs, gt_feature=gt_feature)
        #self.model.gauss_params["features"] = features

        #self.assert_feature_stable(gt_feature=gt_feature, text_embedding = text_embedding, step=step, name='gt')
        #output = self.model.get_outputs(camera=camera_position)
        #self.assert_feature_stable(gt_feature=output['features'], text_embedding = text_embedding, step=step, name='predict')

        del batch["features_image"]
        del batch["colors_image"]


    def train(self, text_embedding):

        CONSOLE.print(f"There are {len(self.dataManager.train_unseen_cameras)} training camera left for update")
        progress = get_progress(f"[cyan] Iterating Unseen Cameras, optimize with {self.optimizer.optimize_function}:", suffix="iters/sec")
        train_len = len(self.dataManager.train_unseen_cameras)

        self.dataManager.cached_train[0]
        
        with progress as iterate_progress:
            task = iterate_progress.add_task("training", total=train_len)

            # Update the progress bar
            for step in range(train_len):
                self.train_iteration(step=step, text_embedding = text_embedding)
                progress.update(task, advance=1)  # Advance the task by 1 step
        
        return self.optimizer.optimize_feature()


if __name__ == '__main__':
    datamanager=SemanticDataManagerConfig(
        _target=SemanticDatamanager[SemanticFieldDataset],
        dataparser=SemanticDataParserConfig(
            data=Path('/home/butian/workspace/10F'),
            load_pretrained_gs=True,
            max_2D_matches_per_3D_point=0,
        )
    )

    trainer = Trainer(
        dataManagerConfig=datamanager,
        modelConfig=SemanticFieldAnalyticalConfig(),
        optimal_function='polynomial'
    )

    text_embedding = torch.tensor(np.load('/home/butian/workspace/convert_gaussian/ground.npy')).cuda()

    features = trainer.train(text_embedding = text_embedding)
    
    CONSOLE.print(f"Training Accomplished")

    np.savez_compressed('/home/butian/workspace/outputs/feature_splat_visualize/updated_optimizer',features.cpu().numpy())
    
    



