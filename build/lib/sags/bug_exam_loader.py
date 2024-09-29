
from nerfstudio.cameras.cameras import Cameras
from semanticField.data.SemanticFieldDataManager import SemanticDataManagerConfig, SemanticDatamanager
from semanticField.data.SemanticFieldDataParser import SemanticDataParserConfig
from semanticField.data.SemanticFieldDataset import SemanticFieldDataset
from semanticField.model.semanticField import SemanticFieldConfig, SemanticField, get_viewmat
from semanticField.rasterization import rasterization
from nerfstudio.utils.rich_utils import CONSOLE, get_progress
import torch
from dataclasses import dataclass, field
from typing import Type, Dict, Union, List, Literal, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


'''
    Ground Truth Does not contain any visuable question
'''


def query(text:List[str], 
          text_token: torch.Tensor = None, 
          attention_threshold: float = 0.5, 
          centre: bool = False
          ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            text (str): one list strs to describe the points one wants to query. Its length is B
            text_token (torch.Tensor): In the shape of (B,C) a text embedding one wants to query. If it is none, we will generate text token from text
            attention_threshold (float): We will return all the point cloud if the attention score is larger then 0.5
            centre(bool): If it is set to True, we will return multiple points, otherwise, we return the mean of all points 

        Returns:
            location List[(torch.Tensor)]: xyz location in my coordinates its length is B. Each tensor is in the shape of (N,3)
            attention_score List[(torch.Tensor)]: the corresponding attention score of each points length is B. Each element is in the shape of (N,)
      """
        

class Trainer:
    def __init__(self, dataManagerConfig: SemanticDataManagerConfig, device = 'cuda', ) -> None:
        "set up the data manager, and take out the Gaussian model"
        self.dataManager:SemanticDatamanager = dataManagerConfig.setup()
        self.dataManager.to(device)


    def assert_feature_stable(self, gt_feature:torch.Tensor, text_embedding:torch.Tensor, step):
        
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
            plt.savefig(f'{step}_{i}.png')
            plt.close()


    def train_iteration(self, step:int, text_embedding):
        '''
            For this part, we need to use the rasterization part of Gaussian
        '''

        camera_position, batch = self.dataManager.next_train(step)
        gt_feature = batch['features_image']
        self.assert_feature_stable(gt_feature=gt_feature, text_embedding = text_embedding, step=step)
        del batch["features_image"]
        del batch["colors_image"]

    def train(self, text_embedding):

        CONSOLE.print(f"There are {len(self.dataManager.train_unseen_cameras)} training camera left for update")
        progress = get_progress(f"[cyan] Iterating :", suffix="iters/sec")
        train_len = len(self.dataManager.train_unseen_cameras)

        self.dataManager.cached_train[0]
        
        with progress as iterate_progress:
            task = iterate_progress.add_task("training", total=train_len)

            # Update the progress bar
            for step in range(train_len):
                self.train_iteration(step=step, text_embedding=text_embedding)
                progress.update(task, advance=1)  # Advance the task by 1 step
        
        return 

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
    )

    text_embedding = torch.tensor(np.load('ground.npy')).cuda()

    features = trainer.train(text_embedding)
    
    CONSOLE.print(f"Training Accomplished")

    

