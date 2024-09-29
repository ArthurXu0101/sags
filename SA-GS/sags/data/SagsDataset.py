from nerfstudio.data.datasets.base_dataset import InputDataset
from sags.data.SagsDataParser import SagsDataparserOutput
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

from scipy.ndimage import zoom

class SagsDataset(InputDataset):
    def __init__(self, dataparser_outputs: SagsDataparserOutput, scale_factor: float = 1):
        super().__init__(dataparser_outputs, scale_factor)
    
    @property
    def mask_filenames(self) -> List[Path]:
        return self._dataparser_outputs.mask_filenames

    @property
    def feature_filenames(self) -> List[Path]:
        return self._dataparser_outputs.feature_filenames
    
    @property
    def color_filenames(self) -> List[Path]:
        return self._dataparser_outputs.color_filenames
    
    def get_data(self, image_idx: int) -> Dict:
        """Returns the FeatureDataset data as a dictionary.
            The returned data should include the followings:
            - Feature Synthesized by Features and masks
            - Corresponding colors for visualization

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
            
            
        """
        masks, features, colors =self.get_numpy_features(image_idx=image_idx)
        data = {"feature_idx": torch.tensor(image_idx), 
                "masks":torch.tensor(masks), 
                "features": torch.tensor(features), 
                "colors": torch.tensor(colors)}
        metadata = self.get_metadata(data)
        
        data.update(metadata)
        return data
    
    def get_numpy_features(self, image_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the image of shape (H, W, C). And also a color called (H,W,3)
            
            The question is that we need to use Masks and a features        

        Args:
            image_idx: The image index in the dataset.
        """
        mask_filename = self.mask_filenames[image_idx]
        feature_filename = self.feature_filenames[image_idx]
        color_filename = self.color_filenames[image_idx]
        
        mask:np.ndarray = np.load(mask_filename)['arr_0']
        feature:np.ndarray = np.load(feature_filename)['arr_0']
        color:np.ndarray = np.load(color_filename)['arr_0']
        
        
        assert mask.shape[0] == color.shape[0], f"mask shape and color shape is not the same! get mask shape: {mask.shape} and color shape{color.shape}"
        assert mask.shape[0] == feature.shape[0], f"batch size of feature and mask are not the same. Mask shape {mask.shape}, and feature shape{feature.shape}"

        if self.scale_factor != 1.0:
            mask = zoom(mask, (1, self.scale_factor, self.scale_factor), order=0)
            color = zoom(color, (1, self.scale_factor, self.scale_factor), order=0)
        return mask, feature, color  
