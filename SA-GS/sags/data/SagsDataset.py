from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from sags.data.SagsDataParser import SagsDataparserOutput
from pathlib import Path
from typing import Dict, List, Tuple, Literal
import numpy as np
import torch

from scipy.ndimage import zoom

class SagsDataset(InputDataset):
    def __init__(self, dataparser_outputs: SagsDataparserOutput, scale_factor: float = 1):
        super().__init__(dataparser_outputs, scale_factor)
        self._dataparser_outputs = dataparser_outputs
    
    @property
    def mask_filenames(self) -> List[Path]:
        return self._dataparser_outputs.mask_filenames
    @property
    def semantic_masks_filenames(self) -> List[Path]:
        return self._dataparser_outputs.semantic_masks_filenames
    
    @property
    def feature_filenames(self) -> List[Path]:
        return self._dataparser_outputs.feature_filenames
    
    @property
    def color_filenames(self) -> List[Path]:
        return self._dataparser_outputs.color_filenames
    @property
    def perplexity_path(self) -> Path:
        return self._dataparser_outputs.perplexity_path
    
    
    @property
    def perplexity_path(self) -> Path:
        return self._dataparser_outputs.perplexity_path
    
    def get_sags_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the FeatureDataset data as a dictionary.
            The returned data should include the followings:
            - Features and masks (optional)
            - Corresponding colors for visualization
            - ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
            
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")
        
        data = {"image_idx": image_idx, "image": image}
        
        # Load optional data (masks, features, colors) if paths are available
        semantic_masks, features, colors = None, None, None
        if self._dataparser_outputs.semantic_masks_filenames:
            semantic_masks_filenames = self.semantic_masks_filenames[image_idx]
            semantic_masks = np.load(semantic_masks_filenames)['arr_0']
        if self._dataparser_outputs.feature_filenames:
            feature_filename = self.feature_filenames[image_idx]
            features = np.load(feature_filename)['arr_0']
        if self._dataparser_outputs.color_filenames:
            color_filename = self.color_filenames[image_idx]
            colors = np.load(color_filename)['arr_0']
        if self._dataparser_outputs.perplexity_path:
            perplexity_path = self.perplexity_path
        
        # Optionally scale masks and colors if available
        if self.scale_factor != 1.0:
            if semantic_masks is not None:
                semantic_masks = zoom(semantic_masks, (1, self.scale_factor, self.scale_factor), order=0)
            if colors is not None:
                colors = zoom(colors, (1, self.scale_factor, self.scale_factor), order=0)
                
        # Convert data to torch tensors and add to the data dictionary
        if semantic_masks is not None:
            data["semantic_masks"] = torch.tensor(semantic_masks)
        if features is not None:
            data["features"] = torch.tensor(features)
        if colors is not None:
            data["colors"] = torch.tensor(colors)
        if perplexity_path is not None:
            data["perplexity_path"] = perplexity_path
        
        # Verify shapes for consistency if all data is available
        if semantic_masks is not None and features is not None and colors is not None:
            assert semantic_masks.shape[0] == colors.shape[0], (
                f"semantic_masks and color batch size mismatch! masks shape: {semantic_masks.shape}, colors shape: {colors.shape}"
            )
            assert semantic_masks.shape[0] == features.shape[0], (
                f"Feature and mask batch size mismatch! masks shape: {semantic_masks.shape}, features shape: {features.shape}"
            )
        
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data
