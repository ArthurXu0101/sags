import os
import cv2


import torch
import numpy as np
from typing import Tuple, List
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from segment_anything import SamAutomaticMaskGenerator,sam_model_registry
from open_clip.model import CLIP
import open_clip

'''
    This is the image loader for sam images. 
    after loading the sam images, we will process it using encoder to extract features in patch
'''

class TransformImage:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int =1024, device = 'cpu') -> None:
        self.target_length = target_length
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(device)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(device)
        
        self.image_size = target_length

    def apply_image(self, image: np.ndarray, device = 'cuda') -> torch.Tensor:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        input_image=  np.array(resize(to_pil_image(image), target_size))
        
        input_image_torch = torch.as_tensor(input_image, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        input_image = self.preprocess(input_image_torch)
        return input_image
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
        
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    

class ImageDataset(Dataset):
    def __init__(self, directory):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): A TransformImage instance or similar for processing images.
            device (string): Device to perform computations on.
        """
        self.directory = directory
        self.images = [os.path.join(directory, img) for img in sorted(os.listdir(directory)) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        basename = os.path.basename(img_path)
        
        return image, basename
    
def load_dataset(directory):
    # Initialize the image transformation class
    
    # Create dataset
    dataset = ImageDataset(directory)
    
    
    return dataset, len(dataset)

    
def load_models(sam_version: str, sam_ckpt: str, clip_version: str, clip_ckpt: str, gpu_number: int = 4) -> Tuple[List[SamAutomaticMaskGenerator], List[CLIP]]:
    """_summary_

    Args:
        sam_version (str): The version of sam
        sam_ckpt (str): The sam checkpoint in os
        clip_version (str): The clip version vit b 16 or b 32 and so on
        clip_ckpt (str): The clip ckpt in os
        gpu_number (int): How many gpu will we use

    Returns:
        Tuple[List[SamAutomaticMaskGenerator], List[CLIP]]: Retrun two list of model, length of model is the number of gpu
    """

    
    sam_models = [SamAutomaticMaskGenerator(sam_model_registry[sam_version](checkpoint=sam_ckpt).to(f'cuda:{i}')) for i in range(gpu_number)]
    sam_models = []
    clip_models = []
    for i in range(gpu_number):
        sam_models.append(SamAutomaticMaskGenerator(sam_model_registry[sam_version](sam_ckpt).to(f'cuda:{i}')))
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name=clip_version,pretrained=clip_ckpt)
        clip_models.append(clip_model.to(f'cuda:{i}'))
    
    return sam_models, clip_models

        