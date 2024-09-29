from open_clip import create_model_and_transforms
from open_clip.model import CLIP
from torchvision.transforms import Compose
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
import torch

from torchvision.transforms.functional import InterpolationMode

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def simplified_transform():
    mean = OPENAI_DATASET_MEAN  # Placeholder, replace with actual values
    std = OPENAI_DATASET_STD   # Placeholder, replace with actual values

    return Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(224),
        Normalize(mean=mean, std=std)
    ])

def process_with_clip(image: np.ndarray, masks: np.ndarray, clip_model: CLIP, device: str) -> np.ndarray:
    """
    Args:
        image (np.ndarray): H,W,3 shape, and it should be masked
        mask (np.ndarray): B, H, W shape, binary mask, 1 means the one we are looking for 
        clip_model (CLIP): get the image encoder work

    Returns:
        mask_features (np.ndarray): for each mask,w e have a CLIP representation
    """
    masks = torch.tensor(masks, dtype=torch.float32).to(device=device)
    image_tensors = apply_masks_and_transform(
        image=image,
        masks=masks,
        device=device,
    )
    
        
    # Get features from the masked image
    with torch.no_grad():
        masks_features = clip_model.encode_image(image_tensors)

    return masks_features.cpu().numpy()

def apply_masks_and_transform(image: np.ndarray, masks: torch.Tensor, device, transform: Compose = simplified_transform()) -> torch.Tensor:
    """_summary_

    Args:
        image (np.ndarray): cv2 images
        masks (torch.Tensor): binary masks
        device (_type_): which gpu I am on
        transform (Compose, optional): openAI transforms Defaults to simplified_transform().

    Returns:
        torch.Tensor: _description_
    """
    # Assume image is a PIL Image and masks is a torch Tensor of shape [N, H, W]
    image_tensor = ToTensor()(image).to(device=device)  # Convert image to tensor [C, H, W]
    if image_tensor.shape[2] <= 4: # This is the C 
        image_tensor = image_tensor.permute((2, 0, 1))
    image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
    # Expand masks to match image tensor shape [N, C, H, W]
    masks = masks.unsqueeze(1)  # [N, 1, H, W]
    masks = masks.expand(-1, image_tensor.shape[1], -1, -1)  # Match the color channels
    # Apply masks
    masked_images = image_tensor * masks  # Broadcast multiplication [N, C, H, W]
    
    masked_images = transform(masked_images)
    
    return masked_images
