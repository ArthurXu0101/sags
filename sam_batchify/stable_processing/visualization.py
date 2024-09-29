from PIL import Image
import torch
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

def text_image_attention_display(original_image: Image.Image, attention_map: torch.Tensor) -> Tuple[Image.Image]:
    '''
    Original image is a PIL Image with the shape of (H, W, 3).
    cross_attention_map is a torch tensor with the shape of (B, H, W), elements range from -1 to 1.
    
    Returns a tuple of images displaying the overlay heatmap of cross_attention_map on the original image in batch.
    '''
    # Convert original image to numpy array
    img_array = np.array(original_image)

    # Initialize list to hold the resulting images
    result_images = []

    # Normalize the attention map to range 0 to 1
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-5)  # Added small value to avoid division by zero
    # Apply a threshold to the attention map
    threshold_value = 0.4  # You can adjust this threshold value
    attention_map = (attention_map > threshold_value) * attention_map
    
    # Convert the normalized attention map to a heatmap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(attention_map.cpu().numpy())  # This produces an RGBA image
    # Convert RGBA heatmap to RGB by ignoring alpha channel
    heatmap = (heatmap[..., :3] * 255).astype(np.uint8)
    
    # Blend the heatmap with the original image
    blended_image = 0.5 * img_array + 0.5 * heatmap
    blended_image = blended_image.astype(np.uint8)
    # Convert blended image to PIL Image and add to results
    result_images.append(Image.fromarray(blended_image))

    return tuple(result_images)