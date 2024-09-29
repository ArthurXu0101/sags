import os 
import torch
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator
import matplotlib.pyplot as plt


def overlay_mask_on_image(
        original_image: np.ndarray, 
        mask_array: np.ndarray, 
        alpha: float = 0.5):
    """
    Overlay a mask as a heatmap onto an original image.

    Parameters:
    - original_image (np.ndarray): The original image array.
    - mask_array (np.ndarray): The final mask array of the original size.
    - alpha (float): Transparency factor of the heatmap.

    Returns:
    - PIL Image: The original image with the heatmap overlay.
    """
    original_size = original_image.shape[:2]  # Height, Width

    # Handle mask values greater than 30
    adjusted_mask = np.mod(mask_array, 30)

    # Create a colormap and normalize mask values
    cmap = plt.get_cmap('viridis', 30)  # Using 30 discrete colors
    heatmap_colors = cmap(adjusted_mask)  # Apply colormap

    # Convert heatmap_colors to an image
    heatmap_image = (heatmap_colors[:, :, :3] * 255).astype(np.uint8)  # Get RGB only
    heatmap_image = Image.fromarray(heatmap_image)

    # Resize the heatmap to match the original image size
    heatmap_image = heatmap_image.resize(original_size[::-1], Image.BILINEAR)  # Resize to (width, height)

    # Convert original image to RGBA
    original_image = Image.fromarray(original_image).convert('RGBA')  # Ensure original is in RGBA mode

    # Blend the original image and the heatmap
    blended_image = Image.blend(original_image, heatmap_image.convert('RGBA'), alpha=alpha)
    
    return blended_image

def generate_integer_mask(masks:np.ndarray):
    # Create an output mask with the same shape as the original image
    output_mask = np.zeros_like(masks[0]['segmentation'], dtype=np.int32)  # Assuming masks are binary
    
    binary_masks = np.zeros((len(masks), masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]))
    for idx, mask in enumerate(masks):
        output_mask[mask['segmentation']] = idx + 1  # Assign unique integer values starting from 1
        binary_masks[idx] = mask['segmentation']
    return output_mask, binary_masks

def process_images_on_gpu(
    gpu_id: int, 
    model: SamAutomaticMaskGenerator, 
    loader: torch.utils.data.Dataset, 
    output_dir:str, 
    debugging:bool
    ):
    device = f"cuda:{gpu_id}"
    model.to(device)
    
    with torch.no_grad():
        for images, names in loader:
            images = images.to(device)
            masks = model.generate(images)
            masks, binary_masks = generate_integer_mask(masks=masks)
            if debugging:
                result_image = overlay_mask_on_image(images.cpu().numpy(), masks)
                result_image.save(os.path.join(output_dir, names[0]))  # Ensure the output path is valid
            else:
                np.savez_compressed(os.path.join(output_dir, names[0].split('.')[0] + '.npz'), masks)
