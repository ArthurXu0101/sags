import os
import torch
import argparse
import open_clip
import numpy as np
import gradio as gr
from PIL import Image
from typing import Tuple
from open_clip import CLIP
from open_clip.tokenizer import SimpleTokenizer
from visualization import text_image_attention_display, mask_display

"display original image, masks, and attention map overlay"



# Dummy function that processes an image based on the selected sequence number
def loading_data(image_folder: str, image_name: str, npz_folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        image_folder (str): Image folder location
        image_name (str): Image name
        npz_folder (str): npz location, where in it, we store masks, and we store embeddings. The masks is not binary masks

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        original_images: (H,W,3)original image in rgb sequence, cv2
        masks: (H,W), number sequence represents which semantic group it belongs to
        embeddings: (B, C), each clip embeddings for representing that mask         
    """
    # For now, return a blank image or any placeholder
    original_image_location = os.path.join(image_folder, image_name)
    npz_location = os.path.join(npz_folder, image_name.split('.')[0]+'.npz')
    original_image = Image.open(original_image_location)  
    npz_file = np.load(npz_location, allow_pickle=True)
    
    masks = npz_file["masks"]
    embeddings = npz_file["embedding"]
    
    return original_image, masks, embeddings


def extend_features(features: torch.Tensor) -> torch.Tensor:
    '''
    features: Tensor of shape (B, C).
    Returns a new tensor of shape (B+1, C) with the first row as zeros.
    '''
    B, C = features.shape

    # Create a zero tensor of shape (1, C) on the same device and with the same dtype as features
    zeros_row = torch.zeros(1, C, device=features.device, dtype=features.dtype).cuda()

    # Concatenate the zero row to the beginning of features
    extended_features = torch.cat([zeros_row, features], dim=0)  # Shape: (B+1, C)

    return extended_features

    
def feature_map_generation(masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    features = extend_features(features)
    '''
    masks: Tensor of shape (H, W), containing integers from 0 to B-1.
    features: Tensor of shape (B, C).
    Returns a feature map of shape (H, W, C), where each vector is a unit vector.
    '''
    # Ensure masks are long tensors for indexing
    masks = masks.long()
    
    # Ensure masks and features are on the same device
    masks = masks.to(features.device)
    
    # Index into features using masks to create the feature map
    feature_map = features[masks]  # Shape: (H, W, C)
    
    # Normalize the feature vectors to unit length
    feature_map = torch.nn.functional.normalize(feature_map, dim=-1)
    
    return feature_map


def process_image(text_embeddings: torch.Tensor, masks: np.ndarray, image_embeddings:torch.Tensor, original_image:np.ndarray) -> Tuple[Image.Image, Image.Image]:
    """_summary_
    Args:
        text_embeddings (torch.Tensor): (C,) clip embedding for representing what you queried
        masks (np.ndarray): (H,W), number sequence represents which semantic group it belongs to
        image_embeddings (torch.Tensor): (B, C), each clip embeddings for representing that mask         
        original_image (np.ndarray): (H,W,3)original image in rgb sequence, cv2

    Returns:
        Tuple[Image.Image, Image.Image]: 
        Masks image: each mask use one color, if no color, then that means no masks
        attention_map: the attention map according our given masks
    """
    masks_tensor = torch.tensor(masks).cuda()
    feature_map = feature_map_generation(masks=masks_tensor, features=image_embeddings)
    
    C = text_embeddings.shape[1]
    H, W, _ = feature_map.shape
    
    feature_map =feature_map.reshape(-1, C)
    # remember, dot product
    with torch.cuda.amp.autocast():
        attention_map = (100.0 * text_embeddings @ feature_map.T).softmax(dim=-1) 
    attention_map = attention_map.reshape(H,W).to(torch.float32)
    
    mask_image: Image.Image = mask_display(masks=masks)
    overlay_attention_map: Image.Image = text_image_attention_display(original_image=original_image, attention_map=attention_map)
    
    return mask_image, overlay_attention_map
    
    
    

def text_embedding_extraction(model: CLIP, tokenizer: SimpleTokenizer, text_discription:str) -> torch.Tensor:
    """_summary_

    Args:
        model (CLIP): CLIP model, we will use CLIP encode text to extract text embeddings
        tokenizer (SimpleTokenizer): we use simple tokenizer to get tokenize text
        text_discription (str): The text desctiption one wants to encode

    Returns:
        torch.Tensor: the text embeddings in theshape of feature channels
    """
    text_tokenizer = tokenizer(text_discription)
    text_tokenizer = text_tokenizer.cuda()
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokenizer)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

def parser():
    parser = argparse.ArgumentParser("SEG-CLIP Generate Semantic Specific Attention Map", add_help=True)
    parser.add_argument("--clip_version", type=str, default="ViT-B-16", required=True, help="the version of CLIP version, in debugging only use ViT-B-16")
    parser.add_argument("--clip_checkpoint", type=str, required=True, help="path to clip checkpoint file")
    parser.add_argument("--image_location", type=str, required=True, help="path to image file")
    parser.add_argument("--npz_location", type=str, required=True, help="path to mask file")
    args = parser.parse_args()
    return args

# Assume parser, text_embedding_extraction, loading_data, process_image are defined elsewhere

# Load models and data once
args = parser()
clip_version = args.clip_version
clip_checkpoint = args.clip_checkpoint
image_folder = args.image_location
npz_location = args.npz_location

# Verify folders have the same number of files
assert len(os.listdir(image_folder)) == len(os.listdir(npz_location)), \
    f"the number of images in the folder should match the number of npz files. Image folder length: {len(os.listdir(image_folder))}, npz file length: {len(os.listdir(npz_location))}"

print("loadig models and tokenizer ...")

selections = os.listdir(image_folder)
model, _, _ = open_clip.create_model_and_transforms(clip_version, pretrained=clip_checkpoint)
tokenizer = open_clip.get_tokenizer(clip_version)
model = model.cuda()

print("loading accomplished")

def process_input(picture_name, text_description):
    text_features = text_embedding_extraction(model=model, tokenizer=tokenizer, text_discription=text_description)
    original_image, masks, embeddings = loading_data(image_folder=image_folder, image_name=picture_name, npz_folder=npz_location)
    mask_image, attention_image = process_image(
        text_embeddings=text_features,
        masks=masks,
        image_embeddings=torch.tensor(embeddings).cuda(),
        original_image=original_image
    )
    return original_image, mask_image, attention_image


with gr.Blocks(title="Image Processor") as demo:
    gr.Markdown("# Image Processor")
    
    # Input Section
    with gr.Row():
        with gr.Column():
            input_dropdown = gr.Dropdown(selections, label="Select Image")
            input_text = gr.Textbox(label="Object Description")
            submit_button = gr.Button("Submit")
        with gr.Column():
            output_original = gr.Image(type="numpy", label="Original Image")
    with gr.Row():
        with gr.Column():
            output_mask = gr.Image(type="pil", label="Mask Image")
        with gr.Column():
            # Bottom Row: Attention Image
            output_attention = gr.Image(type="pil", label="Attention Image")
    
    # Connect inputs and outputs
    submit_button.click(
        fn=process_input,
        inputs=[input_dropdown, input_text],
        outputs=[output_original, output_mask, output_attention]
    )


demo.launch()