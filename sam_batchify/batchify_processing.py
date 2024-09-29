'''
    We first extract and clustering features in batch
    We then clustering prototype
'''
from segment_anything import SamAutomaticMaskGenerator
import torch.utils
import torch.utils.data
from stable_processing.loader import load_dataset
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

import torch
import argparse

import numpy as np
import threading
import os 
from queue import Queue
from stable_processing.mask_utils import overlay_mask_on_image, generate_integer_mask, process_images_on_gpu
from stable_processing.clip_processing import process_with_clip
from stable_processing.loader import load_models
from open_clip.model import CLIP



def run_inference_on_multiple_gpus(models, loaders, output_dirs, debugging):
    threads = []
    for gpu_id, (model, loader, output_dir) in enumerate(zip(models, loaders, output_dirs)):
        thread = threading.Thread(target=process_images_on_gpu, args=(gpu_id, model, loader, output_dir, debugging))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def process_images(
    queue: Queue, 
    sam_model: SamAutomaticMaskGenerator, 
    clip_model: CLIP,
    output_dir:str, 
    debugging:bool, 
    progress:Progress, 
    task: int):
    while True:
        image, name = queue.get()
        if image is None:  # Check for termination signal
            break

        with torch.no_grad():
            masks = sam_model.generate(image)
            masks, binary_masks = generate_integer_mask(masks=masks)
            masks_feature = process_with_clip(
                image=image,
                masks=binary_masks,
                clip_model= clip_model,
                device=sam_model.predictor.device
            )  
            if debugging:
                result_image = overlay_mask_on_image(image, masks)
                result_image.save(os.path.join(output_dir, name))  # Ensure output path is valid
            else:
                np.savez_compressed(os.path.join(output_dir, name.split('.')[0] + '.npz'), masks = masks, embedding = masks_feature)

        queue.task_done()  # Mark the processed item as done
        
        # Update the progress bar after processing an image
        progress.update(task, advance=1)

 
def parser():
    parser = argparse.ArgumentParser("SAM Encoder Test", add_help=True)
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--clip_checkpoint", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--clip_version", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--debugging", type=str, default="False", help="if to save the internal output to image folder")
    parser.add_argument("--gpu_number", type=int, default=4, help="running on 4 gpu only!, default=False")
    args = parser.parse_args()
    return args
   
def main():
    args = parser()
    sam_checkpoint = args.sam_checkpoint
    sam_version = args.sam_version
    clip_ckpt = args.clip_checkpoint
    clip_version = args.clip_version
    gpu_number = args.gpu_number
    
    print(f'Sam Checkpoint is :{sam_checkpoint}')
    print(f'Sam version is :{sam_version}')
    
    image_directory = args.image_dir
    output_directory = args.output_dir
    
    print(f'Image Directory is :{image_directory}')
    print(f'Output Directory is :{output_directory}')
    
    debugging = (args.debugging == 'True')

    queue = Queue()

    # Load models
    sam_models, clip_models = load_models(
        sam_version=sam_version,sam_ckpt=sam_checkpoint,clip_version=clip_version, clip_ckpt=clip_ckpt, gpu_number=gpu_number)
    
    # Start the loader thread
    loader_thread_instance = threading.Thread(target=loader_thread, args=(image_directory, queue))
    loader_thread_instance.start()
    initial_queue_size = len(os.listdir(image_directory))
    with Progress(BarColumn(), TimeRemainingColumn(), TextColumn("[cyan]Processing images...")) as progress:
        task = progress.add_task("[green]Images processed:", total=initial_queue_size)  # Set total based on initial size
        # Start processing threads
        processing_threads = []
        for i in range(len(sam_models)):
            sam_model = sam_models[i]
            clip_model = clip_models[i]
            thread = threading.Thread(target=process_images, 
                                      args=(queue, sam_model, clip_model, output_directory, debugging, progress, task))
            processing_threads.append(thread)
            thread.start()

        # Wait for the loader to finish
        loader_thread_instance.join()
        # Wait for processing threads to complete their tasks
        for thread in processing_threads:
            thread.join()
        
        
def load_images(image_directory):
    loader, _ = load_dataset(image_directory)
    for images, names in loader:
        yield images, names
        
def loader_thread(image_directory, queue):
    for images, names in load_images(image_directory):
        queue.put((images, names))
    for _ in range(4):  # Assuming 4 processing threads
        queue.put((None, None))

if __name__ == '__main__':
    main()    