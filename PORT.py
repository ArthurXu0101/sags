'''
    This document is the main guidance for the designate part for software architecture

    The system is mainly contain 3 part
    - Edge extraction and analysis part 
    - Mask generation part
    - Gaussian Training Part

    We define the correlation between these three part
'''

import numpy as np
import torch
from typing import Tuple
from pathlib import Path
from dataclasses import dataclass

@dataclass
class segmentation_args:
    threshold: float = 0.5
    """This threshold is to adjust how scatter a segmentation is. For example, if the model segments too fine, we can adjust this threshold"""

    TBD = 'TBD'
    """whats next"""

@dataclass
class edge_args: 
    output_mode: str = 'edge' 
    """if the mode is set to be edge, then we need to use linear to approximate our figure. Otherwise, we will need to use Fourier series"""

    processing_mode: str = 'svg'
    """if the mode is set to be svg, we then use traditional way, otherwise use deep learning model"""

    TBD = 'TBD'
    """whats next"""

    assert output_mode in ['edge', 'frequency'] # if it is model, we use deep model
    assert processing_mode in ['svg', 'model'] # if it is model, we use deep model

def mask_generation(image: np.ndarray, 
                    model, 
                    visualization: bool = False, 
                    logging_location: Path = None, 
                    args: segmentation_args = segmentation_args()) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        The mask generation will take in image and extract different mask according to semantic segmentation mask

        Args:
            image: np.ndarray (H,W,3) uint8, RGB
            model: this is a segmentation model, now is set as SAM, but in future, we would like to use self-supervised model to segment different semantic group
            visualization: if visualization is set to true, the visualized result should be store at location specify by logging_location
            logging_location: if visualization is set, we should save image to logging_location
        Return:
            Masks: torch.Tensor The masks is in the shape of (H,W) uint8, each pixel represent which semantic group it belongs to
            Masks_logits: The mask probability of seperation (we might use this to solve over segmentation)
    '''

def edge_extraction(
        image: np.ndarray, 
        visualization: bool,
        logging_location: Path = None, 
        args: edge_args = edge_args(),
        model = None) -> Tuple[torch.Tensor]:
    '''
        The edge extraction will takes in images and it will convert it into edges figure
        Args: 
            image: np.ndarray (H,W,3) uint8, RGB
            visualization: if visualization is set to true, the visualized result should be store at location specify by logging_location
            logging_location: if visualization is set, we should save image to logging_location
            args: argument list, check the class definition
            model: if the args.processing_mode is set to be model, passing model here
        
        Return: 
            edges: torch.Tensor in the shape of (n,4) if args.output_mode set to be edge. Otherwise (n,2)
            (n,4) means (x1, y1, x2, y2), (n, 2) means (fxi, fyi)
    '''
    if args.processing_mode == 'model':
        raise NotImplementedError

def edge_analysis(
        masks: torch.Tensor,
        edges: torch.Tensor,
        args: edge_args
) -> Tuple[torch.Tensor]:
    '''
        The edge power analysis should follow the following stage, we analysis frequency or edge distribution in each mask region
        and we give out a mask shape guidance. This shape guidance is mask-wise. Basically, if the frequncy power is really high,
        we need to make gaussian splats slim, otherwise fat

        Args: 
            masks: The masks is in the shape of (H,W) uint8, each pixel represent which semantic group it belongs to
            edges: can be (n,4) or (n,2) according to args.output_mode
            args: need to 
        Return: 
            shape: torch.Tensor (H,W,2). The first represent the guided shape of (sigma_x/sigma_y, sigma_x/sigma_z)
    '''


def gaussian_split_call_back(
        gaussian_attributs,
        edge_map,
)-> None:
    '''
        according to edge and project inference, we can do the following, if CROSS boundaries, we split to different sides, otherwise
    '''

def gaussian_loss(
        gaussian_attributes,
        mask_map, 
):
