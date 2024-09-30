import argparse
from PIL import Image
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Preprocess images by resizing and saving them as JPGs.")
parser.add_argument('--input_folder', required=True, help='Path to input images')
parser.add_argument('--output_folder', required=True, help='Path to save preprocessed images')
parser.add_argument('--max_resolution', type=int, default=1024, help='Maximum resolution for image resizing')
args = parser.parse_args()

# Set input and output directories
input_folder = args.input_folder
output_folder = args.output_folder
max_resolution = (args.max_resolution, args.max_resolution)

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through the input folder and process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):  # Supports multiple image formats
        # Open image
        img = Image.open(os.path.join(input_folder, filename))
        
        # Resize image while maintaining aspect ratio
        img.thumbnail(max_resolution, Image.Resampling.LANCZOS)
        
        # Get the output file name and path
        base_filename = os.path.splitext(filename)[0]  # Get filename without extension
        output_path = os.path.join(output_folder, f"{base_filename}.JPG")
        
        # Convert image to RGB mode if necessary (JPEG does not support transparency)
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        
        # Save image as JPG
        img.save(output_path, format="JPEG")
        print(f"Resized and saved {filename} as JPG with aspect ratio maintained.")

print("All images have been resized, converted to JPG, and the aspect ratio has been maintained.")
