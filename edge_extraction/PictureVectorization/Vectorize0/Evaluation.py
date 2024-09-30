import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from svgpathtools import svg2paths, Line, CubicBezier, QuadraticBezier, Arc
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Suppress specific warnings if desired
warnings.filterwarnings("ignore")

# Ensure the directory exists; create if it does not
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Step 1: Canny edge detection for original 4K image as Ground Truth
def canny_edge_detection(image_path, edge_dir):
    ensure_dir_exists(edge_dir)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Canny edge detection
    edges = cv2.Canny(image, 170, 250)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Save the edge detection result
    file_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(edge_dir, f"{file_name_without_ext}_edges.png")
    cv2.imwrite(output_path, edges_dilated)

    return edges_dilated

# Step 2: Sample the vector paths from the SVG file to create a contour image of 4K size
def svg_to_edge_image(svg_file, svgEdge_folder):
    ensure_dir_exists(svgEdge_folder)
    paths, _ = svg2paths(svg_file)
    height, width = 3643, 5474
    svg_image = np.zeros((height, width), dtype=np.uint8)

    all_points = []
    for path in paths:
        for segment in path:
            all_points.append((segment.start.real, segment.start.imag))
            all_points.append((segment.end.real, segment.end.imag))
            if isinstance(segment, (CubicBezier, QuadraticBezier, Arc)):
                num_samples = 10
                for t in np.linspace(0, 1, num_samples):
                    point = segment.point(t)
                    all_points.append((point.real, point.imag))

    if not all_points:
        return svg_image

    all_points = np.array(all_points)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    svg_width, svg_height = max_x - min_x, max_y - min_y
    scale = min(width / svg_width, height / svg_height)

    offset_x = (width - svg_width * scale) / 2 - min_x * scale
    offset_y = (height - svg_height * scale) / 2 - min_y * scale

    def transform_point(point):
        return int(round(point.real * scale + offset_x)), int(round(point.imag * scale + offset_y))

    for path in paths:
        for segment in path:
            if isinstance(segment, Line):
                start, end = transform_point(segment.start), transform_point(segment.end)
                cv2.line(svg_image, start, end, 255, 2)
            elif isinstance(segment, (CubicBezier, QuadraticBezier, Arc)):
                num_points = 100
                points = [transform_point(segment.point(t / num_points)) for t in range(num_points + 1)]
                for i in range(len(points) - 1):
                    cv2.line(svg_image, points[i], points[i + 1], 255, 2)

    output_filename = os.path.splitext(os.path.basename(svg_file))[0] + "_edge_image.png"
    output_path = os.path.join(svgEdge_folder, output_filename)
    cv2.imwrite(output_path, svg_image)

    return svg_image

# Step 3: Evaluate edge detection using precision, recall, F1-score, IoU, and SSIM
def evaluate_edges(ground_truth, detected_edges, tolerance=3):
    ground_truth_bin = (ground_truth > 127).astype(np.uint8)
    detected_edges_bin = (detected_edges > 127).astype(np.uint8)

    # Distance transform
    distance_ground_truth = cv2.distanceTransform(1 - ground_truth_bin, cv2.DIST_L2, 5)
    distance_detected = cv2.distanceTransform(1 - detected_edges_bin, cv2.DIST_L2, 5)

    TP = np.sum((detected_edges_bin == 1) & (distance_ground_truth <= tolerance))
    FP = np.sum((detected_edges_bin == 1) & (distance_ground_truth > tolerance))
    FN = np.sum((ground_truth_bin == 1) & (distance_detected > tolerance))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    similarity_index, _ = ssim(ground_truth_bin, detected_edges_bin, full=True)
    return precision, recall, f1, iou, similarity_index

# Process a single pair of image and SVG file
def process_single_file(img_file, svg_file, image_dir, svg_dir, edge_dir, svgEdge_folder, tolerance):
    img_path = os.path.join(image_dir, img_file)
    svg_path = os.path.join(svg_dir, svg_file)

    ground_truth = canny_edge_detection(img_path, edge_dir)
    detected_edges = svg_to_edge_image(svg_path, svgEdge_folder)
    precision, recall, f1, iou, ssim_value = evaluate_edges(ground_truth, detected_edges, tolerance=tolerance)

    # Return results for further visualization
    return precision, recall, f1, iou, ssim_value, ground_truth, detected_edges

# Process the entire dataset using parallel threads
def process_dataset(image_dir, svg_dir, edge_dir, svgEdge_folder, tolerance=3):
    precisions, recalls, f1_scores, ious, ssims = [], [], [], [], []
    ground_truths, detected_edges_list = [], []
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    svg_files = sorted([f for f in os.listdir(svg_dir) if f.lower().endswith('.svg')])
    min_len = min(len(image_files), len(svg_files))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_file, image_files[i], svg_files[i], image_dir, svg_dir, edge_dir, svgEdge_folder, tolerance)
            for i in range(min_len)
        ]
        for future in tqdm(as_completed(futures), total=min_len, desc="Processing images and SVGs"):
            result = future.result()
            precision, recall, f1, iou, ssim_value, ground_truth, detected_edges = result
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            ious.append(iou)
            ssims.append(ssim_value)
            ground_truths.append(ground_truth)
            detected_edges_list.append(detected_edges)

    return precisions, recalls, f1_scores, ious, ssims, ground_truths, detected_edges_list

# Visualize the evaluation results as a boxplot
def save_visualization(precisions, recalls, f1_scores, ious, ssims, output_dir):
    ensure_dir_exists(output_dir)
    metrics = {'Precision': precisions, 'Recall': recalls, 'F1-score': f1_scores, 'IoU': ious, 'SSIM': ssims}
    plt.figure(figsize=(12, 8))
    plt.boxplot([precisions, recalls, f1_scores, ious, ssims], labels=metrics.keys())
    plt.title('Evaluation Metrics for Edge Detection on Dataset')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics_boxplot.png'))
    plt.close()

# Visualize contour matching between the ground truth and detected edges
def visualize_contour_matching(ground_truth, detected_edges, output_dir, img_name, tolerance=3):
    ground_truth_bin = (ground_truth > 127).astype(np.uint8)
    detected_edges_bin = (detected_edges > 127).astype(np.uint8)

    # Distance transforms
    distance_ground_truth = cv2.distanceTransform(1 - ground_truth_bin, cv2.DIST_L2, 5)
    distance_detected = cv2.distanceTransform(1 - detected_edges_bin, cv2.DIST_L2, 5)

    # Create an RGB image for comparison
    height, width = ground_truth_bin.shape
    comparison_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Matching rules:
    # - Unmatched ground truth edges in blue
    comparison_image[(ground_truth_bin == 1) & (distance_detected > tolerance)] = [0, 0, 255]

    # - Detected edges in red
    comparison_image[(detected_edges_bin == 1) & (distance_ground_truth > tolerance)] = [255, 0, 0]

    # - Matched edges in green
    comparison_image[(detected_edges_bin == 1) & (distance_ground_truth <= tolerance)] = [0, 255, 0]

    # Save and show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(comparison_image)
    plt.title(f"Contour Matching for {img_name}")
    plt.axis('off')

    output_file = os.path.join(output_dir, f"{img_name}_contour_matching.png")
    plt.savefig(output_file)
    plt.close()

# Main function to run the evaluation
def main(image_dir, svg_dir, output_dir, edge_dir, svgEdge_folder, tolerance=3):
    ensure_dir_exists(output_dir)
    ensure_dir_exists(edge_dir)
    ensure_dir_exists(svgEdge_folder)

    precisions, recalls, f1_scores, ious, ssims, ground_truths, detected_edges_list = process_dataset(
        image_dir, svg_dir, edge_dir, svgEdge_folder, tolerance
    )

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    for i, img_file in enumerate(image_files):
        img_name = os.path.splitext(img_file)[0]
        visualize_contour_matching(ground_truths[i], detected_edges_list[i], output_dir, img_name, tolerance)

    save_visualization(precisions, recalls, f1_scores, ious, ssims, output_dir)

# Entry point to the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate edge detection and vectorization results.")
    parser.add_argument('--image_dir', required=True, help='Directory containing the original images')
    parser.add_argument('--svg_dir', required=True, help='Directory containing the SVG files')
    parser.add_argument('--output_dir', required=True, help='Directory to save the evaluation visualizations')
    parser.add_argument('--edge_dir', required=True, help='Directory to save the original edge detection results')
    parser.add_argument('--svgEdge_folder', required=True, help='Directory to save the SVG edge images')
    parser.add_argument('--tolerance', type=int, default=3, help='Tolerance for edge matching (default: 3)')
    
    args = parser.parse_args()

    main(args.image_dir, args.svg_dir, args.output_dir, args.edge_dir, args.svgEdge_folder, args.tolerance)
