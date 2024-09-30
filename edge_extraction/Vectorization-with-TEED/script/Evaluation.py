import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim
from svgpathtools import svg2paths, Path, Line, CubicBezier, QuadraticBezier, Arc
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import seaborn as sns

# Suppress specific warnings if desired
warnings.filterwarnings("ignore")

# Helper function to ensure directories exist
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Step 1: Use Canny edge detection to extract contours from a 4K original image as ground truth
def canny_edge_detection(image_path, edge_dir):
    ensure_dir_exists(edge_dir)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Canny edge detection
    edges = cv2.Canny(image, 150, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)

    # Save edge detection result
    file_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(edge_dir, f"{file_name_without_ext}_edges.png")
    cv2.imwrite(output_path, edges_eroded)

    return edges_eroded

# Step 2: Sample the vector from the provided SVG file into a 4K contour image
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

# Step 3: Evaluate the detection results using different evaluation metrics
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

# Batch process images and SVG files in a dataset
def process_single_file(img_file, svg_file, image_dir, svg_dir, edge_dir, svgEdge_folder, tolerance):
    img_path = os.path.join(image_dir, img_file)
    svg_path = os.path.join(svg_dir, svg_file)

    ground_truth = canny_edge_detection(img_path, edge_dir)
    detected_edges = svg_to_edge_image(svg_path, svgEdge_folder)
    precision, recall, f1, iou, ssim_value = evaluate_edges(ground_truth, detected_edges, tolerance=tolerance)

    return precision, recall, f1, iou, ssim_value, ground_truth, detected_edges

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

# Visualization
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

# Main Function
def main(image_dir, svg_dir, output_dir, edge_dir, svgEdge_folder, tolerance=3):
    ensure_dir_exists(output_dir)
    ensure_dir_exists(edge_dir)
    ensure_dir_exists(svgEdge_folder)

    precisions, recalls, f1_scores, ious, ssims, ground_truths, detected_edges_list = process_dataset(
        image_dir, svg_dir, edge_dir, svgEdge_folder, tolerance
    )

    save_visualization(precisions, recalls, f1_scores, ious, ssims, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image and SVG datasets")
    parser.add_argument('--image_dir', required=True, help="Directory containing input images")
    parser.add_argument('--svg_dir', required=True, help="Directory containing input SVG files")
    parser.add_argument('--output_dir', required=True, help="Directory to save results")
    parser.add_argument('--edge_dir', required=True, help="Directory to save edge detection results")
    parser.add_argument('--svgEdge_folder', required=True, help="Directory to save SVG edge images")
    parser.add_argument('--tolerance', type=int, default=3, help="Tolerance for contour matching")

    args = parser.parse_args()

    main(
        image_dir=args.image_dir,
        svg_dir=args.svg_dir,
        output_dir=args.output_dir,
        edge_dir=args.edge_dir,
        svgEdge_folder=args.svgEdge_folder,
        tolerance=args.tolerance
    )
