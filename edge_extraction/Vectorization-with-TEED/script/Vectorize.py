import argparse
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from rdp import rdp
import svgwrite
import os
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, to_tree
from multiprocessing import Pool, cpu_count
import functools

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Vectorize images with hierarchical clustering and mask filtering')
parser.add_argument('--dataset_path', required=True, help='Path to input dataset with images')
parser.add_argument('--edge_dir', required=True, help='Path to the edge detection results')
parser.add_argument('--output_dir', required=True, help='Directory to save the output SVG files')
parser.add_argument('--n_clusters', type=int, default=16, help='Number of clusters for KMeans')
parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon for RDP algorithm')
parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth for hierarchical clustering')
parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for mask filtering')
args = parser.parse_args()

# Check if CUDA is available
def is_cuda_available():
    try:
        cv2.cuda.getCudaEnabledDeviceCount()
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except AttributeError:
        return False

CUDA_AVAILABLE = is_cuda_available()

# Helper function: Convert BGR to RGB
def bgr_to_rgb(color):
    return [color[2], color[1], color[0]]

# Step 1: Clustering function (using MiniBatchKMeans on CPU)
def clustering(image, n_clusters=16):
    pixels = image.reshape(-1, 3).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000)
    clusters = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    return clusters.reshape(image.shape[:2]), centers

# Step 2: Hierarchical clustering using scipy
def hierarchical_clustering(centers, method='average'):
    Z = linkage(centers, method=method, metric='euclidean')
    return Z

# Helper function: Convert linkage matrix to tree
def linkage_to_tree(Z):
    root, _ = to_tree(Z, rd=True)
    return root

# Step 3: Extract contours for each cluster with mask filtering
def extract_cluster_contours(clusters, mask, threshold=0.9):
    contours_list = []
    for cluster_val in np.unique(clusters):
        mask_cluster = (clusters == cluster_val).astype(np.uint8) * 255
        # Use more efficient retrieval mode and approximation method to extract contours
        contours, _ = cv2.findContours(mask_cluster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Calculate the proportion of contour points inside the mask
            if len(contour) == 0:
                continue
            contour_points = contour[:, 0, :]  # Shape is (num_points, 2)
            # Ensure that the coordinates are within the image boundaries
            valid_indices = (contour_points[:, 0] >= 0) & (contour_points[:, 0] < mask.shape[1]) & \
                            (contour_points[:, 1] >= 0) & (contour_points[:, 1] < mask.shape[0])
            contour_points = contour_points[valid_indices]
            if len(contour_points) == 0:
                continue
            mask_values = mask[contour_points[:, 1], contour_points[:, 0]]
            ratio = np.sum(mask_values > 0) / len(mask_values)
            if ratio >= threshold:
                contours_list.append((cluster_val, contour))
    return contours_list

# Step 4: Simplify contour path (RDP algorithm)
def simplify_path(contour, epsilon):
    simplified = rdp(contour[:, 0, :], epsilon=epsilon)
    return simplified

# Step 5: Recursive function to draw each level of the hierarchical tree with depth control
def draw_hierarchical_level(dwg, node, depth, max_depth, centers, clusters, mask, image_shape, epsilon, threshold):
    if depth > max_depth:
        return

    if node.is_leaf():
        cluster_idx = node.id
        # Get the color of the current cluster
        color = centers[cluster_idx]
        color_rgb = bgr_to_rgb(color)
        hex_color = '#%02x%02x%02x' % (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))

        # Create the mask for the current cluster
        mask_cluster = (clusters == cluster_idx).astype(np.uint8) * 255
        # Use more efficient retrieval mode and approximation method to extract contours
        contours, _ = cv2.findContours(mask_cluster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculate the proportion of contour points inside the mask
            if len(contour) == 0:
                continue
            contour_points = contour[:, 0, :]
            # Ensure that the coordinates are within the image boundaries
            valid_indices = (contour_points[:, 0] >= 0) & (contour_points[:, 0] < mask.shape[1]) & \
                            (contour_points[:, 1] >= 0) & (contour_points[:, 1] < mask.shape[0])
            contour_points = contour_points[valid_indices]
            if len(contour_points) == 0:
                continue
            mask_values = mask[contour_points[:, 1], contour_points[:, 0]]
            ratio = np.sum(mask_values > 0) / len(mask_values)
            if ratio < threshold:
                continue  # Ignore contours that do not meet the threshold

            simplified = simplify_path(contour, epsilon)
            if len(simplified) < 2:
                continue  # Ignore overly simple paths
            path_data = "M " + " L ".join(f"{int(point[0])},{int(point[1])}" for point in simplified) + " Z"
            dwg.add(dwg.path(d=path_data, fill=hex_color, stroke='black', stroke_width=0.5))
    else:
        # Non-leaf node: Recursively draw child nodes
        draw_hierarchical_level(dwg, node.left, depth + 1, max_depth, centers, clusters, mask, image_shape, epsilon, threshold)
        draw_hierarchical_level(dwg, node.right, depth + 1, max_depth, centers, clusters, mask, image_shape, epsilon, threshold)

# Step 6: Save the result as SVG
def save_as_svg_recursive(tree, centers, clusters, mask, output_svg, image_shape, epsilon, max_depth, threshold):
    dwg = svgwrite.Drawing(output_svg, profile='tiny', size=(image_shape[1], image_shape[0]))

    # Start from the root node of the tree
    draw_hierarchical_level(dwg, tree, depth=0, max_depth=max_depth, centers=centers, clusters=clusters,
                            mask=mask, image_shape=image_shape, epsilon=epsilon, threshold=threshold)

    dwg.save()

# Gaussian blur: Use GPU acceleration if available, otherwise use CPU
def gaussian_blur(image, ksize=(7,7), sigma=0):
    if CUDA_AVAILABLE:
        try:
            image_gpu = cv2.cuda_GpuMat()
            image_gpu.upload(image)
            gaussian_filter = cv2.cuda.createGaussianFilter(image_gpu.type(), image_gpu.type(), ksize, sigma)
            blurred_gpu = gaussian_filter.apply(image_gpu)
            blurred = blurred_gpu.download()
            return blurred
        except cv2.error as e:
            print(f"CUDA Gaussian blur failed: {e}. Using CPU.")
    # Use CPU if GPU is not available
    return cv2.GaussianBlur(image, ksize, sigma)

# Main processing function: Process a single image
def process_image(image_path, output_dir, edge_dir, n_clusters=16, epsilon=1.0, max_depth=3, threshold=0.9):
    filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(filename)
    output_svg = os.path.join(output_dir, f"{base_name}.svg")

    # Build the path to the edge detection result
    edge_filename = f"{base_name}.png"
    edge_path = os.path.join(edge_dir, edge_filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read image {image_path}. Skipping.")
        return

    # Read the edge detection result
    edge_image = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if edge_image is None:
        print(f"Warning: Unable to read edge image {edge_path}. Skipping.")
        return

    # Create a mask: Since edges are black (0) and background is white (255), we need to invert the values
    _, mask = cv2.threshold(edge_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Use GPU-accelerated Gaussian blur if available
    blurred_image = gaussian_blur(image, (3, 3), 0)

    # Step 1: Perform clustering (using MiniBatchKMeans)
    clusters, centers = clustering(blurred_image, n_clusters)

    # Step 2: Perform hierarchical clustering using scipy
    Z = hierarchical_clustering(centers, method='average')
    tree = linkage_to_tree(Z)

    # Step 5 & 6: Save as SVG file, pass the mask and threshold
    save_as_svg_recursive(tree, centers, clusters, mask, output_svg, blurred_image.shape, epsilon, max_depth, threshold)

# Main function: Process the entire dataset in parallel
def Vectorize(dataset_path, edge_dir, output_dir, n_clusters=16, epsilon=1.0, max_depth=3, threshold=0.9):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the list of image files in the dataset (only JPG and PNG formats)
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.png'))]

    if not image_files:
        print("No valid image files found.")
        return

    # Define partially fixed parameters
    process_func = functools.partial(
        process_image,
        output_dir=output_dir,
        edge_dir=edge_dir,
        n_clusters=n_clusters,
        epsilon=epsilon,
        max_depth=max_depth,
        threshold=threshold
    )

    # Use multiprocessing to process the images in parallel
    cpu_cores = cpu_count()
    with Pool(processes=cpu_cores) as pool:
        # Display a progress bar using tqdm
        list(tqdm(pool.imap(process_func, image_files), total=len(image_files), desc="Processing images"))

if __name__ == "__main__":
    Vectorize(
        dataset_path=args.dataset_path,
        edge_dir=args.edge_dir,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        epsilon=args.epsilon,
        max_depth=args.max_depth,
        threshold=args.threshold
    )
