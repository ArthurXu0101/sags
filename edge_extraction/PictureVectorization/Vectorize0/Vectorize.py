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

# Check if CUDA is available
def is_cuda_available():
    try:
        cv2.cuda.getCudaEnabledDeviceCount()
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except AttributeError:
        return False

CUDA_AVAILABLE = is_cuda_available()

# Helper function to convert BGR to RGB
def bgr_to_rgb(color):
    return [color[2], color[1], color[0]]

# Function to perform KMeans clustering on the image
def clustering(image, n_clusters=16):
    pixels = image.reshape(-1, 3).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000)
    clusters = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    return clusters.reshape(image.shape[:2]), centers

# Perform hierarchical clustering
def hierarchical_clustering(centers, method='average'):
    Z = linkage(centers, method=method, metric='euclidean')
    return Z

# Convert linkage matrix to tree structure
def linkage_to_tree(Z):
    root, _ = to_tree(Z, rd=True)
    return root

# Extract contours from the image based on cluster masks
def extract_cluster_contours(clusters):
    contours_list = []
    for cluster_val in np.unique(clusters):
        mask = (clusters == cluster_val).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append((cluster_val, contours))
    return contours_list

# Simplify contour using the RDP algorithm
def simplify_path(contour, epsilon):
    simplified = rdp(contour[:, 0, :], epsilon=epsilon)
    return simplified

# Draw SVG paths recursively for hierarchical clustering
def draw_hierarchical_level(dwg, node, depth, max_depth, centers, clusters, image_shape, epsilon):
    if depth > max_depth:
        return

    if node.is_leaf():
        cluster_idx = node.id
        color = centers[cluster_idx]
        color_rgb = bgr_to_rgb(color)
        hex_color = '#%02x%02x%02x' % (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))

        mask = (clusters == cluster_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            simplified = simplify_path(contour, epsilon)
            if len(simplified) < 2:
                continue
            path_data = "M " + " L ".join(f"{int(point[0])},{int(point[1])}" for point in simplified) + " Z"
            dwg.add(dwg.path(d=path_data, fill=hex_color, stroke='black', stroke_width=0.5))
    else:
        draw_hierarchical_level(dwg, node.left, depth + 1, max_depth, centers, clusters, image_shape, epsilon)
        draw_hierarchical_level(dwg, node.right, depth + 1, max_depth, centers, clusters, image_shape, epsilon)

# Save the result as an SVG file
def save_as_svg_recursive(tree, centers, clusters, output_svg, image_shape, epsilon, max_depth):
    dwg = svgwrite.Drawing(output_svg, profile='tiny', size=(image_shape[1], image_shape[0]))
    draw_hierarchical_level(dwg, tree, 0, max_depth, centers, clusters, image_shape, epsilon)
    dwg.save()

# Apply Gaussian blur with optional GPU acceleration
def gaussian_blur(image, ksize=(3,3), sigma=0):
    if CUDA_AVAILABLE:
        try:
            image_gpu = cv2.cuda_GpuMat()
            image_gpu.upload(image)
            blurred_gpu = cv2.cuda.createGaussianFilter(image_gpu.type(), image_gpu.type(), ksize, sigma).apply(image_gpu)
            blurred = blurred_gpu.download()
            return blurred
        except cv2.error as e:
            print(f"CUDA Gaussian blur failed: {e}. Using CPU.")
    return cv2.GaussianBlur(image, ksize, sigma)

# Process a single image and convert it into SVG format
def process_image(image_path, output_dir, n_clusters=16, epsilon=1.0, max_depth=3):
    filename = os.path.basename(image_path)
    output_svg = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.svg")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read image {image_path}. Skipping.")
        return

    # Apply Gaussian blur
    blurred_image = gaussian_blur(image, (3, 3), 0)

    # Perform clustering
    clusters, centers = clustering(blurred_image, n_clusters)

    # Perform hierarchical clustering
    Z = hierarchical_clustering(centers, method='average')
    tree = linkage_to_tree(Z)

    # Save result as SVG
    save_as_svg_recursive(tree, centers, clusters, output_svg, blurred_image.shape, epsilon, max_depth)

# Vectorize an entire dataset in parallel
def Vectorize(dataset_path, output_dir, n_clusters=16, epsilon=1.0, max_depth=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.png'))]

    if not image_files:
        print("No image files found.")
        return

    process_func = functools.partial(process_image, output_dir=output_dir, n_clusters=n_clusters, epsilon=epsilon, max_depth=max_depth)
    cpu_cores = cpu_count()
    with Pool(processes=cpu_cores) as pool:
        list(tqdm(pool.imap(process_func, image_files), total=len(image_files), desc="Processing images"))

# Parse command-line arguments for vectorization process
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize images and save them as SVG files.")
    parser.add_argument('--dataset_path', required=True, help='Path to input dataset')
    parser.add_argument('--output_dir', required=True, help='Path to save vectorized images')
    parser.add_argument('--n_clusters', type=int, default=16, help='Number of clusters for KMeans')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon for RDP algorithm')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth for hierarchical clustering')
    args = parser.parse_args()

    Vectorize(args.dataset_path, args.output_dir, args.n_clusters, args.epsilon, args.max_depth)

