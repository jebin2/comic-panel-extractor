from PIL import Image, ImageDraw
import imageio.v2 as imageio
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import shutil
from glob import glob
from typing import List, Union
from .config import Config, load_config
from shapely.geometry import Polygon

config = load_config()

def remove_duplicate_boxes(boxes, compare_single=None, iou_threshold=0.7):
	"""
	Removes duplicate or highly overlapping boxes, keeping the larger one.
	:param boxes: List of (x1, y1, x2, y2) boxes.
	:param compare_single: Optional single box to compare against the list.
	:param iou_threshold: IOU threshold to consider as duplicate.
	:return: 
		- If compare_single is None: deduplicated list of boxes.
		- If compare_single is provided: tuple (is_duplicate, updated_box_or_none)
	"""
	def compute_iou(boxA, boxB):
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
		interArea = max(0, xB - xA) * max(0, yB - yA)
		if interArea == 0:
			return 0.0
		boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
		boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
		return interArea / float(boxAArea + boxBArea - interArea)

	def compute_area(box):
		return (box[2] - box[0]) * (box[3] - box[1])

	# Single comparison mode
	if compare_single is not None:
		single_area = compute_area(compare_single)
		for existing_box in boxes:
			iou = compute_iou(compare_single, existing_box)
			if iou > iou_threshold:
				existing_area = compute_area(existing_box)
				if single_area > existing_area:
					return True, compare_single  # Keep new (larger) box
				else:
					return True, None  # Existing box is better, discard new
		return False, compare_single  # No overlap found, keep it

	# Bulk deduplication mode
	unique_boxes = []
	for box in boxes:
		box_area = compute_area(box)
		replaced_existing = False
		
		# Check against existing unique boxes
		for i, ubox in enumerate(unique_boxes):
			if compute_iou(box, ubox) > iou_threshold:
				ubox_area = compute_area(ubox)
				# If current box is larger, replace the existing one
				if box_area > ubox_area:
					unique_boxes[i] = box
					replaced_existing = True
				# If existing box is larger or equal, ignore current box
				break
		
		# If no overlap found, add the box
		if not replaced_existing and not any(compute_iou(box, ubox) > iou_threshold for ubox in unique_boxes):
			unique_boxes.append(box)

	print(f"✅ Found {abs(len(unique_boxes) - len(boxes))} duplicates")
	return unique_boxes

def count_panels_inside(target_box, other_boxes, height=None, width=None):
	x1a, y1a, x2a, y2a = target_box
	target_area = (x2a - x1a) * (y2a - y1a)
	count = 0
	total_covered_area = 0
	for x1b, y1b, x2b, y2b in other_boxes:
		if x1a <= x1b and y1a <= y1b and x2a >= x2b and y2a >= y2b:
			count += 1

	# Only apply area threshold check if height and width are provided
	if height is not None and width is not None:
		if total_covered_area / target_area < 0.8:
			return 0
	return count

def extend_boxes_to_image_border(boxes, image_shape, min_width_ratio, min_height_ratio):
	"""
	Extends any side of a bounding box to the image border if it's close enough.
	
	:param boxes: List of (x1, y1, x2, y2) tuples.
	:param image_shape: (height, width) of the image.
	:param threshold: Pixel threshold to snap to border.
	:return: List of adjusted boxes.
	"""
	if not boxes:
		return boxes
	extended_boxes = [list(box) for box in boxes]
	width, height = image_shape
	adjusted_boxes = []

	width_threshold = width * min_width_ratio
	height_threshold = height * min_height_ratio

	# width_threshold = self.config.min_width_ratio * width
	# height_threshold = self.config.min_height_ratio * height

	percent_threshold=0.8
	for x1, y1, x2, y2 in boxes:
		box_width = x2 - x1
		box_height = y2 - y1

		# Snap if close to left or top
		if abs(x1 - 0) <= width_threshold or box_width >= percent_threshold * width:
			x1 = 0
		if abs(y1 - 0) <= height_threshold or box_height >= percent_threshold * height:
			y1 = 0

		# Snap if close to right or bottom
		if abs(x2 - width) <= width_threshold or box_width >= percent_threshold * width:
			x2 = width
		if abs(y2 - height) <= height_threshold or box_height >= percent_threshold * height:
			y2 = height
		adjusted_boxes.append((x1, y1, x2, y2))

	return adjusted_boxes

def draw_black(image_path, accepted_boxes, output_path, stripe = True) -> str:
	orig_pil = Image.fromarray(imageio.imread(image_path))
	width, height = orig_pil.size

	# Create a global stripe pattern (black and white horizontal stripes)
	stripe_img = Image.new("RGB", (width, height), (255, 255, 255))
	draw = ImageDraw.Draw(stripe_img)
	stripe_height = 10

	if stripe:
		for y in range(0, height, stripe_height):
			if (y // stripe_height) % 2 == 0:
				draw.rectangle([0, y, width, min(y + stripe_height, height)], fill=(0, 0, 0))

	# Create a mask where accepted boxes will be applied
	mask = Image.new("L", (width, height), 0)
	mask_draw = ImageDraw.Draw(mask)
	for x1, y1, x2, y2 in accepted_boxes:
		mask_draw.rectangle([x1, y1, x2, y2], fill=255)

	# Paste the striped image only where mask is white (inside accepted boxes)
	orig_pil.paste(stripe_img, (0, 0), mask)

	orig_pil.save(output_path)
	return output_path

def extend_to_nearby_boxes(boxes, image_shape, min_width_ratio, min_height_ratio):
	"""
	Extends boxes to the edge of any close neighboring box without causing
	unintended merging by using an atomic update approach.

	A box is represented by (x1, y1, x2, y2).
	"""
	if not boxes:
		return boxes

	width, height = image_shape

	width_threshold = width * min_width_ratio
	height_threshold = height * min_height_ratio

	final_boxes = []
	# For each box, calculate its new coordinates based on the original list
	for i in range(len(boxes)):
		# Start with the original coordinates for the box we're currently processing
		x1, y1, x2, y2 = boxes[i]

		# These will store the closest boundaries we can extend to,
		# initialized to the image edges.
		closest_left_boundary = 0
		closest_right_boundary = width
		closest_top_boundary = 0
		closest_bottom_boundary = height

		# Find the closest neighbor on each of the four sides by checking against ALL other boxes
		for j in range(len(boxes)):
			if i == j:
				continue

			x1_j, y1_j, x2_j, y2_j = boxes[j]

			# Check for neighbors to the RIGHT of box `i`
			is_vert_overlap = (y1 < y2_j and y2 > y1_j) # Do they overlap vertically?
			is_right_neighbor = (x1_j >= x2)			 # Is box `j` to the right of `i`?
			if is_vert_overlap and is_right_neighbor:
				closest_right_boundary = min(closest_right_boundary, x1_j)

			# Check for neighbors to the LEFT of box `i`
			is_left_neighbor = (x2_j <= x1)			  # Is box `j` to the left of `i`?
			if is_vert_overlap and is_left_neighbor:
				closest_left_boundary = max(closest_left_boundary, x2_j)

			# Check for neighbors BELOW box `i`
			is_horiz_overlap = (x1 < x2_j and x2 > x1_j) # Do they overlap horizontally?
			is_bottom_neighbor = (y1_j >= y2)			# Is box `j` below `i`?
			if is_horiz_overlap and is_bottom_neighbor:
				closest_bottom_boundary = min(closest_bottom_boundary, y1_j)
			
			# Check for neighbors ABOVE box `i`
			is_top_neighbor = (y2_j <= y1)			   # Is box `j` above `i`?
			if is_horiz_overlap and is_top_neighbor:
				closest_top_boundary = max(closest_top_boundary, y2_j)

		# --- Apply the calculated extensions ---
		
		# Extend right if the closest gap on the right is within the threshold
		if 0 < (closest_right_boundary - x2) <= width_threshold:
			x2 = closest_right_boundary

		# Extend left
		if 0 < (x1 - closest_left_boundary) <= width_threshold:
			x1 = closest_left_boundary

		# Extend down
		if 0 < (closest_bottom_boundary - y2) <= height_threshold:
			y2 = closest_bottom_boundary
			
		# Extend up
		if 0 < (y1 - closest_top_boundary) <= height_threshold:
			y1 = closest_top_boundary
			
		final_boxes.append(tuple(map(int, (x1, y1, x2, y2))))
		
	return final_boxes

def convert_to_grayscale_pil(input_path, output_path):
	with Image.open(input_path) as img:
		gray_img = img.convert("L")  # "L" mode = grayscale
		gray_img.save(output_path)

	return output_path

def convert_to_clahe(input_path, output_path):
	# Read image from disk
	image = cv2.imread(input_path)

	if image is None:
		raise FileNotFoundError(f"Could not read image from path: {input_path}")

	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Apply CLAHE
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	output = clahe.apply(gray)

	# Save the processed image
	cv2.imwrite(output_path, output)

	return output_path

def convert_to_lab_l(input_path, output_path):
	# Read image from disk
	image = cv2.imread(input_path)

	if image is None:
		raise FileNotFoundError(f"Could not read image from path: {input_path}")

	output = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]

	# Save the processed image
	cv2.imwrite(output_path, output)

	return output_path

def convert_to_group_colors(input_path, output_path, num_clusters: int = 5):
	# Load image
	image = Image.open(input_path).convert("RGB")
	np_image = np.array(image)
	h, w = np_image.shape[:2]
	pixels = np_image.reshape(-1, 3)

	# Run KMeans
	kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
	labels = kmeans.fit_predict(pixels)
	centers = kmeans.cluster_centers_.astype(np.uint8)

	# Replace pixels with their cluster center color
	clustered_pixels = centers[labels].reshape(h, w, 3)

	# Save using OpenCV (convert RGB to BGR)
	output = clustered_pixels[:, :, ::-1]

	# Save the processed image
	cv2.imwrite(output_path, output)

	return output_path

def get_black_white_ratio(image_path: str, threshold: int = 128) -> dict:
	"""
	Calculate the ratio of black and white pixels in a binary image.
	
	Args:
		image_path: Path to the image file
		threshold: Threshold value for binarization
	
	Returns:
		Dictionary with pixel ratios and counts
	"""
	# Load and process image
	img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise FileNotFoundError(f"Image not found: {image_path}")

	# Convert to binary
	_, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

	# Calculate ratios
	total_pixels = binary.size
	white_count = np.count_nonzero(binary == 255)
	black_count = total_pixels - white_count

	return {
		"black_ratio": black_count / total_pixels,
		"white_ratio": white_count / total_pixels,
		"black_count": black_count,
		"white_count": white_count,
		"total_pixels": total_pixels
	}

def box_covered_ratio(boxes, image_shape) -> float:
	"""
	Calculate the ratio of area covered by boxes to the image area,
	accounting for overlapping boxes by using a mask.

	Args:
		boxes (List[Tuple[int, int, int, int]]): List of (x1, y1, x2, y2) boxes.
		image_shape (Tuple[int, int]): (width, height) of the image.

	Returns:
		float: Ratio between 0 and 1.
	"""
	width, height = image_shape
	image_area = width * height

	if image_area == 0 or not boxes:
		return 0.0

	# Create a white mask
	mask = np.ones((height, width), dtype=np.uint8) * 255

	# Draw black rectangles (panels)
	for x1, y1, x2, y2 in boxes:
		cv2.rectangle(mask, (x1, y1), (x2, y2), color=0, thickness=-1)

	# Count black pixels
	black_pixels = np.sum(mask == 0)

	return black_pixels / image_area

def find_similar_remaining_regions(boxes, image_shape, debug_image_path, w_t=0.25, h_t=0.25):
	"""
	Find remaining regions not covered by original boxes that match any original box's
	width and height within a given threshold.

	Args:
		boxes (List[Tuple[int, int, int, int]]): Original (x1, y1, x2, y2) boxes.
		image_shape (Tuple[int, int]): (width, height) of the image.
		debug_image_path (str): Path to save debug image.
		w_t (float): Width threshold (e.g., 0.1 = ±10%)
		h_t (float): Height threshold (e.g., 0.1 = ±10%)

	Returns:
		Tuple[List[Tuple[int, int, int, int]], np.ndarray]: 
			- List of new similar boxes
			- Debug image with overlays
	"""
	width, height = image_shape
	mask = np.ones((height, width), dtype=np.uint8) * 255

	for x1, y1, x2, y2 in boxes:
		mask[y1:y2, x1:x2] = 0

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if not boxes:
		return []

	similar_boxes = []
	debug_img = np.full((height, width, 3), 255, dtype=np.uint8)

	# Draw original boxes in green
	for x1, y1, x2, y2 in boxes:
		cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 10)

	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		box = (x, y, x + w, y + h)

		matched = False
		for x1, y1, x2, y2 in boxes:
			bw = x2 - x1
			bh = y2 - y1

			width_match = abs(w - bw) / bw <= w_t
			height_match = abs(h - bh) / bh <= h_t

			if width_match and height_match:
				matched = True
				break

		if matched:
			similar_boxes.append(box)
			cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 10)  # Blue: Accepted
		else:
			cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 10)  # Red: Rejected

	cv2.imwrite(debug_image_path, debug_img)
	return similar_boxes

def get_remaining_areas(image_size, boxes):
	"""
	Given the image size and a list of bounding boxes, returns the remaining uncovered areas
	as rectangles.

	Args:
		image_size: (width, height) of the image.
		boxes: List of (x1, y1, x2, y2) rectangles.

	Returns:
		List of rectangles representing the remaining uncovered areas.
	"""
	width, height = image_size
	# Create a binary mask of the image (0 = uncovered, 255 = covered)
	mask = np.zeros((height, width), dtype=np.uint8)

	# Mark the covered boxes
	for x1, y1, x2, y2 in boxes:
		mask[y1:y2, x1:x2] = 255

	# Invert mask to get the remaining area
	remaining_mask = cv2.bitwise_not(mask)

	# Find contours in the remaining area
	contours, _ = cv2.findContours(remaining_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	remaining_boxes = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		remaining_boxes.append((x, y, x + w, y + h))

	return remaining_boxes

def is_valid_panel(
	image_size,
	boxes,
	min_width_ratio: float,
	min_height_ratio: float
):
	"""
	Check if each panel (box) is valid based on minimum width and height ratio of image size.

	Args:
		image_size: (width, height) of the image.
		boxes: List of (x1, y1, x2, y2) panel boxes.
		min_width_ratio: Minimum allowed width as a ratio of image width (e.g. 0.05).
		min_height_ratio: Minimum allowed height as a ratio of image height (e.g. 0.05).

	Returns:
		List of booleans indicating if each panel is valid.
	"""
	image_width, image_height = image_size
	min_width = image_width * min_width_ratio
	min_height = image_height * min_height_ratio

	validity = []
	for x1, y1, x2, y2 in boxes:
		box_width = x2 - x1
		box_height = y2 - y1
		is_valid = box_width >= min_width and box_height >= min_height
		if is_valid:
			validity.append((x1, y1, x2, y2))

	return validity

def get_abs_path(relative_path: str) -> str:
	"""Convert relative path to absolute path."""
	return os.path.abspath(relative_path)

def get_image_paths(directories: Union[str, List[str]]) -> List[str]:
	"""
	Get all image paths from given directories.
	
	Args:
		directories: Single directory path or list of directory paths
		
	Returns:
		List of image file paths
	"""
	if isinstance(directories, str):
		directories = [directories]
	
	all_images = []
	for directory in directories:
		abs_dir = get_abs_path(directory)
		if not os.path.isdir(abs_dir):
			print(f"⚠️ Warning: Skipping non-directory {abs_dir}")
			continue
			
		# Support multiple image extensions
		for ext in config.SUPPORTED_EXTENSIONS:
			pattern = os.path.join(abs_dir, f'*.{ext}')
			images = sorted(glob(pattern))
			all_images.extend(images)
	
	return list(set(all_images))  # Remove duplicates

def clean_directory(directory: str, create_if_not_exists: bool = True) -> None:
	"""Clean directory contents or create if it doesn't exist."""
	shutil.rmtree(directory, ignore_errors=True)

	if create_if_not_exists:
		os.makedirs(directory, exist_ok=True)

def backup_file(source_path: str, backup_path: str) -> str:
	"""Backup a file to specified location."""
	backup_path = get_abs_path(backup_path)
	os.makedirs(os.path.dirname(backup_path), exist_ok=True)
	shutil.copy(source_path, backup_path)
	print(f"✅ File backed up to: {backup_path}")
	return backup_path

def douglas_peucker_simplify(points, epsilon):
	"""Simplify polygon using Douglas-Peucker algorithm"""
	polygon = Polygon(points)
	simplified = polygon.simplify(epsilon, preserve_topology=True)
	return list(simplified.exterior.coords[:-1])  # Remove duplicate last point

def filter_close_points(points, min_distance=5.0):
	"""Remove points that are closer than min_distance to previous point"""
	if len(points) < 2:
		return points
	
	filtered = [points[0]]
	
	for i in range(1, len(points)):
		current = np.array(points[i])
		previous = np.array(filtered[-1])
		distance = np.linalg.norm(current - previous)
		
		if distance >= min_distance:
			filtered.append(points[i])
	
	return filtered

def remove_thin_extensions_morphological(annotation_points, kernel_size=5):
	"""Remove thin extensions using morphological operations"""
	
	# Convert points to image mask
	points_array = np.array(annotation_points)
	min_x, min_y = np.min(points_array, axis=0).astype(int)
	max_x, max_y = np.max(points_array, axis=0).astype(int)
	
	# Create binary mask
	mask = np.zeros((max_y - min_y + 20, max_x - min_x + 20), dtype=np.uint8)
	
	# Adjust points to mask coordinates
	adjusted_points = points_array - [min_x - 10, min_y - 10]
	adjusted_points = adjusted_points.astype(np.int32)
	
	# Fill polygon
	cv2.fillPoly(mask, [adjusted_points], 255)
	
	# Morphological operations to remove thin extensions
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	
	# Erosion removes thin parts
	eroded = cv2.erode(mask, kernel, iterations=1)
	
	# Dilation restores the main body
	cleaned = cv2.dilate(eroded, kernel, iterations=1)
	
	# Extract contour from cleaned mask
	contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	if contours:
		# Get the largest contour
		largest_contour = max(contours, key=cv2.contourArea)
		
		# Convert back to original coordinate system
		cleaned_points = largest_contour.reshape(-1, 2) + [min_x - 10, min_y - 10]
		return cleaned_points.tolist()
	
	return annotation_points

def str_format(points_list):
	"""Convert points list to segmentation format string"""
	# Points should be a list of tuples/lists [(x1, y1), (x2, y2), ...]
	coords = []
	for point in points_list:
		coords.extend([point[0], point[1]])
	
	# Format as string with 6 decimal places
	coords_str = ' '.join(f'{coord:.6f}' for coord in coords)
	print(coords_str)
	return coords_str


def array_format(coords_str):
	"""Convert segmentation format string to points list"""
	# Parse coords_str to list of floats
	coords = list(map(float, coords_str.split()))
	
	# Convert to list of points [(x1, y1), (x2, y2), ...]
	points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
	print(points)
	return points

def normalize_segmentation(annotations, min_distance=8.0, epsilon=5.0, remove_extensions=True):
	"""Complete normalization pipeline for segmentation points"""
	processed_annotations = []

	for annotation in annotations:
		if annotation["type"] == "segmentation":
			original_points = [(p["x"], p["y"]) for p in annotation["points"]]
			# Step 1: Remove thin extensions first (if enabled)
			normalized_points = remove_thin_extensions_morphological(original_points, kernel_size=7)
			
			# Step 2: Filter out points too close together
			normalized_points = filter_close_points(normalized_points, min_distance)
			
			# Step 3: Apply Douglas-Peucker simplification
			normalized_points = douglas_peucker_simplify(normalized_points, epsilon)
			
			# Update annotation with normalized points
			annotation["points"] = [{"x": p[0], "y": p[1]} for p in normalized_points]
		
		processed_annotations.append(annotation)

	return processed_annotations