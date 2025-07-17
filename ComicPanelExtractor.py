import numpy as np
import os
import json
from text_detector import TextDetector, Config as CVP_Config
import cv2
import shutil

# ----------------------------------------------------------
# MASK TEXT REGIONS
# ----------------------------------------------------------

def mask_text_regions(image_path, bboxes, output_path=None, color=(0, 0, 0)):
	"""
	Make the text regions in an image white (or given color) to reduce panel extraction noise.

	Args:
		image_path (str): Path to the input image.
		bboxes (list of list): List of bounding boxes in [x1, y1, x2, y2] format.
		output_path (str, optional): Path to save the modified image.
		color (tuple): Color to fill the bounding boxes (default black).
	Returns:
		masked_image (numpy array): Image with masked text regions.
	"""
	image = cv2.imread(image_path)
	if image is None:
		raise Exception(f"Could not load image: {image_path}")

	for bbox in bboxes:
		x1, y1, x2, y2 = bbox
		cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)  # Fill rectangle

	if output_path:
		cv2.imwrite(output_path, image)
		print(f"✅ Text-masked image saved to: {output_path}")

	return image


# ----------------------------------------------------------
# PRE PROCESS METHOD
# ----------------------------------------------------------

def pre_process(image_path, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Load and preprocess image
	image = cv2.imread(image_path)
	if image is None:
		raise Exception(f"Could not load image: {image_path}")

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

	# Dilate to strengthen borders and fill small gaps
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	dilated = cv2.dilate(binary, kernel, iterations=2)

	cv2.imwrite(os.path.join(output_dir, "2_gray.jpg"), gray)
	cv2.imwrite(os.path.join(output_dir, "3_binary.jpg"), binary)
	cv2.imwrite(os.path.join(output_dir, "4_dilated.jpg"), dilated)


# ----------------------------------------------------------
# CLEAN DILATED IMAGE
# ----------------------------------------------------------

def clean_dilated_with_row_priority(dilated_path, output_path, max_neighbors=2):
	"""
	Clean a dilated comic page by thinning thick borders using Game-of-Life logic,
	with preference to clean rows that have fewer black pixels.
	"""
	dilated = cv2.imread(dilated_path, cv2.IMREAD_GRAYSCALE)
	if dilated is None:
		raise Exception("Could not load dilated image.")

	binary = (dilated == 0).astype(np.uint8)
	padded = np.pad(binary, pad_width=1, mode="constant", constant_values=0)
	cleaned = binary.copy()

	height, width = binary.shape
	row_black_counts = np.sum(binary, axis=1)

	for y in range(1, height + 1):
		for x in range(1, width + 1):
			if padded[y, x] == 1:
				neighbors = np.sum(padded[y-1:y+2, x-1:x+2]) - 1
				if neighbors > max_neighbors:
					neighbor_rows = [r for r in [y-1, y, y+1] if 1 <= r <= height]
					if neighbor_rows:
						row_to_clear = min(neighbor_rows, key=lambda r: row_black_counts[r-1])
						if y == row_to_clear:
							cleaned[y-1, x-1] = 0

	cleaned_img = (1 - cleaned) * 255
	cv2.imwrite(output_path, cleaned_img)
	print(f"✅ Cleaned dilated image saved to: {output_path}")
	return output_path


# ----------------------------------------------------------
# EXTRACT PANELS - BLACK PERCENTAGE METHOD
# ----------------------------------------------------------

def extract_panels_by_black_percentage_fixed(
	dilated_path, original_image_path, output_dir,
	row_thresh=20, col_thresh=20,
	min_width_ratio=0.1, min_height_ratio=0.1
):
	"""
	Extract comic panels using black percentage scan with smart width & height filtering.
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	dilated = cv2.imread(dilated_path, cv2.IMREAD_GRAYSCALE)
	original = cv2.imread(original_image_path)
	if dilated is None or original is None:
		raise Exception("Could not load dilated or original image.")

	height, width = dilated.shape
	visual_output = original.copy()

	# Detect row gutters
	row_black_percentage = np.sum(dilated == 0, axis=1) / width * 100
	row_gutters, panel_rows = [], []
	in_gutter = False
	for y, percent_black in enumerate(row_black_percentage):
		if percent_black >= row_thresh and not in_gutter:
			start_row = y
			in_gutter = True
		elif percent_black < row_thresh and in_gutter:
			end_row = y
			row_gutters.append((start_row, end_row))
			in_gutter = False

	prev_end = 0
	for start, end in row_gutters:
		if start - prev_end > 10:
			panel_rows.append((prev_end, start))
		prev_end = end
	if height - prev_end > 10:
		panel_rows.append((prev_end, height))

	# Extract panels
	all_panels, panel_count, panel_images, panel_points = [], 0, [], []
	for y1, y2 in panel_rows:
		row_slice = dilated[y1:y2, :]
		col_black_percentage = np.sum(row_slice == 0, axis=0) / (y2 - y1) * 100
		col_gutters, panel_cols = [], []
		in_gutter_col = False
		for x, percent_black in enumerate(col_black_percentage):
			if percent_black >= col_thresh and not in_gutter_col:
				start_col = x
				in_gutter_col = True
			elif percent_black < col_thresh and in_gutter_col:
				end_col = x
				col_gutters.append((start_col, end_col))
				in_gutter_col = False

		prev_end_col = 0
		for start, end in col_gutters:
			if start - prev_end_col > 10:
				panel_cols.append((prev_end_col, start))
			prev_end_col = end
		if width - prev_end_col > 10:
			panel_cols.append((prev_end_col, width))

		for x1, x2 in panel_cols:
			w, h = x2 - x1, y2 - y1
			if w * h < (width * height) * 0.005:
				continue
			all_panels.append((x1, y1, x2, y2))

	# Post-filter
	panel_widths = [x2 - x1 for x1, _, x2, _ in all_panels]
	panel_heights = [y2 - y1 for _, y1, _, y2 in all_panels]
	avg_width = np.mean(panel_widths) if panel_widths else 0
	avg_height = np.mean(panel_heights) if panel_heights else 0
	min_allowed_width = max(avg_width * 0.5, width * min_width_ratio)
	min_allowed_height = max(avg_height * 0.5, height * min_height_ratio)

	for x1, y1, x2, y2 in all_panels:
		panel_width, panel_height = x2 - x1, y2 - y1
		if panel_width >= min_allowed_width and panel_height >= min_allowed_height:
			panel = original[y1:y2, x1:x2]
			panel_count += 1
			panel_images.append(panel)
			panel_points.append({
				"x_start": x1, "y_start": y1, "x_end": x2, "y_end": y2
			})
			panel_path = os.path.join(output_dir, f"panel_{panel_count}.jpg")
			cv2.imwrite(panel_path, panel)
			cv2.rectangle(visual_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(visual_output, f"#{panel_count}", (x1+5, y1+25),
						cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

	print(f"✅ Extracted {panel_count} panels after smart width & height filtering.")
	return output_dir, panel_images, panel_points


# ----------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------
if __name__ == "__main__":
	image_path = "input.jpg"
	output_dir = "extracted_panels"
	shutil.rmtree(output_dir, ignore_errors=True)
	os.makedirs(output_dir, exist_ok=True)

	# Detect and mask text regions
	cvp_config = CVP_Config()
	cvp_config.main_file_name = image_path
	cvp_config.temp_folder = output_dir
	cvp_config.comic_image = image_path
	cvp_config.output_video = f"{output_dir}/test.mp4"

	with TextDetector(cvp_config) as text_detector:
		bubbles_path = text_detector.detect_and_group_text(cvp_config.comic_image)
	with open(bubbles_path, "r", encoding="utf-8") as f:
		bubbles = json.load(f)

	output_path = os.path.join(output_dir, "1_text_removed.jpg")
	masked_image = mask_text_regions(image_path, [box["bbox"] for box in bubbles], output_path=output_path)

	pre_process(output_path, output_dir)

	# Clean dilated image
	dilated_path = os.path.join(output_dir, "4_dilated.jpg")
	cleaned_dilated_path = os.path.join(output_dir, "5_dilated_cleaned.jpg")
	clean_dilated_with_row_priority(dilated_path, cleaned_dilated_path, max_neighbors=2)

	# Extract panels - black percentage
	extract_panels_by_black_percentage_fixed(
		cleaned_dilated_path,
		image_path,
		output_dir,
		min_width_ratio=0.1,  # Panels must be at least 10% of total width
	)
