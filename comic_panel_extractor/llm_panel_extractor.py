from .config import Config
from ultralytics import YOLO 
from PIL import Image 
import cv2
from . import constant
from . import utils
import os
import shutil
import requests
from pathlib import Path

class LLMPanelExtractor:
	"""Handles image preprocessing operations."""
	
	def __init__(self, config: Config = None):
		self.config = config or Config()

		# Check if YOLO model exists; if not, download it to the specified path
		yolo_base_model_path = f'{self.config.yolo_base_model_path}_best.pt'
		yolo_base_model_path = f'{self.config.yolo_trained_model_path}'
		if not os.path.exists(yolo_base_model_path):
			url = "https://huggingface.co/mosesb/best-comic-panel-detection/resolve/main/best.pt"
			print(f"Downloading YOLO model to {yolo_base_model_path}...")
			response = requests.get(url)
			response.raise_for_status()  # Raise an error if the download fails
			with open(yolo_base_model_path, "wb") as f:
				f.write(response.content)
			print("YOLO model downloaded successfully.")

		self.yolo_model = YOLO(yolo_base_model_path)
		os.makedirs(self.config.output_folder, exist_ok=True)

	def extract_bounding_boxes(self, detection_result_boxes):
		"""Extract bounding box coordinates from YOLO detection results."""
		bounding_boxes = []
		for detection_box in detection_result_boxes.xyxy:
			# Extract coordinates
			x_min, y_min, x_max, y_max = map(int, detection_box)
			bounding_boxes.append((x_min, y_min, x_max, y_max))

		return bounding_boxes

	def crop_and_save_detected_panels(self, detected_boxes):
		"""Crop detected boxes and save them in separate folders"""
		if len(detected_boxes) == 0:
			print(f"No boxes detected for {self.config.org_input_path}")
			return

		source_image = cv2.imread(self.config.org_input_path)
		for box_coordinates in detected_boxes:
			# Extract coordinates
			x_min, y_min, x_max, y_max = box_coordinates
			
			# Crop the image
			cropped_panel = source_image[y_min:y_max, x_min:x_max]
			
			# Save cropped image
			constant.INDEX += 1
			panel_output_path = f"{self.config.output_folder}/{constant.INDEX:04d}_panel_{x_min, y_min, x_max, y_max}.jpg"
			cv2.imwrite(panel_output_path, cropped_panel)

	def pre_all_processed_boxes(self, all_processed_boxes, image_width, image_height):
		all_processed_boxes = utils.extend_boxes_to_image_border(
			all_processed_boxes, 
			(image_width, image_height), 
			self.config.min_width_ratio, 
			self.config.min_height_ratio
		)
		all_processed_boxes = sorted(all_processed_boxes, key=lambda box: (box[1], box[0]))  # sort by y_min, then x_min
		all_processed_boxes = utils.extend_to_nearby_boxes(
			all_processed_boxes, 
			(image_width, image_height), 
			self.config.min_width_ratio, 
			self.config.min_height_ratio
		)
		return all_processed_boxes

	def detect_and_extract_panels(self, input_image_path=None, existing_boxes=None, confidence_threshold=0.9):
		"""Main method to detect and extract panels from an image."""
		if not input_image_path:
			input_image_path = self.config.org_input_path

		# Get image dimensions
		with Image.open(input_image_path) as input_image:
			image_width, image_height = input_image.size

		# Run YOLO detection
		detection_results = self.yolo_model.predict(source=input_image_path)
		first_detection_result = detection_results[0]
		newly_detected_boxes = None
		all_processed_boxes = []
		
		# Add existing boxes if provided
		if existing_boxes:
			all_processed_boxes.extend(existing_boxes)
		
		# Filter boxes by confidence threshold
		if first_detection_result.boxes is not None:
			high_confidence_filter = first_detection_result.boxes.conf >= confidence_threshold
			if high_confidence_filter.sum() > 0:
				first_detection_result.boxes = first_detection_result.boxes[high_confidence_filter]
				newly_detected_boxes = self.extract_bounding_boxes(first_detection_result.boxes)
				newly_detected_boxes = utils.is_valid_panel((image_width, image_height), newly_detected_boxes, self.config.min_width_ratio, self.config.min_height_ratio)
				if newly_detected_boxes:
					all_processed_boxes.extend(self.extract_bounding_boxes(first_detection_result.boxes))

					# Process and extend boxes
					all_processed_boxes = self.pre_all_processed_boxes(all_processed_boxes, image_width, image_height)

					# Crop and save detected panels
					self.crop_and_save_detected_panels(newly_detected_boxes)
					
					# Save prediction visualization
					visualization_result = first_detection_result.plot(masks=False)
					constant.INDEX += 1
					debug_output_path = f"{self.config.output_folder}/{constant.INDEX:04d}_debug.jpg"
					Image.fromarray(visualization_result[..., ::-1]).save(debug_output_path)
					
					# Create black and white mask
					constant.INDEX += 1
					masked_output_path = f"{self.config.output_folder}/{constant.INDEX:04d}_draw_black.jpg"
					masked_image_path = utils.draw_black(self.config.org_input_path, all_processed_boxes, masked_output_path, stripe=False)
					return masked_image_path, newly_detected_boxes

		# Process boxes even if no new detections
		all_processed_boxes = self.pre_all_processed_boxes(all_processed_boxes, image_width, image_height)

		constant.INDEX += 1
		masked_output_path = f"{self.config.output_folder}/{constant.INDEX:04d}_draw_black.jpg"
		masked_image_path = utils.draw_black(self.config.org_input_path, all_processed_boxes, masked_output_path, stripe=False)
		return masked_image_path, newly_detected_boxes

	def check_for_remaining_similarity(self, current_processed_image_path, existing_boxes):
		# Get image dimensions
		with Image.open(self.config.org_input_path) as input_image:
			image_width, image_height = input_image.size

		all_processed_boxes = self.pre_all_processed_boxes(existing_boxes, image_width, image_height)

		constant.INDEX += 1
		similar_remaining_regions_path = f"{self.config.output_folder}/{constant.INDEX:04d}_remaining_similarity_debug.jpg"
		similar_remaining_box = utils.find_similar_remaining_regions(all_processed_boxes, (image_width, image_height), similar_remaining_regions_path)
		if similar_remaining_box:
			similar_remaining_box = utils.is_valid_panel((image_width, image_height), similar_remaining_box, self.config.min_width_ratio, self.config.min_height_ratio)
			if similar_remaining_box:
				self.crop_and_save_detected_panels(similar_remaining_box)
				existing_boxes.extend(similar_remaining_box)

				all_processed_boxes = self.pre_all_processed_boxes(existing_boxes, image_width, image_height)

				constant.INDEX += 1
				current_processed_image_path = f"{self.config.output_folder}/{constant.INDEX:04d}_remaining_similarity_draw_black.jpg"
				current_processed_image_path = utils.draw_black(self.config.org_input_path, all_processed_boxes, current_processed_image_path, stripe=False)

		return current_processed_image_path, existing_boxes

def extract_panel_via_llm(input_image_path, config=None, reset=True):
	"""Main function to extract panels using various image processing techniques."""
	# Initialize configuration
	extractor_config = config or Config()
	extractor_config.org_input_path = input_image_path

	# Clean output folder
	if reset:
		if Path(extractor_config.output_folder).exists():
			shutil.rmtree(extractor_config.output_folder, ignore_errors=True)
		Path(extractor_config.output_folder).mkdir(exist_ok=True)

	# Initialize extractor
	panel_extractor = LLMPanelExtractor(extractor_config)

	current_processed_image_path = extractor_config.org_input_path
	accumulated_detected_boxes = []
	all_processed_boxes = []
	
	# Get original image dimensions
	with Image.open(current_processed_image_path) as original_image:
		original_width, original_height = original_image.size

	# Define image processing techniques to try
	processing_techniques = [
		{
			'name': 'clahe',
			'function': utils.convert_to_clahe,
			'confidence_level': 1.0,
			'description': 'CLAHE (Contrast Limited Adaptive Histogram Equalization)'
		},
		{
			'name': 'grayscale',
			'function': utils.convert_to_grayscale_pil,
			'confidence_level': 1.0,
			'description': 'Grayscale conversion'
		},
		{
			'name': 'lab_l',
			'function': utils.convert_to_lab_l,
			'confidence_level': 1.0,
			'description': 'LAB L-channel extraction'
		},
		{
			'name': 'group_color',
			'function': utils.convert_to_group_colors,
			'confidence_level': 0.1,
			'image_path': extractor_config.org_input_path,
			'description': 'Group Color extraction'
		}
	]

	# Process with different techniques until white ratio threshold is met
	for technique in processing_techniques:
		iteration_count = 0
		confidence_level = technique["confidence_level"]
		if technique.get("image_path", None) and utils.box_covered_ratio(panel_extractor.pre_all_processed_boxes(accumulated_detected_boxes, original_width, original_height), (original_width, original_height)) < 0.95:
			current_processed_image_path = technique.get("image_path")
		
		while (utils.box_covered_ratio(panel_extractor.pre_all_processed_boxes(accumulated_detected_boxes, original_width, original_height), (original_width, original_height)) < 0.95 and confidence_level > 0):
			
			print(f"\n{technique['description']} process iteration: {iteration_count} confidence level: {confidence_level}")
			iteration_count += 1
			confidence_level -= 0.1
			
			# Apply image processing technique
			constant.INDEX += 1
			processed_output_path = f"{extractor_config.output_folder}/{constant.INDEX:04d}_convert_to_{technique['name']}.jpg"
			current_processed_image_path = technique['function'](current_processed_image_path, processed_output_path)
			
			# Run panel detection on processed image
			current_processed_image_path, newly_detected_boxes = panel_extractor.detect_and_extract_panels(
				input_image_path=current_processed_image_path, 
				existing_boxes=accumulated_detected_boxes, 
				confidence_threshold=confidence_level
			)
			if newly_detected_boxes:
				accumulated_detected_boxes.extend(newly_detected_boxes)

			current_processed_image_path, accumulated_detected_boxes = panel_extractor.check_for_remaining_similarity(current_processed_image_path, accumulated_detected_boxes)
			all_processed_boxes = panel_extractor.pre_all_processed_boxes(accumulated_detected_boxes, original_width, original_height)

	remain_boxes = utils.get_remaining_areas((original_width, original_height), all_processed_boxes)
	if remain_boxes:
		remain_boxes = utils.is_valid_panel((original_width, original_height), remain_boxes, extractor_config.min_width_ratio, extractor_config.min_height_ratio)
		if remain_boxes:
			panel_extractor.crop_and_save_detected_panels(remain_boxes)
			all_processed_boxes.extend(remain_boxes)
			accumulated_detected_boxes.extend(remain_boxes)

	all_path = [file for file in os.listdir(extractor_config.output_folder) if "_panel_" in file]
	print(f"Processing complete. Final result saved to: {extractor_config.output_folder}")
	print(f"Total panels detected: {len(all_path)}")
	return all_path, accumulated_detected_boxes, all_processed_boxes


if __name__ == "__main__":
	import argparse

	# Parse command-line arguments
	argument_parser = argparse.ArgumentParser(description="Run panel extractor on an image")
	argument_parser.add_argument("--input", type=str, required=True, help="Path to input image")
	parsed_arguments = argument_parser.parse_args()
	
	final_result_path, total_detected_boxes = extract_panel_via_llm(parsed_arguments.input)