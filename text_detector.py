import json
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os

import numpy as np
from moviepy.editor import *

@dataclass
class Config:
	"""Configuration settings for the comic-to-video pipeline."""
	main_file_name: str = ""
	comic_image: str = ""
	temp_folder: str = ""
	distance_threshold: int = 70
	vertical_threshold: int = 30
	tts_engine: str = "chatterbox"
	resolution: Tuple[int, int] = (1920, 1080)
	margin_ratio: float = 0.08
	auto_scroll: bool = True
	zoom_enabled: bool = False
	zoom_factor: float = 1.1
	output_video: str = "comic_text.mp4"
	min_text_length: int = 2


@dataclass
class TextDetection:
	"""Represents a detected text region."""
	bbox: List[int]
	text: str
	confidence: float
	id: Optional[int] = None

class TextDetector:
	"""Handles text detection and grouping from comic images."""
	
	def __init__(self, config: Config):
		self.config = config

	def load(self):
		import easyocr
		self.reader = easyocr.Reader(['en'])
	
	def detect_text(self, image_path: str) -> List[TextDetection]:
		"""Detect text regions in the image."""
		self.load()
		results = self.reader.readtext(image_path)
		print(f"EasyOCR found {len(results)} raw detections")
		
		detections = []
		for box, text, confidence in results:
			bbox = [
				min(x[0] for x in box),
				min(x[1] for x in box),
				max(x[0] for x in box),
				max(x[1] for x in box)
			]
			detections.append(TextDetection(
				bbox=bbox,
				text=text.strip(),
				confidence=float(confidence)
			))
		
		return detections
	
	@staticmethod
	def calculate_distance(bbox1: List[int], bbox2: List[int]) -> float:
		"""Calculate Euclidean distance between two bounding box centers."""
		center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
		center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
		return np.linalg.norm(np.subtract(center1, center2))
	
	def group_text_regions(self, detections: List[TextDetection]) -> List[TextDetection]:
		"""Group nearby text regions into speech bubbles."""
		# Filter out single character detections
		filtered_detections = [
			det for det in detections 
			if len(det.text.strip()) >= self.config.min_text_length
		]
		
		# Sort by vertical position (top to bottom)
		filtered_detections.sort(key=lambda d: d.bbox[1])
		
		groups = []
		for detection in filtered_detections:
			added_to_group = False
			
			for group in groups:
				if self.calculate_distance(detection.bbox, group.bbox) < self.config.distance_threshold:
					# Merge with existing group
					group.text += " " + detection.text
					group.bbox = [
						min(group.bbox[0], detection.bbox[0]),
						min(group.bbox[1], detection.bbox[1]),
						max(group.bbox[2], detection.bbox[2]),
						max(group.bbox[3], detection.bbox[3])
					]
					added_to_group = True
					break
			
			if not added_to_group:
				groups.append(detection)
		
		# Sort groups by vertical position and assign IDs
		groups.sort(key=lambda g: g.bbox[1])
		for idx, group in enumerate(groups):
			group.id = idx + 1
		
		return groups
	
	def detect_and_group_text(self, image_path: str) -> str:
		"""Main method to detect and group text, saving results to JSON."""
		
		# Save to JSON
		output_path = self.config.output_video.replace(".mp4", "_detect_and_group_text.json")
		if not os.path.exists(output_path):
			detections = self.detect_text(image_path)
			groups = self.group_text_regions(detections)
			groups_data = []
			for group in groups:
				groups_data.append({
					"id": group.id,
					"bbox": [int(x) for x in group.bbox],
					"text": group.text,
					"confidence": group.confidence
				})
			
			with open(output_path, "w", encoding="utf-8") as f:
				json.dump(groups_data, f, indent=2, ensure_ascii=False)
			
			print(f"Grouped bubbles saved: {output_path}")
		return str(output_path)

	def cleanup(self):
		try:
			del self.reader

		except: pass

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.cleanup()

	def __del__(self):
		self.cleanup()