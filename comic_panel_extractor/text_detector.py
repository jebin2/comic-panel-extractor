
import json
import os
from typing import List, Optional
from dataclasses import dataclass
import numpy as np

from .config import Config

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
        self.reader = None

    def load(self):
        """Load the OCR reader."""
        if self.reader is None:
            import easyocr
            self.reader = easyocr.Reader(['en'])
    
    def detect_text(self) -> List[TextDetection]:
        """Detect text regions in the image."""
        self.load()
        results = self.reader.readtext(self.config.input_path)
        print(f"EasyOCR found {len(results)} raw detections")
        
        detections = []
        for box, text, confidence in results:
            bbox = self._normalize_bbox(box)
            detections.append(TextDetection(
                bbox=bbox,
                text=text.strip(),
                confidence=float(confidence)
            ))
        
        return detections
    
    def _normalize_bbox(self, box: List[List[int]]) -> List[int]:
        """Convert box coordinates to normalized bbox format."""
        return [
            min(x[0] for x in box),
            min(x[1] for x in box),
            max(x[0] for x in box),
            max(x[1] for x in box)
        ]
    
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
            merged = False
            
            for group in groups:
                if self.calculate_distance(detection.bbox, group.bbox) < self.config.distance_threshold:
                    self._merge_detections(group, detection)
                    merged = True
                    break
            
            if not merged:
                groups.append(detection)
        
        # Sort groups by vertical position and assign IDs
        groups.sort(key=lambda g: g.bbox[1])
        for idx, group in enumerate(groups):
            group.id = idx + 1
        
        return groups
    
    def _merge_detections(self, group: TextDetection, detection: TextDetection):
        """Merge two text detections."""
        group.text += " " + detection.text
        group.bbox = [
            min(group.bbox[0], detection.bbox[0]),
            min(group.bbox[1], detection.bbox[1]),
            max(group.bbox[2], detection.bbox[2]),
            max(group.bbox[3], detection.bbox[3])
        ]
    
    def detect_and_group_text(self) -> str:
        """Main method to detect and group text, saving results to JSON."""
        if not os.path.exists(self.config.text_cood_path):
            detections = self.detect_text()
            groups = self.group_text_regions(detections)
            self._save_groups_to_json(groups, self.config.text_cood_path)
            print(f"Grouped bubbles saved: {self.config.text_cood_path}")
        
        return self.config.text_cood_path
    
    def _save_groups_to_json(self, groups: List[TextDetection], output_path: str):
        """Save grouped text detections to JSON file."""
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

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.reader:
                del self.reader
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()