from typing import List, Tuple
from pathlib import Path
from .config import Config

import numpy as np
import cv2

class ImageProcessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def mask_text_regions(self, bboxes: List[List[int]], output_filename: str = "1_text_removed.jpg", color: Tuple[int, int, int] = (0, 0, 0)) -> str:
        """Mask text regions in the image to reduce panel extraction noise."""
        image = cv2.imread(self.config.input_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {self.config.input_path}")

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)

        output_path = f'{self.config.output_folder}/{output_filename}'
        cv2.imwrite(output_path, image)
        print(f"✅ Text-masked image saved to: {output_path}")
        return str(output_path)
    
    def preprocess_image(self, masked_image_path) -> Tuple[str, str, str]:
        """Preprocess image for panel extraction."""
        image = cv2.imread(masked_image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {masked_image_path}")

        # Convert to grayscale and binary
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Dilate to strengthen borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Save intermediate results
        gray_path = f'{self.config.output_folder}/2_gray.jpg'
        binary_path = f'{self.config.output_folder}/3_binary.jpg'
        dilated_path = f'{self.config.output_folder}/4_dilated.jpg'
        
        cv2.imwrite(str(gray_path), gray)
        cv2.imwrite(str(binary_path), binary)
        cv2.imwrite(str(dilated_path), dilated)
        
        return str(gray_path), str(binary_path), str(dilated_path)
    
    def clean_dilated_image(self, dilated_path: str, 
                           output_filename: str = "5_dilated_cleaned.jpg",
                           max_neighbors: int = 2) -> str:
        """Clean dilated image by thinning thick borders."""
        dilated = cv2.imread(dilated_path, cv2.IMREAD_GRAYSCALE)
        if dilated is None:
            raise FileNotFoundError(f"Could not load dilated image: {dilated_path}")

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
        output_path = f'{self.config.output_folder}/{output_filename}'
        cv2.imwrite(str(output_path), cleaned_img)
        print(f"✅ Cleaned dilated image saved to: {output_path}")
        return str(output_path)