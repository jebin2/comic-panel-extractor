from typing import List, Tuple
from pathlib import Path
from .config import Config

import numpy as np
import cv2

class ImageProcessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def mask_text_regions(self, input_path, bboxes: List[List[int]], output_filename: str = "1_text_removed.jpg", color: Tuple[int, int, int] = (0, 0, 0)) -> str:
        """Mask text regions in the image to reduce panel extraction noise."""
        image = cv2.imread(input_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {input_path}")

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)

        output_path = f'{self.config.output_folder}/{output_filename}'
        cv2.imwrite(output_path, image)
        print(f"✅ Text-masked image saved to: {output_path}")
        return str(output_path)
    
    def preprocess_image(self, processed_image_path) -> Tuple[str, str, str]:
        """Preprocess image for panel extraction."""
        image = cv2.imread(processed_image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {processed_image_path}")

        # Convert to grayscale and binary
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        is_inverted = False
        # binary, is_inverted = self.invert_if_black_dominates(binary)

        if not is_inverted:
            # Dilate to strengthen borders
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)
        else: dilated = edges

        # Save intermediate results
        gray_path = f'{self.config.output_folder}/2_gray.jpg'
        binary_path = f'{self.config.output_folder}/3_binary.jpg'
        dilated_path = f'{self.config.output_folder}/4_dilated.jpg'
        
        cv2.imwrite(str(gray_path), gray)
        cv2.imwrite(str(binary_path), edges)
        cv2.imwrite(str(dilated_path), dilated)
        
        return str(gray_path), str(binary_path), str(dilated_path), is_inverted

    def invert_if_black_dominates(self, binary):
        # Threshold to binary image
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        # Count black and white pixels
        black_pixels = np.sum(binary == 0)
        white_pixels = np.sum(binary == 255)

        # If black dominates, invert
        if black_pixels > white_pixels:
            print("🔄 Inverting image because black > white")
            inverted = cv2.bitwise_not(binary)
        else:
            print("✅ No inversion needed, white >= black")
            inverted = binary

        # Save result
        return inverted, black_pixels > white_pixels

    def remove_inner_sketch(self, input_path, output_filename="5_remove_inner_sketch.jpg"):
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape

        # Threshold image to binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Find all contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create mask for large contours (likely panel borders)
        mask = np.zeros_like(binary)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= (height * width * self.config.min_area_ratio):
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Apply mask to original image (keeps only large borders)
        cleaned = cv2.bitwise_and(binary, binary, mask=mask)

        # Optional: Apply morphological opening to clean tiny sketch lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Invert back if needed
        cleaned = cv2.bitwise_not(cleaned)

        # Save
        output_path = f'{self.config.output_folder}/{output_filename}'
        cv2.imwrite(output_path, cleaned)
        print(f"✅ Remove Inner Sketch image saved to: {output_path}")
        return str(output_path)

    def thin_image_borders(self, processed_image_path: str, output_filename: str = "6_thin_border.jpg") -> str:
        """
        Clean dilated image by thinning thick borders and removing hanging clusters.
        """
        from skimage.morphology import skeletonize, remove_small_objects
        from skimage.measure import label

        # Load image
        img = cv2.imread(processed_image_path)
        # Convert to grayscale and binary
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)

        # Skeletonize
        skeleton = skeletonize(edges).astype(np.uint8)

        # Remove small hanging clusters
        labeled = label(skeleton, connectivity=2)
        cleaned = remove_small_objects(labeled, min_size=150)  # Adjust min_size for more/less pruning

        # Convert back to 0–255 uint8 image
        final = (cleaned > 0).astype(np.uint8) * 255

        # Invert back if needed
        result = 255 - final

        # Save
        output_path = f'{self.config.output_folder}/{output_filename}'
        cv2.imwrite(output_path, result)
        print(f"✅ Cleaned and thinned image saved to: {output_path}")
        return str(output_path)

    
    def clean_dilated_image(self, dilated_path: str, 
                           output_filename: str = "6_dilated_cleaned.jpg",
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