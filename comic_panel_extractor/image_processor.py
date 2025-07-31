from typing import List, Tuple
from pathlib import Path
from .config import Config

import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from skimage import measure
from tqdm import tqdm

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import math

class ImageProcessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.index = 0

    def get_output_path(self, output_folder, file_name):
        self.index += 1
        return f'{output_folder}/{self.index:02d}_{file_name}'

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
        return str(output_path)
    
    def preprocess_image(self, processed_image_path) -> Tuple[str, str, str]:
        """Preprocess image for panel extraction."""
        image = cv2.imread(processed_image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {processed_image_path}")

        # Convert to grayscale and binary
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Save intermediate results
        gray_path = self.get_output_path(self.config.output_folder, "gray.jpg")
        binary_path = self.get_output_path(self.config.output_folder, "binary.jpg")
        dilated_path = self.get_output_path(self.config.output_folder, "dilated.jpg")
        
        cv2.imwrite(str(gray_path), gray)
        cv2.imwrite(str(binary_path), edges)
        cv2.imwrite(str(dilated_path), dilated)
        
        return str(gray_path), str(binary_path), str(dilated_path)

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

    def group_colors(self, processed_image_path, num_clusters: int = 5, file_name="group_colors.jpg", output_folder=None) -> Image.Image:
        """
        Groups similar colors in an image using KMeans clustering.

        Args:
            processed_image_path (str): Path to the image to be color-grouped.
            num_clusters (int): Number of color clusters to form.
            file_name (str): Name of the output image file.
            output_folder (str): Optional output directory.

        Returns:
            str: Path to the saved grouped-color image.
        """
        output_folder = output_folder or self.config.output_folder
        # Load image
        image = Image.open(processed_image_path).convert("RGB")
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
        output_path = self.get_output_path(output_folder, file_name)
        clustered_bgr = clustered_pixels[:, :, ::-1]
        cv2.imwrite(output_path, clustered_bgr)

        return str(output_path)

    def thin_image_borders(self, processed_image_path: str, file_name="thin_border.jpg", output_folder=None) -> str:
        """
        Clean dilated image by thinning thick borders and removing hanging clusters.
        """
        output_folder = output_folder or self.config.output_folder
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
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, result)
        return str(output_path)

    def remove_dangling_lines(self, image_path, file_name="dangling_lines_removed.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Threshold to binary mask (black lines = True, white = False)
        binary = gray < 128  # black parts (lines/dangling strokes)
        binary = binary.astype(bool)

        # Label connected components
        labeled = label(binary, connectivity=2)

        # Remove small connected components (dangling lines, fragments)
        cleaned = remove_small_objects(labeled, min_size=500)  # Adjust min_size as needed

        # Convert back to mask (255 = black lines kept, 255 background = white)
        final_mask = (cleaned > 0).astype(np.uint8) * 255

        # Invert mask to match original layout: black lines on white background
        final_image = 255 - final_mask
        # Save result
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, final_image)
        return output_path

    def remove_diagonal_lines(self, image_path, file_name="remove_diagonal_lines.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder
        
        # Read the image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create binary image (black lines on white background)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Create kernels for detecting horizontal and vertical lines
        # Adjust kernel size based on your image - larger for thicker lines
        kernel_length = max(gray.shape[0], gray.shape[1]) // 30
        
        # Horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        # Vertical kernel  
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine horizontal and vertical lines
        rect_lines = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
        
        # Create final result - white background with black rectangular lines only
        result = np.ones_like(gray) * 255  # White background
        result[rect_lines > 0] = 0  # Black lines where rectangular lines were detected

        # Save result
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, result)
        return output_path

    def thick_black(self, image_path, thickness=20, file_name="thick_black.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder
        # Load image
        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create a binary mask where black pixels are 1 (foreground)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

        # Define kernel size based on desired thickness
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))

        # Dilate the black areas
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Invert back so black is 0 again
        # result_mask = cv2.bitwise_not(dilated)

        # Apply mask on original image
        result = img.copy()
        result[np.where(dilated == 255)] = (0, 0, 0)

        # Save result
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, result)
        return output_path

    def to_int_box(self, line):
        return map(int, line[0])  # Works for both Hough and LSD formats

    def remove_diagonal_lines_and_set_white(self, image_path, file_name="remove_diagonal_lines_and_set_white.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder
        # Load image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Dilate to connect broken segments
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # More sensitive Hough transform
        # HoughLinesP_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=5, maxLineGap=10)

        # Detect lines using Hough Transform
        lsd = cv2.createLineSegmentDetector(0)
        lines, _, _, _ = lsd.detect(gray)

        # Copy image to edit
        output = image.copy()

        combined_lines = []

        if lines is not None:
            combined_lines.extend(lines)

        # if HoughLinesP_lines is not None:
        #     combined_lines.extend(HoughLinesP_lines)

        if combined_lines is not None:
            for line in combined_lines:
                x1, y1, x2, y2 = self.to_int_box(line)  # Convert float to int

                # Calculate angle
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)

                # Filter out horizontal and vertical lines
                if (80 < angle < 100) or (170 < angle < 190) or angle < 10 or angle > 350:
                    continue
                else:
                    # Get bounding box with padding
                    padding = 2
                    xmin = min(x1, x2) - padding
                    xmax = max(x1, x2) + padding
                    ymin = min(y1, y2) - padding
                    ymax = max(y1, y2) + padding

                    # Draw white rectangle (erase diagonal line)
                    cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=-1)

        # Save cleaned image
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, output)
        return output_path

    def remove_small_regions(self, image_path, file_name="remove_small_regions.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder

        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        visual = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # For debugging with colored rectangles

        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        height_, width_ = img.shape
        min_area = height_ * width_ * self.config.min_area_ratio

        # Threshold: make black = foreground
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Label connected regions
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        # Create clean mask (copy of original binary)
        clean_mask = np.copy(binary)

        for region in regions:
            area = region.area
            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr

            # Bounding box filter
            if width < width_ * self.config.min_width_ratio and height < height_ * self.config.min_height_ratio:
                if (width/width_) < 0.9 and (height/height_) < 0.9:
                    clean_mask[labeled == region.label] = 0  # Remove small region
                    cv2.rectangle(visual, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
                    continue

            # Crop and analyze region for line orientation
            region_crop = binary[minr:maxr, minc:maxc]
            edges = cv2.Canny(region_crop, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=5)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                    # length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    line_width = abs(x2 - x1)
                    line_height = abs(y2 - y1)

                    if line_height < height_ * self.config.min_height_ratio and line_width < width_ * self.config.min_width_ratio:
                        break
                else:
                    # Only runs if no 'break' occurred
                    # If no qualifying line found, remove region
                    clean_mask[labeled == region.label] = 0
                    cv2.rectangle(visual, (minc, minr), (maxc, maxr), (0, 255, 255), 2)
            elif width < width_ * self.config.min_width_ratio and height < height_ * self.config.min_height_ratio:
                # No lines, remove region
                clean_mask[labeled == region.label] = 0
                cv2.rectangle(visual, (minc, minr), (maxc, maxr), (255, 0, 0), 2)

        # Save debug visualization
        output_path = self.get_output_path(output_folder, f"debug_{file_name}")
        cv2.imwrite(output_path, visual)

        # Invert back to original format: black lines on white
        cleaned = cv2.bitwise_not(clean_mask)
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, cleaned)
        return output_path


    def thin_black(self, image_path, file_name="thin_black.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder
        # Load the image (replace 'debug_dilated.jpg' with your actual file path if needed)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image loaded correctly
        if img is None:
            raise ValueError("Image not loaded. Check the file path.")

        # Threshold to binary (invert if lines are black on white)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

        # Perform thinning to reduce to 1-pixel lines
        try:
            # Use Zhang-Suen thinning if opencv-contrib is installed
            thinned = cv2.ximgproc.thinning(binary)
        except AttributeError:
            # Fallback: Morphological skeletonization
            skel = np.zeros(binary.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                eroded = cv2.erode(binary, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(binary, temp)
                skel = cv2.bitwise_or(skel, temp)
                binary = eroded.copy()
                if cv2.countNonZero(binary) == 0:
                    break
            thinned = skel

        # Invert back if needed (for white lines on black background)
        thinned = 255 - thinned

        # Save result
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, thinned)
        return output_path

    def thin_lines_direct(self, image_path, file_name="thin_lines_direct.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder
        
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not load image")
        
        # Convert to binary (0 = black lines, 255 = white background)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Create result image (start with white background)
        result = np.full_like(binary, 255)  # All white
        
        height, width = binary.shape
        print("Processing thick lines...")
        
        # Method 1: Scan rows - for each thick horizontal segment, keep only bottom pixel
        print("Step 1: Thinning horizontal segments...")
        for row in range(height):
            col = 0
            while col < width:
                # If we hit a black pixel
                if binary[row, col] == 0:  # Black pixel
                    # Find the end of this horizontal segment
                    start_col = col
                    while col < width and binary[row, col] == 0:
                        col += 1
                    end_col = col - 1
                    
                    # For this horizontal segment, check if it's part of a thick vertical region
                    segment_width = end_col - start_col + 1
                    
                    if segment_width >= 1:  # Any horizontal segment
                        # Check how thick this region is vertically at the middle
                        mid_col = (start_col + end_col) // 2
                        
                        # Find vertical thickness at this point
                        thickness = self.get_vertical_thickness(binary, row, mid_col)
                        
                        if thickness > 1:
                            # This is part of a thick region - keep only the bottom pixel
                            bottom_row = row + thickness - 1
                            if bottom_row < height:
                                result[bottom_row, start_col:end_col+1] = 0  # Draw black line
                        else:
                            # Already thin - keep as is
                            result[row, start_col:end_col+1] = 0
                else:
                    col += 1
        
        # Save step 1
        # cv2.imwrite(f'{self.config.output_folder}/step1_horizontal_thinned.png', result)
        
        # Method 2: Scan columns - for each thick vertical segment, keep only right pixel
        print("Step 2: Thinning vertical segments...")
        
        # Start fresh for vertical processing
        result_v = np.full_like(binary, 255)  # All white
        
        for col in range(width):
            row = 0
            while row < height:
                # If we hit a black pixel
                if binary[row, col] == 0:  # Black pixel
                    # Find the end of this vertical segment
                    start_row = row
                    while row < height and binary[row, col] == 0:
                        row += 1
                    end_row = row - 1
                    
                    segment_height = end_row - start_row + 1
                    
                    if segment_height >= 1:  # Any vertical segment
                        # Check how thick this region is horizontally at the middle
                        mid_row = (start_row + end_row) // 2
                        
                        # Find horizontal thickness at this point
                        thickness = self.get_horizontal_thickness(binary, mid_row, col)
                        
                        if thickness > 1:
                            # This is part of a thick region - keep only the right pixel
                            right_col = col + thickness - 1
                            if right_col < width:
                                result_v[start_row:end_row+1, right_col] = 0  # Draw black line
                        else:
                            # Already thin - keep as is
                            result_v[start_row:end_row+1, col] = 0
                else:
                    row += 1
        
        # Save step 2
        # cv2.imwrite(f'{self.config.output_folder}/step2_vertical_thinned.png', result_v)
        
        # Method 3: Combine both results
        print("Step 3: Combining results...")
        final_result = cv2.bitwise_and(result, result_v)  # Keep both thin lines
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, final_result)
        
        return output_path

    def get_vertical_thickness(self, binary, start_row, col):
        """Get the vertical thickness of a black region starting from start_row, col"""
        height = binary.shape[0]
        thickness = 0
        
        row = start_row
        while row < height and binary[row, col] == 0:  # Black pixel
            thickness += 1
            row += 1
        
        return thickness

    def get_horizontal_thickness(self, binary, row, start_col):
        """Get the horizontal thickness of a black region starting from row, start_col"""
        width = binary.shape[1]
        thickness = 0
        
        col = start_col
        while col < width and binary[row, col] == 0:  # Black pixel
            thickness += 1
            col += 1
        
        return thickness

    def remove_diagonal_only_cells(self, image_path, file_name="remove_diagonal_only_cells.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to load the image. Check the file path.")
        
        # Threshold to binary (invert if lines are black on white background)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Pad image to handle border cells easily
        padded = np.pad(binary, pad_width=1, mode='constant', constant_values=0)
        rows, cols = binary.shape
        output = padded.copy()
        
        # Scan each cell (excluding padding)
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                if padded[r, c] == 255:  # Assuming white (255) represents active cells/lines
                    # Get 8 neighbors
                    neighbors = {
                        'top_left': padded[r-1, c-1],
                        'top': padded[r-1, c],
                        'top_right': padded[r-1, c+1],
                        'left': padded[r, c-1],
                        'right': padded[r, c+1],
                        'bottom_left': padded[r+1, c-1],
                        'bottom': padded[r+1, c],
                        'bottom_right': padded[r+1, c+1]
                    }
                    
                    # Helper: Count active neighbors (255)
                    active_count = sum(1 for v in neighbors.values() if v == 255)
                    
                    # Conditions as specified:
                    # 1) Only top-left and bottom-right
                    cond1 = (neighbors['top_left'] == 255 and neighbors['bottom_right'] == 255 and
                            active_count == 2)
                    
                    # 2) Only top-left
                    cond2 = (neighbors['top_left'] == 255 and active_count == 1)
                    
                    # 3) Only bottom-right
                    cond3 = (neighbors['bottom_right'] == 255 and active_count == 1)
                    
                    # 4) Only top-right and bottom-left
                    cond4 = (neighbors['top_right'] == 255 and neighbors['bottom_left'] == 255 and
                            active_count == 2)
                    
                    # 5) Only top-right
                    cond5 = (neighbors['top_right'] == 255 and active_count == 1)
                    
                    # 6) Only bottom-left
                    cond6 = (neighbors['bottom_left'] == 255 and active_count == 1)
                    
                    # Remove cell if any condition matches (set to 0)
                    if cond1 or cond2 or cond3 or cond4 or cond5 or cond6:
                        output[r, c] = 0
        
        # Remove padding and invert back to original style (black lines on white)
        cleaned = output[1:-1, 1:-1]
        result = cv2.bitwise_not(cleaned)
        
        # Save the result
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, result)
        return output_path

    def remove_small_continuity_components(
        self,
        image_path,
        file_name="remove_small_continuity_components.jpg",
        output_folder=None,
    ):
        output_folder = output_folder or self.config.output_folder

        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to load the image. Check the file path.")

        height, width = img.shape
        min_height = height * self.config.min_height_ratio
        min_width = width * self.config.min_width_ratio

        # Threshold to binary (invert if lines are black on white background)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

        # Perform connected component labeling (8-connectivity)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Create output copies
        cleaned_output = binary.copy()
        debug_output = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)  # For visualizing removed components

        for label in tqdm(range(1, num_labels), desc="Processing labels"):
            x, y, w, h, area = stats[label]

            # Filter out small components based on width and height
            if h < min_height and w < min_width:
                cleaned_output[labels == label] = 0
                debug_output[labels == label] = [0, 0, 255]  # Mark removed components in red

        # Invert back to original style
        final_result = cv2.bitwise_not(cleaned_output)

        # Save the final and debug outputs
        output_path = self.get_output_path(output_folder, file_name)
        debug_path = self.get_output_path(output_folder, file_name.replace(".jpg", "_debug.jpg"))

        cv2.imwrite(output_path, final_result)
        cv2.imwrite(debug_path, debug_output)

        return output_path


    def connect_horizontal_vertical_gaps(self, image_path, file_name='connected_output.jpg', output_folder=None):
        output_folder = output_folder or self.config.output_folder

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect all lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        output = image.copy()

        def angle_of_line(x1, y1, x2, y2):
            return abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

        # Filter for only horizontal (≈0°) and vertical (≈90°) lines
        filtered_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = angle_of_line(x1, y1, x2, y2)
                min_width = 0
                min_height = 0

                if angle < 5:
                    line_width = abs(x2 - x1)
                    if line_width >= min_width:
                        filtered_lines.append([x1, y1, x2, y2])

                elif 85 < angle < 95:
                    line_height = abs(y2 - y1)
                    if line_height >= min_height:
                        filtered_lines.append([x1, y1, x2, y2])


        # Merge similar lines (if needed)
        merged_lines = []
        used = [False] * len(filtered_lines)
        horizontal_alignment_threshold = 5
        horizontal_distance_threshold = width * self.config.min_width_ratio
        vertical_alignment_threshold = 5
        vertical_distance_threshold = height * self.config.min_height_ratio
        overlap_allowance = 10

        for i in range(len(filtered_lines)):
            if used[i]:
                continue
            x1a, y1a, x2a, y2a = filtered_lines[i]
            merged = [x1a, y1a, x2a, y2a]
            used[i] = True
            for j in range(i + 1, len(filtered_lines)):
                if used[j]:
                    continue
                x1b, y1b, x2b, y2b = filtered_lines[j]

                # Check if both are horizontal
                if abs(y1a - y2a) < horizontal_alignment_threshold and abs(y1b - y2b) < horizontal_alignment_threshold and abs(y1a - y1b) < horizontal_distance_threshold:
                    if max(x1a, x2a) >= min(x1b, x2b) - overlap_allowance or max(x1b, x2b) >= min(x1a, x2a) - overlap_allowance:
                        merged = [
                            min(merged[0], merged[2], x1b, x2b),
                            y1a,
                            max(merged[0], merged[2], x1b, x2b),
                            y1a
                        ]
                        used[j] = True

                # Check if both are vertical
                elif abs(x1a - x2a) < vertical_alignment_threshold and abs(x1b - x2b) < vertical_alignment_threshold and abs(x1a - x1b) < vertical_distance_threshold:
                    if max(y1a, y2a) >= min(y1b, y2b) - overlap_allowance or max(y1b, y2b) >= min(y1a, y2a) - overlap_allowance:
                        merged = [
                            x1a,
                            min(merged[1], merged[3], y1b, y2b),
                            x1a,
                            max(merged[1], merged[3], y1b, y2b)
                        ]
                        used[j] = True


            merged_lines.append(merged)

        # Draw merged lines
        for x1, y1, x2, y2 in merged_lines:
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 0), 20)
        
        # Save the result
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, output)
        return output_path

    def detect_objects_and_draw_boxess_and_set_white(self, image_path, file_name="all_objects_detected.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder

        # Load image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold to binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours (external only or all)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes
        output = image.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if h < height * self.config.min_height_ratio and w < width * self.config.min_width_ratio:
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), -1)

        # Save output
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, output)
        return output_path
