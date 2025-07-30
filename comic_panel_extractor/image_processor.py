from typing import List, Tuple
from pathlib import Path
from .config import Config

import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from skimage import measure
from tqdm import tqdm

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
            if (width < width_ * self.config.min_width_ratio or height < height_ * self.config.min_height_ratio):
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
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                    if 80 < angle < 100:
                        if length / height_ > self.config.min_height_ratio:
                            break  # keep region
                    elif angle < 10 or angle > 170:
                        if length / width_ > self.config.min_width_ratio:
                            break  # keep region
                else:
                    # If no qualifying line found, remove region
                    clean_mask[labeled == region.label] = 0
                    cv2.rectangle(visual, (minc, minr), (maxc, maxr), (0, 255, 255), 2)
            else:
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

    def remove_small_continuity_components(self, image_path, file_name="remove_small_continuity_components.jpg", output_folder=None):
        output_folder = output_folder or self.config.output_folder
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to load the image. Check the file path.")

        height, width = img.shape
        continuity_threshold = height * self.config.min_height_ratio
        # Threshold to binary (invert if lines are black on white background)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Perform connected component labeling (8-connectivity)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Create a copy for output
        output = binary.copy()
        
        # Iterate over components (skip label 0, which is background)
        for label in tqdm(range(1, num_labels), desc="Processing labels"):
            # Get the size (area) of the component
            size = stats[label, cv2.CC_STAT_AREA]
            
            # If size is below threshold, remove the component (set to 0)
            if size < continuity_threshold:
                output[labels == label] = 0
        
        # Invert back to original style (black lines on white)
        result = cv2.bitwise_not(output)
        
        # Save the result
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, result)
        return output_path

    def connect_horizontal_vertical_gaps(self, image_path, file_name='connected_output.jpg', output_folder=None):
        output_folder = output_folder or self.config.output_folder
        
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to load the image. Check the file path.")
        height, width = img.shape
        # Threshold to binary (invert if lines are black on white background)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        
        rows, cols = binary.shape
        canvas = binary.copy()  # Work on a copy (lines=255 on black)

        gap_threshold = width * self.config.min_width_ratio
        # Scan row by row to connect small horizontal gaps
        for r in range(rows):
            col = 0
            while col < cols:
                if canvas[r, col] == 255:
                    # Find start and end of current segment
                    start = col
                    while col < cols and canvas[r, col] == 255:
                        col += 1
                    end = col - 1
                    
                    # Look for next segment in the same row
                    next_start = col
                    while next_start < cols and canvas[r, next_start] == 0:
                        next_start += 1
                    if next_start < cols:
                        gap = next_start - end - 1
                        if gap >= 0 and gap <= gap_threshold:
                            # Fill the gap
                            for fill_col in range(end + 1, next_start):
                                canvas[r, fill_col] = 255
                            col = next_start  # Jump to next segment
                        else:
                            col = next_start
                    else:
                        col = next_start
                else:
                    col += 1
        gap_threshold = height * self.config.min_height_ratio
        # Scan column by column to connect small vertical gaps
        for c in range(cols):
            row = 0
            while row < rows:
                if canvas[row, c] == 255:
                    # Find start and end of current segment
                    start = row
                    while row < rows and canvas[row, c] == 255:
                        row += 1
                    end = row - 1
                    
                    # Look for next segment in the same column
                    next_start = row
                    while next_start < rows and canvas[next_start, c] == 0:
                        next_start += 1
                    if next_start < rows:
                        gap = next_start - end - 1
                        if gap >= 0 and gap <= gap_threshold:
                            # Fill the gap
                            for fill_row in range(end + 1, next_start):
                                canvas[fill_row, c] = 255
                            row = next_start  # Jump to next segment
                        else:
                            row = next_start
                    else:
                        row = next_start
                else:
                    row += 1
        
        # Invert back to original style (black lines on white)
        result = cv2.bitwise_not(canvas)
        
        # Save the result
        output_path = self.get_output_path(output_folder, file_name)
        cv2.imwrite(output_path, result)
        return output_path