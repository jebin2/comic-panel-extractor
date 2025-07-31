import os
import re
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import measure
from scipy import ndimage as ndi
from skimage.morphology import remove_small_holes
import cv2

from .config import Config
from .image_processor import ImageProcessor
from .utils import remove_duplicate_boxes, count_panels_inside

class BorderPanelExtractor:
    """
    Handles image preprocessing operations for extracting comic/manga panels.
    
    This class provides functionality to:
    - Create segmentation masks from images
    - Extract white panels from segmented images
    - Remove panels from original images
    - Merge nearby panels
    """
    
    def __init__(self, config: Config = None):
        """Initialize the BorderPanelExtractor with optional configuration."""
        self.config = config or Config()
        self.output_folder = f'{self.config.output_folder}/border_panel_extractor'
        os.makedirs(self.output_folder, exist_ok=True)
        self.PANEL_FILENAME_PATTERN = re.compile(self.config.panel_filename_pattern)

    def create_segmentation_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create segmentation mask from image using edge detection and hole filling.
        
        Args:
            image: Input RGB image as numpy array
        
        Returns:
            Binary segmentation mask as numpy array
        """
        Image.fromarray(image).save(f"{self.output_folder}/00_original.jpg")

        # Convert to grayscale and detect edges
        grayscale = rgb2gray(image)
        edges = canny(grayscale)

        self._save_debug_image(grayscale, f"{self.output_folder}/01_grayscale.jpg")
        self._save_debug_image(edges, f"{self.output_folder}/02_edges.jpg")

        # Process edges with morphological operations
        segmentation = self._process_edges_for_segmentation(edges)
        
        # Check if additional processing is needed
        if self._needs_edge_fallback(segmentation):
            print("⚠️ White ratio too high, reverting to basic edge filling")
            segmentation = ndi.binary_fill_holes(edges)

        # Clean up small holes
        segmentation_cleaned = remove_small_holes(
            segmentation, 
            area_threshold=500
        )

        segmentation_filled_path = f"{self.output_folder}/03_segmentation_filled.jpg"
        self._save_debug_image(
            segmentation_cleaned, 
            segmentation_filled_path
        )

        return segmentation_cleaned, segmentation_filled_path

    def extract_fully_white_panels(
        self,
        original_image: np.ndarray,
        segmentation_mask: np.ndarray
    ):
        """
        Extract fully white panels from a segmented image.
        
        Args:
            original_image: Original RGB image as numpy array
            segmentation_mask: Binary segmentation mask
        
        Returns:
            List of saved panel file paths
        """
        # Get image dimensions and prepare data
        img_h, img_w = segmentation_mask.shape
        image_area = img_h * img_w
        orig_pil = Image.fromarray(original_image)
        
        # Find and process regions
        labeled_mask = measure.label(segmentation_mask)
        regions = measure.regionprops(labeled_mask)

        accepted_boxes = []

        for idx, region in enumerate(regions):
            # Extract region properties
            minr, minc, maxr, maxc = region.bbox
            w, h = maxc - minc, maxr - minr
            area = w * h

            # Check size thresholds
            if self._meets_size_requirements(area, w, h, image_area, img_w, img_h):
                continue

            # Check if region is mostly white
            if not self._is_mostly_white_region(region, idx):
                continue

            # Save valid panel
            accepted_boxes.append((minc, minr, maxc, maxr))

        self._create_visualization(orig_pil, accepted_boxes, "extract_fully_white_panels.jpg")

        return accepted_boxes

    def extract_with_contours(
        self,
        original_image: np.ndarray,
        segmentation_mask_path: str
    ):
        img = cv2.imread(segmentation_mask_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        accepted_boxes = []
        # Draw bounding rectangles
        img_h, img_w = original_image.shape[:2]
        image_area = img_h * img_w
        max_ratio = 0.7  # Max box area must be less than 70% of image
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            box_area = w * h
            if box_area / image_area < max_ratio:
                minc, minr = x, y
                maxc, maxr = x + w, y + h
                accepted_boxes.append((minc, minr, maxc, maxr))

        orig_pil = Image.fromarray(original_image)
        self._create_visualization(orig_pil, accepted_boxes, "extract_with_contours.jpg")

        return accepted_boxes

    def remove_duplicate_boxes(self, boxes, iou_threshold=0.7):
        """
        Removes duplicate or highly overlapping boxes, keeping the larger one.
        :param boxes: List of (x1, y1, x2, y2) boxes.
        :param iou_threshold: Threshold above which boxes are considered duplicates.
        :return: List of unique boxes.
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
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou

        def compute_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

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

    def extend_boxes_to_image_border(self, boxes, image_shape):
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
        height, width = image_shape[:2]
        adjusted_boxes = []

        width_threshold = min(x2 - x1 for x1, y1, x2, y2 in extended_boxes)
        height_threshold = min(y2 - y1 for x1, y1, x2, y2 in extended_boxes)

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

    def remove_swallow_boxes(self, boxes):
        filtered_boxes = []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            current_box = (x1, y1, x2, y2)
            # Count how many other boxes are fully inside this one
            inside_count = count_panels_inside(current_box, [b for j, b in enumerate(boxes) if j != i])

            # Skip this box if it fully contains at least one other box (i.e., it's swallowing)
            if inside_count >= 1:
                continue

            # Keep boxes that don't swallow others
            filtered_boxes.append(current_box)

        print(f"✅ Found {abs(len(filtered_boxes) - len(boxes))} swallowed boxes")
        return filtered_boxes


    def create_image_with_panels_removed(
        self,
        original_image: np.ndarray,
        segmentation_mask: np.ndarray,
        segmentation_mask_path: str
    ) -> None:
        """
        Create a version of the original image with detected panels blacked out.
        
        Args:
            original_image: Original RGB image as numpy array
            segmentation_mask: Binary segmentation mask
            output_path: Path to save the modified image
        """
        # Extract panels
        accepted_boxes = self.extract_fully_white_panels(
            original_image=original_image,
            segmentation_mask=segmentation_mask
        )

        accepted_boxes.extend(
            self.extract_with_contours(
                original_image=original_image,
                segmentation_mask_path=segmentation_mask_path
            )
        )

        accepted_boxes = remove_duplicate_boxes(accepted_boxes)

        accepted_boxes = self.threshold_based_filter(accepted_boxes, original_image.shape)

        accepted_boxes = remove_duplicate_boxes(accepted_boxes)

        accepted_boxes = self.extend_boxes_to_image_border(accepted_boxes, original_image.shape)

        accepted_boxes = remove_duplicate_boxes(accepted_boxes)

        accepted_boxes = sorted(accepted_boxes, key=lambda b: (b[1], b[0]))  # sort by y1, then x1

        accepted_boxes = self.extend_to_nearby_boxes(accepted_boxes, original_image.shape)

        accepted_boxes = remove_duplicate_boxes(accepted_boxes)

        accepted_boxes = self.remove_swallow_boxes(accepted_boxes)

        all_paths = self._save_panel(accepted_boxes)

        output_path = self.draw_black(original_image, accepted_boxes)

        return all_paths, output_path

    def draw_black(self, original_image, accepted_boxes) -> None:
        orig_pil = Image.fromarray(original_image.copy())
        draw = ImageDraw.Draw(orig_pil)

        stripe_height = 10

        for x1, y1, x2, y2 in accepted_boxes:
            for y in range(y1, y2, stripe_height):
                color = (0, 0, 0) if ((y - y1) // stripe_height) % 2 == 0 else (255, 255, 255)
                y_end = min(y + stripe_height, y2)
                draw.rectangle([x1, y, x2, y_end], fill=color)
        
        # Save the result
        output_path = os.path.join(self.config.output_folder, "00_original_with_panels_removed.jpg")
        orig_pil.save(output_path)

        return output_path

    def get_black_white_ratio(self, image_path: str, threshold: int = 128) -> dict:
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

    def get_region_count(self, binary_seg: np.ndarray) -> int:
        """
        Count valid regions in binary segmentation based on size thresholds.
        
        Args:
            binary_seg: Binary segmentation mask
            
        Returns:
            Number of valid regions
        """
        labeled_mask = measure.label(binary_seg)
        regions = measure.regionprops(labeled_mask)

        img_h, img_w = binary_seg.shape
        image_area = img_h * img_w
        count = 0
        
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            w, h = maxc - minc, maxr - minr
            area = w * h

            if self._meets_size_requirements(area, w, h, image_area, img_w, img_h):
                continue
            count += 1

        return count

    def main(self, processed_image_path) -> str:
        """
        Main execution function for panel extraction and removal.

        Returns:
            Path to the processed image with panels removed
        """
        # Load images
        image = imageio.imread(processed_image_path)
        original_image = imageio.imread(processed_image_path)
        
        # Create initial segmentation mask
        segmentation_mask, segmentation_mask_path = self.create_segmentation_mask(image)

        # Check if additional processing is needed
        pixel_ratios = self.get_black_white_ratio(segmentation_mask_path)
        
        if pixel_ratios['black_ratio'] < 0.8:
            print("✅ Black ratio is low, applying additional image processing")
            segmentation_mask, segmentation_mask_path = self._apply_additional_processing(segmentation_mask_path)

        # Create final output
        all_paths, output_path = self.create_image_with_panels_removed(
            original_image=original_image,
            segmentation_mask=segmentation_mask,
            segmentation_mask_path=segmentation_mask_path
        )

        return output_path

    def _save_debug_image(self, image_array: np.ndarray, path: str) -> None:
        """Save debug image with proper format conversion."""
        if image_array.dtype == bool or image_array.max() <= 1:
            image_uint8 = (image_array * 255).astype('uint8')
        else:
            image_uint8 = image_array.astype('uint8')
        Image.fromarray(image_uint8).save(path)

    def _process_edges_for_segmentation(self, edges: np.ndarray) -> np.ndarray:
        """Process edges with morphological operations and fill holes."""
        edges_uint8 = (edges * 255).astype('uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg = cv2.dilate(edges_uint8, kernel, iterations=2)
        seg = cv2.ximgproc.thinning(seg)
        return ndi.binary_fill_holes(seg)

    def _needs_edge_fallback(self, segmentation: np.ndarray) -> bool:
        """Check if edge fallback processing is needed."""
        binary_seg = segmentation.astype(np.uint8)
        total_pixels = binary_seg.size
        white_pixels = np.count_nonzero(binary_seg)
        white_ratio = white_pixels / total_pixels
        region_count = self.get_region_count(binary_seg)
        return white_ratio > 0.8 or region_count == 1

    def _meets_size_requirements(self, area: int, width: int, height: int, image_area: int, img_width: int, img_height: int) -> bool:
        """Check if region meets minimum size requirements."""
        return (area < self.config.min_area_ratio * image_area or
                width < self.config.min_width_ratio * img_width or
                height < self.config.min_height_ratio * img_height)

    def _is_mostly_white_region(self, region, idx: int) -> bool:
        """Check if region is mostly white (allowing small percentage of black)."""
        black_pixel_count = np.count_nonzero(region.image == 0)
        total_pixels = region.image.size
        black_ratio = black_pixel_count / total_pixels
        
        if black_ratio > 0.1:
            print(f"❌ Region #{idx} rejected — {round(black_ratio * 100, 2)}% black pixels")
            self._save_black_region_debug(region, idx)
            return False
        return True

    def _save_black_region_debug(self, region, idx: int) -> None:
        """Save debug information for rejected black regions."""
        debug_dir = os.path.join(self.output_folder, f"region_{idx}_skipped_black_inside")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create highlighted visualization
        highlighted = np.stack([region.image] * 3, axis=-1) * 255
        highlighted[region.image == 0] = [255, 0, 0]  # Red for black pixels
        
        # Save zoomed version
        highlighted_img = Image.fromarray(highlighted.astype('uint8'))
        zoomed = highlighted_img.resize(
            (highlighted.shape[1] * 4, highlighted.shape[0] * 4), 
            resample=Image.NEAREST
        )
        zoomed.save(os.path.join(debug_dir, f"region_{idx}_highlight_black_zoomed.jpg"))

    def _save_panel(self, accepted_boxes) -> str:
        """Save extracted panel with coordinates in filename."""
        original_image = imageio.imread(self.config.input_path)
        orig_pil = Image.fromarray(original_image.copy())
        panel_idx = 0
        all_paths = []
        for minc, minr, maxc, maxr in accepted_boxes:
            panel_idx += 1
            bbox_str = f"({minc}, {minr}, {maxc}, {maxr})"
            panel_path = os.path.join(self.config.output_folder, f"panel_{panel_idx}_{bbox_str}.jpg")
            cropped_img = orig_pil.crop((minc, minr, maxc, maxr))
            cropped_img.save(panel_path)
            all_paths.append(panel_path)

        print(f'✅ Extracted {len(all_paths)} panels.')
        return all_paths

    def _save_debug_panel(self, orig_pil: Image.Image, segmentation_mask: np.ndarray, minr: int, minc: int, maxr: int, maxc: int, idx: int, debug_region_dir: str) -> None:
        """Save debug images for accepted panels."""
        crop_name_prefix = f"region_{idx+1}"
        
        # Save cropped original
        cropped_img = orig_pil.crop((minc, minr, maxc, maxr))
        cropped_img.save(os.path.join(debug_region_dir, f"{crop_name_prefix}_saved_orig.jpg"))
        
        # Save cropped mask
        cropped_mask = segmentation_mask[minr:maxr, minc:maxc]
        mask_pil = Image.fromarray((cropped_mask * 255).astype('uint8'))
        mask_pil.save(os.path.join(debug_region_dir, f"{crop_name_prefix}_saved_mask.jpg"))

    def _create_visualization(self, orig_pil: Image.Image, accepted_boxes: list, file_name: str) -> None:
        """Create debug image showing all accepted panel boxes."""
        debug_img = orig_pil.copy()
        draw = ImageDraw.Draw(debug_img)
        for (x1, y1, x2, y2) in accepted_boxes:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=10)
        debug_img.save(os.path.join(self.output_folder, file_name))

    def extend_to_nearby_boxes(self, boxes, image_shape):
        """
        Extend smaller boxes to the edge of close larger boxes, without merging or reducing the box count.

        A box is represented by (x1, y1, x2, y2).
        """
        if not boxes:
            return boxes
        extended_boxes = [list(box) for box in boxes]
        height, width = image_shape[:2]

        width_threshold = min(x2 - x1 for x1, y1, x2, y2 in extended_boxes)
        height_threshold = min(y2 - y1 for x1, y1, x2, y2 in extended_boxes)

        # width_threshold = self.config.min_width_ratio * width
        # height_threshold = self.config.min_height_ratio * height

        # print(f"[DEBUG] Image Shape: {image_shape}, Width Threshold: {width_threshold:.2f}, Height Threshold: {height_threshold:.2f}\n")

        for i in range(len(extended_boxes)):
            for j in range(len(extended_boxes)):
                if i == j:
                    continue

                box1 = extended_boxes[i]
                box2 = extended_boxes[j]

                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                if area1 >= area2:
                    continue

                # print(f"[DEBUG] Comparing smaller Box {i} {box1} with larger Box {j} {box2}")

                x1_1, y1_1, x2_1, y2_1 = box1
                x1_2, y1_2, x2_2, y2_2 = box2

                # Horizontal Extension Check
                is_vertically_aligned = (y1_1 < y2_2 and y2_1 > y1_2)
                if is_vertically_aligned:
                    gap_right = x1_2 - x2_1
                    if 0 < gap_right <= width_threshold:
                        # print(f"  [INFO] Extending right of Box {i}. Gap ({gap_right:.2f}) <= Threshold ({width_threshold:.2f})")
                        extended_boxes[i][2] = x1_2
                    # elif gap_right > width_threshold:
                        # print(f"  [DEBUG] Did not extend right: Gap ({gap_right:.2f}) > Threshold ({width_threshold:.2f})")

                    gap_left = x1_1 - x2_2
                    if 0 < gap_left <= width_threshold:
                        # print(f"  [INFO] Extending left of Box {i}. Gap ({gap_left:.2f}) <= Threshold ({width_threshold:.2f})")
                        extended_boxes[i][0] = x2_2
                    # elif gap_left > width_threshold:
                        #  print(f"  [DEBUG] Did not extend left: Gap ({gap_left:.2f}) > Threshold ({width_threshold:.2f})")
                # else:
                    # print(f"  [DEBUG] Not vertically aligned for horizontal extension.")


                # Vertical Extension Check
                is_horizontally_aligned = (x1_1 < x2_2 and x2_1 > x1_2)
                if is_horizontally_aligned:
                    gap_bottom = y1_2 - y2_1
                    if 0 < gap_bottom <= height_threshold:
                        # print(f"  [INFO] Extending bottom of Box {i}. Gap ({gap_bottom:.2f}) <= Threshold ({height_threshold:.2f})")
                        extended_boxes[i][3] = y1_2
                    # elif gap_bottom > height_threshold:
                        # print(f"  [DEBUG] Did not extend bottom: Gap ({gap_bottom:.2f}) > Threshold ({height_threshold:.2f})")

                    gap_top = y1_1 - y2_2
                    if 0 < gap_top <= height_threshold:
                        # print(f"  [INFO] Extending top of Box {i}. Gap ({gap_top:.2f}) <= Threshold ({height_threshold:.2f})")
                        extended_boxes[i][1] = y2_2
                    # elif gap_top > height_threshold:
                        # print(f"  [DEBUG] Did not extend top: Gap ({gap_top:.2f}) > Threshold ({height_threshold:.2f})")
                # else:
                    # print(f"  [DEBUG] Not horizontally aligned for vertical extension.")
            # print("-" * 20)

        return [tuple(box) for box in extended_boxes]

    def threshold_based_filter(self, boxes, image_shape):
        img_h, img_w = image_shape[:2]
        image_area = img_h * img_w

        filtered_box = []
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            area = w * h

            if self._meets_size_requirements(area, w, h, image_area, img_w, img_h):
                continue

            filtered_box.append((x1, y1, x2, y2))

        return filtered_box

    def _apply_additional_processing(self, segmentation_mask_path: str) -> np.ndarray:
        """Apply additional image processing steps when needed."""
        image_processor = ImageProcessor()
        
        # Step 5: Thicken black lines
        processed_path = image_processor.thick_black(
            segmentation_mask_path, 
            file_name="04_thick.jpg", 
            output_folder=f"{self.output_folder}"
        )
        
        # Step 6: Connect gaps
        processed_path = image_processor.connect_horizontal_vertical_gaps(
            processed_path, 
            file_name="05_continuity.jpg", 
            output_folder=f"{self.output_folder}"
        )
        
        # Check if more processing is needed
        pixel_ratios = self.get_black_white_ratio(processed_path)
        if pixel_ratios['black_ratio'] < 0.8:
            # Additional processing steps
            processed_path = image_processor.thin_image_borders(
                processed_path, 
                file_name="06_thin.jpg", 
                output_folder=f"{self.output_folder}"
            )
            
            processed_path = image_processor.remove_dangling_lines(
                processed_path, 
                file_name="07_remove_dangling_lines.jpg", 
                output_folder=f"{self.output_folder}"
            )
            
            processed_path = image_processor.thick_black(
                processed_path, 
                file_name="08_thick.jpg", 
                output_folder=f"{self.output_folder}"
            )
        
        return cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE), processed_path


if __name__ == "__main__":
    config = Config()
    config.input_path = "test0.jpg"

    import shutil
    shutil.rmtree(config.output_folder, ignore_errors=True)

    extractor = BorderPanelExtractor(config)
    result_path = extractor.main()
    print(f"Processing complete. Result saved to: {result_path}")