import os
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio  # Fix for imageio warning
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import measure
from scipy import ndimage as ndi
import re
from skimage.morphology import remove_small_holes
from .image_processor import ImageProcessor
import cv2

pattern = re.compile(r"panel_\d+_\((\d+), (\d+), (\d+), (\d+)\)\.jpg")

def extract_fully_white_panels(
    original_image: np.ndarray,
    segmentation_mask: np.ndarray,
    output_dir: str = "panel_output",
    debug_region_dir: str = "temp_dir/panel_debug_regions",
    min_area_ratio: float = 0.05,
    min_width_ratio: float = 0.05,
    min_height_ratio: float = 0.05,
    save_debug: bool = True
):
    """
    Extract fully white panels from a segmented image.
    
    Args:
        original_image: Original RGB image as numpy array
        segmentation_mask: Binary segmentation mask
        output_dir: Directory to save extracted panels
        debug_region_dir: Directory to save debug images
        min_area_ratio: Minimum area ratio threshold
        min_width_ratio: Minimum width ratio threshold
        min_height_ratio: Minimum height ratio threshold
        save_debug: Whether to save debug images
    
    Returns:
        List of saved panel file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_debug:
        os.makedirs(debug_region_dir, exist_ok=True)

    img_h, img_w = segmentation_mask.shape
    image_area = img_h * img_w

    orig_pil = Image.fromarray(original_image)
    labeled_mask = measure.label(segmentation_mask)
    regions = measure.regionprops(labeled_mask)

    saved_panels = []
    accepted_boxes = []
    panel_idx = 0

    for idx, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        w = maxc - minc
        h = maxr - minr
        area = w * h
        crop_box = (minc, minr, maxc, maxr)
        crop_name_prefix = f"region_{idx+1}"

        # Crops
        cropped_img = orig_pil.crop(crop_box)
        cropped_mask = segmentation_mask[minr:maxr, minc:maxc]
        # Fix for Pillow warning: Remove mode parameter
        mask_pil = Image.fromarray((cropped_mask * 255).astype('uint8'))

        # 1. Threshold check
        if (
            area < min_area_ratio * image_area or
            w < min_width_ratio * img_w or
            h < min_height_ratio * img_h
        ):
            # if save_debug:
            #     cropped_img.save(os.path.join(debug_region_dir, f"{crop_name_prefix}_too_small_orig.jpg"))
            #     mask_pil.save(os.path.join(debug_region_dir, f"{crop_name_prefix}_too_small_mask.jpg"))
            continue

        # 2. Check if region is mostly white (allow small % of black)
        black_pixel_count = np.count_nonzero(region.image == 0)
        total_pixels = region.image.size
        black_ratio = black_pixel_count / total_pixels

        if black_ratio > 0.1:  # Allow up to 1% black pixels
            print(f"❌ Black ratio panel #{idx} — {round(black_ratio * 100, 2)}% black")
            # Save debug info if desired
            if save_debug:
                debug_region_dir_specific = os.path.join(output_dir, f"region_{idx}_skipped_black_inside")
                os.makedirs(debug_region_dir_specific, exist_ok=True)
                
                # Save cropped mask
                cropped_mask = segmentation_mask[minr:maxr, minc:maxc]
                # Fix for Pillow warning: Remove mode parameter
                mask_pil = Image.fromarray((cropped_mask * 255).astype("uint8"))
                mask_pil.save(os.path.join(debug_region_dir_specific, f"region_{idx}_mask.jpg"))
                
                # Highlight black pixels in red and zoom
                highlighted = np.stack([cropped_mask]*3, axis=-1) * 255
                highlighted[cropped_mask == 0] = [255, 0, 0]
                highlighted_zoom = Image.fromarray(highlighted.astype('uint8')).resize(
                    (highlighted.shape[1]*4, highlighted.shape[0]*4), resample=Image.NEAREST
                )
                highlighted_zoom.save(os.path.join(debug_region_dir_specific, f"region_{idx}_highlight_black_zoomed.jpg"))
            
            continue

        # 3. Save valid panel with bbox coordinates in filename
        bbox_str = f"({minc}, {minr}, {maxc}, {maxr})"
        panel_idx = panel_idx + 1
        panel_path = os.path.join(output_dir, f"panel_{panel_idx}_{bbox_str}.jpg")
        cropped_img.save(panel_path)
        saved_panels.append(panel_path)
        accepted_boxes.append((minc, minr, maxc, maxr))

        if save_debug:
            cropped_img.save(os.path.join(debug_region_dir, f"{crop_name_prefix}_saved_orig.jpg"))
            mask_pil.save(os.path.join(debug_region_dir, f"{crop_name_prefix}_saved_mask.jpg"))

    # 4. Debug image with accepted boxes
    if save_debug:
        debug_img = orig_pil.copy()
        draw = ImageDraw.Draw(debug_img)
        for (x1, y1, x2, y2) in accepted_boxes:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        debug_img.save(os.path.join(output_dir, "debug_all_saved_panels.jpg"))

    return saved_panels

def get_region_count(binary_seg):
    labeled_mask = measure.label(binary_seg)
    regions = measure.regionprops(labeled_mask)

    img_h, img_w = binary_seg.shape
    image_area = img_h * img_w
    count = 0
    for idx, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        w = maxc - minc
        h = maxr - minr
        area = w * h

        if (
            area < 0.05 * image_area or
            w < 0.05 * img_w or
            h < 0.05 * img_h
        ):
            continue

        count += 1

    return count

def get_black_white_ratio(image_path, threshold=128):
    """
    Calculates the ratio of black and white pixels in a binary image.
    
    Parameters:
        image_path (str): Path to the image file.
        threshold (int): Threshold value for binarization (default: 128).
    
    Returns:
        dict: Dictionary with black_ratio, white_ratio, black_count, white_count, total_pixels.
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to binary using the given threshold
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    total_pixels = binary.size
    white_count = np.count_nonzero(binary == 255)
    black_count = total_pixels - white_count

    black_ratio = black_count / total_pixels
    white_ratio = white_count / total_pixels

    return {
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
        "black_count": black_count,
        "white_count": white_count,
        "total_pixels": total_pixels
    }


def create_segmentation_mask(image: np.ndarray, save_debug: bool = True) -> np.ndarray:
    """
    Create segmentation mask from image using edge detection and hole filling.
    
    Args:
        image: Input RGB image as numpy array
        save_debug: Whether to save intermediate processing steps
    
    Returns:
        Binary segmentation mask
    """
    if save_debug:
        os.makedirs("temp_dir/panel_debug_steps", exist_ok=True)
        Image.fromarray(image).save("temp_dir/panel_debug_steps/step1_original.jpg")

    # Convert to grayscale
    grayscale = rgb2gray(image)
    if save_debug:
        gray_uint8 = (grayscale * 255).astype('uint8')
        # Fix for Pillow warning: Remove mode parameter
        Image.fromarray(gray_uint8).save("temp_dir/panel_debug_steps/step2_grayscale.jpg")

    # Edge detection
    edges = canny(grayscale)
    edges_uint8 = (edges * 255).astype('uint8')
    if save_debug:
        Image.fromarray(edges_uint8).save("temp_dir/panel_debug_steps/step3_edges.jpg")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    seg = cv2.dilate(edges_uint8, kernel, iterations=2)
    seg = cv2.ximgproc.thinning(seg)
    # Fill holes in edges
    segmentation = ndi.binary_fill_holes(seg)
    # Ensure it's a NumPy boolean or 0/1 array
    binary_seg = segmentation.astype(np.uint8)

    # Count white and black pixels
    total_pixels = binary_seg.size
    white_pixels = np.count_nonzero(binary_seg)  # 1s

    # Ratios
    white_ratio = white_pixels / total_pixels

    region_count = get_region_count(binary_seg)
    if white_ratio > 0.8 or region_count == 1:
        print(f"⚠️ white is maximum hence reverting to only binary_fill_holes")
        # Fill holes in edges
        segmentation = ndi.binary_fill_holes(edges)

    # ✅ Remove small black clusters (holes in white regions)
    segmentation_cleaned = remove_small_holes(segmentation, area_threshold=500)  # adjust threshold as needed

    if save_debug:
        segmentation_uint8 = (segmentation_cleaned * 255).astype('uint8')
        Image.fromarray(segmentation_uint8).save("temp_dir/panel_debug_steps/step4_segmentation_filled.jpg")

    return segmentation_cleaned

def boxes_are_close(box1, box2, thresh):
    # Horizontal overlap or near
    horiz_close = (box1[2] >= box2[0] - thresh and box1[0] <= box2[2] + thresh)
    # Vertical overlap or near
    vert_close = (box1[3] >= box2[1] - thresh and box1[1] <= box2[3] + thresh)
    return horiz_close and vert_close

def merge_close_panels(saved_panels, draw, distance_thresh=20):
    """Merge panels with close bounding boxes and fill them on draw object."""
    # Step 1: Extract bounding boxes
    boxes = []
    for panel_path in saved_panels:
        panel_name = os.path.basename(panel_path)
        match = pattern.match(panel_name)
        if match:
            minc, minr, maxc, maxr = map(int, match.groups())
            boxes.append([minc, minr, maxc, maxr])

    # Step 2: Merge nearby boxes
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        box1 = boxes[i]
        merged_box = box1.copy()

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            box2 = boxes[j]

            # Check if boxes are close (horizontal and vertical)
            if boxes_are_close(box1, box2, distance_thresh):
                # Merge boxes
                merged_box = [
                    min(merged_box[0], box2[0]),
                    min(merged_box[1], box2[1]),
                    max(merged_box[2], box2[2]),
                    max(merged_box[3], box2[3])
                ]
                used[j] = True

        used[i] = True
        merged.append(merged_box)

    # Step 3: Fill merged boxes
    for box in merged:
        draw.rectangle(box, fill=(0, 0, 0))

def create_image_with_panels_removed(
    original_image: np.ndarray,
    segmentation_mask: np.ndarray,
    output_folder: str,
    output_path: str,
    save_debug: True
) -> None:
    """
    Create a version of the original image with detected panels blacked out.
    
    Args:
        original_image: Original RGB image as numpy array
        segmentation_mask: Binary segmentation mask
        output_path: Path to save the modified image
    """
    # Get panel information
    saved_panels = extract_fully_white_panels(
        original_image=original_image,
        segmentation_mask=segmentation_mask,
        output_dir=output_folder,
        debug_region_dir="temp_dir/panel_debug_regions",
        save_debug=save_debug
    )
    
    # Create modified image
    im_no_panels = Image.fromarray(original_image.copy())
    draw = ImageDraw.Draw(im_no_panels)
    
    # Get regions and black them out
    # labeled_mask = measure.label(segmentation_mask)
    # regions = measure.regionprops(labeled_mask)
    
    # for panel_path in saved_panels:
    #     # Extract panel index from filename with bbox format
    #     panel_name = os.path.basename(panel_path)
    #     match = pattern.match(panel_name)
    #     minc, minr, maxc, maxr = map(int, match.groups())

    #     draw.rectangle([minc, minr, maxc, maxr], fill=(0, 0, 0))

    merge_close_panels(saved_panels, draw, distance_thresh=25)
    
    # Save the result
    im_no_panels.save(output_path)


def main(output_folder, input_image_path, original_image_path):
    """Main execution function."""
    # Load the input image
    image = imageio.imread(input_image_path)
    original_image = imageio.imread(original_image_path)
    save_debug = True
    # Create segmentation mask
    segmentation_mask = create_segmentation_mask(image, save_debug=save_debug)
    segmentation_mask_output_path = f"temp_dir/panel_debug_steps/step4_segmentation_filled.jpg"

    pixel_ratios = get_black_white_ratio(segmentation_mask_output_path)

    if pixel_ratios['black_ratio'] < 0.8:
        print(f"✅ black is less hence applying other features")
        image_pros = ImageProcessor()
        new_path = image_pros.thick_black(segmentation_mask_output_path, file_name="step5_thick.jpg", output_folder="temp_dir/panel_debug_steps")

        new_path = image_pros.connect_horizontal_vertical_gaps(new_path, file_name="step6_continuity.jpg", output_folder="temp_dir/panel_debug_steps")

        pixel_ratios = get_black_white_ratio(new_path)
        if pixel_ratios['black_ratio'] < 0.8:
            new_path = image_pros.thin_image_borders(new_path, file_name="step7_thin.jpg", output_folder="temp_dir/panel_debug_steps")

            new_path = image_pros.remove_dangling_lines(new_path, file_name="step8_remove_dangling_lines.jpg", output_folder="temp_dir/panel_debug_steps")

            new_path = image_pros.thick_black(new_path, file_name="step9_thick.jpg", output_folder="temp_dir/panel_debug_steps")

            segmentation_mask = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)

    pre_process_path = f"{output_folder}/00_original_with_panels_removed.jpg"
    # Create image with panels removed
    create_image_with_panels_removed(
        original_image=original_image,
        segmentation_mask=segmentation_mask,
        output_folder=output_folder,
        output_path=pre_process_path,
        save_debug=save_debug
    )

    return pre_process_path


if __name__ == "__main__":
    main('panel_output', 'test7.jpg', 'test7.jpg')
