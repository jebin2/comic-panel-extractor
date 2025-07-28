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


def extract_fully_white_panels(
    original_image: np.ndarray,
    segmentation_mask: np.ndarray,
    output_dir: str = "panel_output",
    debug_region_dir: str = "panel_debug_regions",
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
            if save_debug:
                cropped_img.save(os.path.join(debug_region_dir, f"{crop_name_prefix}_too_small_orig.jpg"))
                mask_pil.save(os.path.join(debug_region_dir, f"{crop_name_prefix}_too_small_mask.jpg"))
            continue

        # 2. Check if region is mostly white (allow small % of black)
        black_pixel_count = np.count_nonzero(region.image == 0)
        total_pixels = region.image.size
        black_ratio = black_pixel_count / total_pixels

        if black_ratio > 0.02:  # Allow up to 1% black pixels
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
        os.makedirs("panel_debug_steps", exist_ok=True)
        Image.fromarray(image).save("panel_debug_steps/step1_original.jpg")

    # Convert to grayscale
    grayscale = rgb2gray(image)
    if save_debug:
        gray_uint8 = (grayscale * 255).astype('uint8')
        # Fix for Pillow warning: Remove mode parameter
        Image.fromarray(gray_uint8).save("panel_debug_steps/step2_grayscale.jpg")

    # Edge detection
    edges = canny(grayscale)
    if save_debug:
        edges_uint8 = (edges * 255).astype('uint8')
        # Fix for Pillow warning: Remove mode parameter
        Image.fromarray(edges_uint8).save("panel_debug_steps/step3_edges.jpg")

    # Fill holes in edges
    segmentation = ndi.binary_fill_holes(edges)

    # ✅ Remove small black clusters (holes in white regions)
    segmentation_cleaned = remove_small_holes(segmentation, area_threshold=500)  # adjust threshold as needed

    if save_debug:
        segmentation_uint8 = (segmentation_cleaned * 255).astype('uint8')
        Image.fromarray(segmentation_uint8).save("panel_debug_steps/step4_segmentation_filled.jpg")

    return segmentation_cleaned


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
        debug_region_dir="panel_debug_regions",
        save_debug=save_debug
    )
    
    # Create modified image
    im_no_panels = Image.fromarray(original_image.copy())
    draw = ImageDraw.Draw(im_no_panels)
    
    # Get regions and black them out
    labeled_mask = measure.label(segmentation_mask)
    regions = measure.regionprops(labeled_mask)
    pattern = re.compile(r"panel_\d+_\((\d+), (\d+), (\d+), (\d+)\)\.jpg")
    
    for panel_path in saved_panels:
        # Extract panel index from filename with bbox format
        panel_name = os.path.basename(panel_path)
        match = pattern.match(panel_name)
        minc, minr, maxc, maxr = map(int, match.groups())

        draw.rectangle([minc, minr, maxc, maxr], fill=(0, 0, 0))
    
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

    pre_process_path = f"{output_folder}/original_with_panels_removed.jpg"
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
