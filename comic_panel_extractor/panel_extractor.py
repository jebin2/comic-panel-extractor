from typing import List, Tuple
from .config import Config, load_config

import numpy as np
import cv2
from dataclasses import dataclass
import os
import re
from .utils import remove_duplicate_boxes, count_panels_inside, extend_boxes_to_image_border

@dataclass
class PanelData:
    """Represents an extracted comic panel."""
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    width: int
    height: int
    area: int
    
    @classmethod
    def from_coordinates(cls, x1: int, y1: int, x2: int, y2: int) -> 'PanelData':
        """Create PanelData from coordinates."""
        return cls(
            x_start=x1,
            y_start=y1,
            x_end=x2,
            y_end=y2,
            width=x2 - x1,
            height=y2 - y1,
            area=(x2 - x1) * (y2 - y1)
        )

class PanelExtractor:
    """Handles comic panel extraction using black percentage analysis."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def extract_panels(self, dilated_path: str, row_thresh: int = 20, col_thresh: int = 20) -> Tuple[List[np.ndarray], List[PanelData]]:
        """Extract comic panels using black percentage scan."""
        dilated = cv2.imread(dilated_path, cv2.IMREAD_GRAYSCALE)
        original = cv2.imread(self.config.input_path)
        
        if dilated is None or original is None:
            raise FileNotFoundError("Could not load dilated or original image")

        height, width = dilated.shape
        
        # Find row gutters and panel rows
        panel_rows = self._find_panel_rows(dilated, row_thresh)
        
        # Extract panels from each row
        all_panels = []
        for y1, y2 in panel_rows:
            row_panels = self._extract_panels_from_row(dilated, y1, y2, col_thresh)
            all_panels.extend(row_panels)
        
        # Filter panels by size
        filtered_panels = self._filter_panels_by_size(
            all_panels, width, height
        )
        
        # Extract panel images and save
        panel_images, panel_data, all_panel_path = self._save_panels(
            filtered_panels, original, width, height
        )
        
        return panel_images, panel_data, all_panel_path
    
    def _find_panel_rows(self, dilated: np.ndarray, row_thresh: int) -> List[Tuple[int, int]]:
        """Find panel rows where consecutive rows meet the threshold and height constraint."""
        height, width = dilated.shape

        # Calculate black percentage for each row
        row_black_percentage = np.sum(dilated == 0, axis=1) / width * 100

        # Find all rows meeting threshold
        black_rows = [y for y, p in enumerate(row_black_percentage) if p >= row_thresh]

        # Forcefully include first and last row
        if 0 not in black_rows:
            black_rows.insert(0, 0)
        if (height) not in black_rows:
                black_rows.append(height)

        print(f'📄 Row Points:: {black_rows}')
        # Group consecutive rows into gutters
        row_gutters = []
        if black_rows:
            start_row = black_rows[0]
            for i, end_row in enumerate(black_rows):
                # Only extend if combined height meets min_height_ratio
                combined_height = end_row - start_row
                if combined_height / height >= self.config.min_height_ratio:
                    print(f'📄 {i+1}) Start: {start_row:04d} | End: {end_row:04d} | Total: {combined_height:04d} | Ratio: {(combined_height / height):04f}')
                    row_gutters.append((start_row, end_row))
                    start_row = end_row
                elif len(black_rows) == i + 1:
                    row_gutters[-1] = (row_gutters[-1][0], end_row)

        print(f"✅ Detected panel row gutters: {row_gutters}")

        # ⚡ Draw detected rows on a color copy
        visual = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        for (y1, y2) in row_gutters:
            cv2.line(visual, (0, y1), (width, y1), (0, 255, 0), thickness=5)
            cv2.line(visual, (0, y2), (width, y2), (0, 0, 255), thickness=5)

        # Save visualization
        output_path = f"{self.config.output_folder}/row_gutters_visualization.jpg"
        cv2.imwrite(output_path, visual)
        print(f"📄 Saved row gutter visualization: {output_path}")

        return row_gutters

    def _find_panel_columns(self, dilated: np.ndarray, col_thresh: int) -> List[Tuple[int, int]]:
        """
        Find panel columns where consecutive columns meet the threshold and width constraint.
        """
        height, width = dilated.shape

        # Calculate black percentage for each column
        col_black_percentage = np.sum(dilated == 0, axis=0) / height * 100

        # Find all columns meeting threshold
        black_cols = [x for x, p in enumerate(col_black_percentage) if p >= col_thresh]

        # Forcefully include first and last column
        if 0 not in black_cols:
            black_cols.insert(0, 0)
        if (width - 1) not in black_cols:
            black_cols.append(width - 1)

        # Group consecutive columns into gutters
        col_gutters = []
        if black_cols:
            start_col = black_cols[0]
            prev_col = black_cols[0]
            for x in black_cols:
                if x != start_col:
                    # Only extend if combined width meets min_width_ratio
                    combined_width = x - start_col + 1
                    if combined_width / width >= self.config.min_width_ratio:
                        prev_col = x
                        col_gutters.append((start_col, prev_col))
                        start_col = x

            if start_col != prev_col:
                col_gutters.append((start_col, prev_col))  # Add last gutter

        print(f"✅ Detected panel column gutters: {col_gutters}")

        # ⚡ Draw detected columns on a color copy
        visual = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        for (x1, x2) in col_gutters:
            cv2.line(visual, (x1, 0), (x1, height), (255, 0, 0), thickness=5)
            cv2.line(visual, (x2, 0), (x2, height), (0, 255, 255), thickness=5)

        # Save visualization
        output_path = f"{self.config.output_folder}/col_gutters_visualization.jpg"
        cv2.imwrite(output_path, visual)
        print(f"📄 Saved column gutter visualization: {output_path}")

        return col_gutters

    def _extract_panels_from_row(self, dilated: np.ndarray, y1: int, y2: int, 
                                col_thresh: int) -> List[Tuple[int, int, int, int]]:
        """Extract panels from a single row."""
        width = dilated.shape[1]
        row_slice = dilated[y1:y2, :]
        col_black_percentage = np.sum(row_slice == 0, axis=0) / (y2 - y1) * 100

        # Find column gutters
        col_gutters = []
        in_gutter = False
        for x, percent_black in enumerate(col_black_percentage):
            if percent_black >= col_thresh and not in_gutter:
                start_col = x
                in_gutter = True
            elif percent_black < col_thresh and in_gutter:
                end_col = x
                col_gutters.append((start_col, end_col))
                in_gutter = False
        
        # Convert gutters to panel columns
        panel_cols = []
        prev_end = 0
        for start, end in col_gutters:
            if start - prev_end > 10:  # Minimum column width
                panel_cols.append((prev_end, start))
            prev_end = end
        
        if width - prev_end > 10:
            panel_cols.append((prev_end, width))
        
        return [(x1, y1, x2, y2) for x1, x2 in panel_cols]
    
    def _filter_panels_by_size(self, panels: List[Tuple[int, int, int, int]], width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """Filter panels by size constraints."""
        new_panel = []

        for x1, y1, x2, y2 in panels:
            w = x2 - x1  # Corrected
            h = y2 - y1  # Corrected

            if (
                w >= self.config.min_width_ratio * width and
                h >= self.config.min_height_ratio * height
            ):
                new_panel.append((x1, y1, x2, y2))

        return new_panel


    def count_panel_files(self, folder_path: str) -> int:
        """
        Count the number of files in a folder that start with 'panel_'.

        Args:
            folder_path: Path to the folder to search.

        Returns:
            Number of files starting with 'panel_'.
        """
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            return 0

        return len([
            fname for fname in os.listdir(folder_path)
            if fname.startswith("panel_") and os.path.isfile(os.path.join(folder_path, fname))
        ])

    def load_existing_panels_from_folder(self, folder: str) -> List[Tuple[int, int, int, int]]:
        """
        Parses filenames like 'panel_1_(1006, 176, 1757, 1085).jpg' and extracts coordinates.
        """
        pattern = re.compile(r"panel_\d+_\((\d+), (\d+), (\d+), (\d+)\)\.jpg")
        coords = []
        for fname in os.listdir(folder):
            match = pattern.match(fname)
            if match:
                coords.append(tuple(map(int, match.groups())))
        return coords

    def limit_coord(self, new_coord, existing_coords):
        """
        Trim a new panel box from any side to completely avoid overlapping with existing panels.
        
        Args:
            new_coord: Tuple (x1, y1, x2, y2) representing the new panel box
            existing_coords: List of tuples [(x1, y1, x2, y2), ...] representing existing panels
        
        Returns:
            Tuple (x1, y1, x2, y2) representing the trimmed panel box with no overlaps
        """
        if not existing_coords:
            return new_coord
        
        x1, y1, x2, y2 = new_coord
        
        # Ensure valid input coordinates
        if x2 <= x1 or y2 <= y1:
            return new_coord
        
        # Keep trimming until no overlaps exist
        current_box = (x1, y1, x2, y2)
        
        for existing_box in existing_coords:
            ex1, ey1, ex2, ey2 = existing_box
            cx1, cy1, cx2, cy2 = current_box
            
            # Check if current box overlaps with this existing box
            if self.boxes_overlap(current_box, existing_box):
                
                # Calculate possible trim options and their resulting box sizes
                trim_options = []
                
                # Option 1: Trim from left (move x1 right)
                if cx1 < ex2 and cx2 > ex2:
                    new_x1 = ex2
                    if new_x1 < cx2:  # Ensure valid box
                        area = (cx2 - new_x1) * (cy2 - cy1)
                        trim_options.append(('left', (new_x1, cy1, cx2, cy2), area))
                
                # Option 2: Trim from right (move x2 left)
                if cx2 > ex1 and cx1 < ex1:
                    new_x2 = ex1
                    if new_x2 > cx1:  # Ensure valid box
                        area = (new_x2 - cx1) * (cy2 - cy1)
                        trim_options.append(('right', (cx1, cy1, new_x2, cy2), area))
                
                # Option 3: Trim from top (move y1 down)
                if cy1 < ey2 and cy2 > ey2:
                    new_y1 = ey2
                    if new_y1 < cy2:  # Ensure valid box
                        area = (cx2 - cx1) * (cy2 - new_y1)
                        trim_options.append(('top', (cx1, new_y1, cx2, cy2), area))
                
                # Option 4: Trim from bottom (move y2 up)
                if cy2 > ey1 and cy1 < ey1:
                    new_y2 = ey1
                    if new_y2 > cy1:  # Ensure valid box
                        area = (cx2 - cx1) * (new_y2 - cy1)
                        trim_options.append(('bottom', (cx1, cy1, cx2, new_y2), area))
                
                # Choose the trim option that preserves the largest area
                if trim_options:
                    # Sort by area (descending) to keep the largest possible box
                    trim_options.sort(key=lambda x: x[2], reverse=True)
                    best_option = trim_options[0]
                    current_box = best_option[1]
                else:
                    # If no valid trim options, return minimal box
                    return (cx1, cy1, cx1 + 1, cy1 + 1)
        
        return current_box


    def boxes_overlap(self, box1, box2):
        """
        Check if two boxes overlap.
        
        Args:
            box1, box2: Tuples (x1, y1, x2, y2)
        
        Returns:
            Boolean indicating if boxes overlap
        """
        x1, y1, x2, y2 = box1
        ex1, ey1, ex2, ey2 = box2
        
        return not (x2 <= ex1 or x1 >= ex2 or y2 <= ey1 or y1 >= ey2)



    def _save_panels(self, panels: List[Tuple[int, int, int, int]], original: np.ndarray, width: int, height: int) -> Tuple[List[np.ndarray], List[PanelData], List[str]]:
        """Save panel images and return panel data."""
        original_image = cv2.imread(self.config.input_path)
        visual_output = original.copy()
        panel_images = []
        panel_data = []
        all_panel_path = []

        panel_idx = self.count_panel_files(self.config.output_folder)
        black_overlay_input = cv2.imread(self.config.black_overlay_input_path)

        image_area = width * height
        maybe_full_page_panel = None

        # Load existing panels from disk
        existing_coords = self.load_existing_panels_from_folder(self.config.output_folder)

        for idx, (x1, y1, x2, y2) in enumerate(panels, 1):
            # Extract panel image from black_overlay_input
            panel_img = black_overlay_input[y1:y2, x1:x2]

            # Check for mostly black/white
            gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)
            total_pixels = gray.size
            black_pixels = np.sum(gray < 30)
            white_pixels = np.sum(gray > 240)
            black_ratio = black_pixels / total_pixels
            white_ratio = white_pixels / total_pixels

            if black_ratio > 0.8:
                print(f"⚠️ Skipping panel #{idx} — {round(black_ratio * 100, 2)}% black")
                continue
            elif white_ratio > 0.9:
                print(f"⚠️ Skipping panel #{idx} — {round(white_ratio * 100, 2)}% white")
                continue
            else:
                print(f"✅ Panel #{idx} — {round(black_ratio * 100, 2)}% black, {round(white_ratio * 100, 2)}% white")

            panel_area = (x2 - x1) * (y2 - y1)
            if panel_area >= 0.9 * image_area:
                print(f"⚠️ Panel #{idx} covers ≥90% of the image — marked for potential use only")
                maybe_full_page_panel = (idx, (x1, y1, x2, y2))
                continue

            # Check for full containment in existing and current session
            already_saved_coords = existing_coords + [ (pd.x_start, pd.y_start, pd.x_end, pd.y_end) for pd in panel_data ]

            # 1. Skip if duplicate
            is_duplicate, _ = remove_duplicate_boxes(already_saved_coords, (x1, y1, x2, y2))
            if is_duplicate:
                print(f"⚠️ Skipping panel #{idx} — fully contained in existing panel")
                continue

            # 2. Skip if this panel contains ≥1 other panels
            contained_count = count_panels_inside((x1, y1, x2, y2), already_saved_coords, height, width)
            if contained_count >= 1:
                print(f"⚠️ Skipping panel #{idx} — contains {contained_count} other panels inside")
                continue

            x1, y1, x2, y2 = extend_boxes_to_image_border([(x1, y1, x2, y2)], [height, width], self.config.min_width_ratio, self.config.min_height_ratio)[0]
            x1, y1, x2, y2 = self.limit_coord((x1, y1, x2, y2), already_saved_coords)

            if not self._filter_panels_by_size(
                [(x1, y1, x2, y2)], width, height
            ):
                continue

            # Save panel
            panel_img = original_image[y1:y2, x1:x2]
            panel_images.append(panel_img)
            panel_info = PanelData.from_coordinates(x1, y1, x2, y2)
            panel_data.append(panel_info)

            panel_idx += 1
            panel_path = f'{self.config.output_folder}/panel_{panel_idx}_{(x1, y1, x2, y2)}.jpg'
            cv2.imwrite(str(panel_path), panel_img)
            all_panel_path.append(panel_path)

            cv2.rectangle(visual_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(visual_output, f"#{idx}", (x1+5, y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # If no valid panels and full-page backup exists
        if not panel_images and maybe_full_page_panel and panel_idx == 0:
            idx, (x1, y1, x2, y2) = maybe_full_page_panel
            panel_img = original_image[y1:y2, x1:x2]
            panel_images.append(panel_img)
            panel_info = PanelData.from_coordinates(x1, y1, x2, y2)
            panel_data.append(panel_info)

            panel_idx += 1
            panel_path = f'{self.config.output_folder}/panel_{panel_idx}_{(x1, y1, x2, y2)}.jpg'
            cv2.imwrite(str(panel_path), panel_img)
            all_panel_path.append(panel_path)

            cv2.rectangle(visual_output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(visual_output, f"#full", (x1+5, y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            print(f"✅ Saved full-page panel as fallback")

        # Save final visualization
        visual_path = f'{self.config.output_folder}/panels_visualization.jpg'
        cv2.imwrite(str(visual_path), visual_output)

        print(f"✅ Extracted {len(panel_images)} panels after filtering.")
        return panel_images, panel_data, all_panel_path

