from typing import List, Tuple
from .config import Config

import numpy as np
import cv2
from dataclasses import dataclass
import os

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
    
    def extract_panels(self, dilated_path: str, row_thresh: int = 20, col_thresh: int = 20, min_width_ratio: float = 0.001, min_height_ratio: float = 0.001, min_area_ratio: float = 0) -> Tuple[List[np.ndarray], List[PanelData]]:
        """Extract comic panels using black percentage scan."""
        dilated = cv2.imread(dilated_path, cv2.IMREAD_GRAYSCALE)
        original = cv2.imread(self.config.input_path)
        
        if dilated is None or original is None:
            raise FileNotFoundError("Could not load dilated or original image")

        height, width = dilated.shape
        
        # Find row gutters and panel rows
        panel_rows = self._find_panel_rows(dilated, row_thresh, min_height_ratio)
        
        # Extract panels from each row
        all_panels = []
        for y1, y2 in panel_rows:
            row_panels = self._extract_panels_from_row(dilated, y1, y2, col_thresh)
            all_panels.extend(row_panels)
        
        # Filter panels by size
        filtered_panels = self._filter_panels_by_size(
            all_panels, width, height, min_width_ratio, min_height_ratio, min_area_ratio
        )
        
        # Extract panel images and save
        panel_images, panel_data, all_panel_path = self._save_panels(
            filtered_panels, original, width, height
        )
        
        return panel_images, panel_data, all_panel_path
    
    def _find_panel_rows(self, dilated: np.ndarray, row_thresh: int, min_height_ratio: float) -> List[Tuple[int, int]]:
        """Find panel rows where consecutive rows meet the threshold and height constraint."""
        height, width = dilated.shape

        # Calculate black percentage for each row
        row_black_percentage = np.sum(dilated == 0, axis=1) / width * 100

        # Find all rows meeting threshold
        black_rows = [y for y, p in enumerate(row_black_percentage) if p >= row_thresh]

        # Forcefully include first and last row
        if 0 not in black_rows:
            black_rows.insert(0, 0)
        if (height - 1) not in black_rows:
            black_rows.append(height - 1)

        # Group consecutive rows into gutters
        row_gutters = []
        if black_rows:
            start_row = black_rows[0]
            prev_row = black_rows[0]
            for y in black_rows:
                if y != start_row:
                    # Only extend if combined height meets min_height_ratio
                    combined_height = y - start_row + 1
                    if combined_height / height >= min_height_ratio:
                        prev_row = y
                        row_gutters.append((start_row, prev_row))
                        start_row = y

            if start_row != prev_row:
                row_gutters.append((start_row, prev_row))  # Add last gutter

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

    def _find_panel_columns(self, dilated: np.ndarray, col_thresh: int, min_width_ratio: float) -> List[Tuple[int, int]]:
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
                    if combined_width / width >= min_width_ratio:
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
    
    def _filter_panels_by_size(self, panels: List[Tuple[int, int, int, int]], 
                              width: int, height: int, min_width_ratio: float, 
                              min_height_ratio: float, min_area_ratio: float) -> List[Tuple[int, int, int, int]]:
        """Filter panels by size constraints."""
        # Remove very small panels first
        panels = [(x1, y1, x2, y2) for x1, y1, x2, y2 in panels 
                 if (x2 - x1) * (y2 - y1) >= (width * height) * min_area_ratio]
        
        if not panels:
            return []
        
        # Calculate average dimensions for smart filtering
        panel_widths = [x2 - x1 for x1, _, x2, _ in panels]
        panel_heights = [y2 - y1 for _, y1, _, y2 in panels]
        avg_width = np.mean(panel_widths)
        avg_height = np.mean(panel_heights)
        
        min_allowed_width = max(avg_width * 0.5, width * min_width_ratio)
        min_allowed_height = max(avg_height * 0.5, height * min_height_ratio)
        
        return [(x1, y1, x2, y2) for x1, y1, x2, y2 in panels 
                if (x2 - x1) >= min_allowed_width and (y2 - y1) >= min_allowed_height]

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

    def _save_panels(self, panels: List[Tuple[int, int, int, int]], original: np.ndarray, width: int, height: int) -> Tuple[List[np.ndarray], List[PanelData], List[str]]:
        """Save panel images and return panel data."""
        visual_output = original.copy()
        panel_images = []
        panel_data = []
        all_panel_path = []

        panel_idx = self.count_panel_files(self.config.output_folder)
        black_overlay_input = cv2.imread(self.config.black_overlay_input_path)

        image_area = width * height
        maybe_full_page_panel = None  # Store panel that is ≥90% of the page

        for idx, (x1, y1, x2, y2) in enumerate(panels, 1):
            # Extract panel image from black_overlay_input
            panel_img = black_overlay_input[y1:y2, x1:x2]

            # Check for mostly black content
            gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)
            black_pixels = np.sum(gray < 30)
            total_pixels = gray.size
            black_ratio = black_pixels / total_pixels

            if black_ratio > 0.8:
                print(f"⚠️ Skipping panel #{idx} — {round(black_ratio * 100, 2)}% black")
                continue
            else:
                print(f"✅ Black ratio panel #{idx} — {round(black_ratio * 100, 2)}% black")

            # Check if this panel is ≥90% of the full image
            panel_area = (x2 - x1) * (y2 - y1)
            if panel_area >= 0.9 * image_area:
                print(f"⚠️ Panel #{idx} covers ≥90% of the image — marked for potential use only")
                maybe_full_page_panel = (idx, (x1, y1, x2, y2))
                continue  # Skip for now

            # Save valid smaller panel
            panel_img = visual_output[y1:y2, x1:x2]
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

        # If no valid panels were saved, and we had a full-page one, save it
        if not panel_images and maybe_full_page_panel and panel_idx == 0:
            idx, (x1, y1, x2, y2) = maybe_full_page_panel
            panel_img = visual_output[y1:y2, x1:x2]
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
