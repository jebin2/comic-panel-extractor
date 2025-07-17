from typing import List, Tuple
from .config import Config

import numpy as np
import cv2
from dataclasses import dataclass

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
    
    def extract_panels(self, dilated_path: str, row_thresh: int = 20, col_thresh: int = 20, min_width_ratio: float = 0.1, min_height_ratio: float = 0.1, min_area_ratio: float = 0.005) -> Tuple[List[np.ndarray], List[PanelData]]:
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
            all_panels, width, height, min_width_ratio, min_height_ratio, min_area_ratio
        )
        
        # Extract panel images and save
        panel_images, panel_data = self._save_panels(
            filtered_panels, original, width, height
        )
        
        return panel_images, panel_data
    
    def _find_panel_rows(self, dilated: np.ndarray, row_thresh: int) -> List[Tuple[int, int]]:
        """Find panel rows by analyzing horizontal black percentages."""
        height, width = dilated.shape
        row_black_percentage = np.sum(dilated == 0, axis=1) / width * 100
        
        # Find row gutters
        row_gutters = []
        in_gutter = False
        for y, percent_black in enumerate(row_black_percentage):
            if percent_black >= row_thresh and not in_gutter:
                start_row = y
                in_gutter = True
            elif percent_black < row_thresh and in_gutter:
                end_row = y
                row_gutters.append((start_row, end_row))
                in_gutter = False
        
        # Convert gutters to panel rows
        panel_rows = []
        prev_end = 0
        for start, end in row_gutters:
            if start - prev_end > 10:  # Minimum row height
                panel_rows.append((prev_end, start))
            prev_end = end
        
        if height - prev_end > 10:
            panel_rows.append((prev_end, height))
        
        return panel_rows
    
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
    
    def _save_panels(self, panels: List[Tuple[int, int, int, int]], 
                    original: np.ndarray, width: int, height: int) -> Tuple[List[np.ndarray], List[PanelData]]:
        """Save panel images and return panel data."""
        visual_output = original.copy()
        panel_images = []
        panel_data = []
        
        for idx, (x1, y1, x2, y2) in enumerate(panels, 1):
            # Extract panel image
            panel_img = original[y1:y2, x1:x2]
            panel_images.append(panel_img)
            
            # Create panel data
            panel_info = PanelData.from_coordinates(x1, y1, x2, y2)
            panel_data.append(panel_info)
            
            # Save panel image
            panel_path = f'{self.config.output_folder}/panel_{idx}.jpg'
            cv2.imwrite(str(panel_path), panel_img)
            
            # Draw visualization
            cv2.rectangle(visual_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(visual_output, f"#{idx}", (x1+5, y1+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Save visualization
        visual_path = f'{self.config.output_folder}/panels_visualization.jpg'
        cv2.imwrite(str(visual_path), visual_output)
        
        print(f"✅ Extracted {len(panels)} panels after filtering.")
        return panel_images, panel_data