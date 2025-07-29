# from .text_detector import TextDetector
from .config import Config
from .image_processor import ImageProcessor
from .panel_extractor import PanelData
from .panel_extractor import PanelExtractor
from .panel_segmentation import main as basic_panel_segmentation

from typing import List, Tuple
from pathlib import Path
import numpy as np
import json
import shutil

class ComicPanelExtractor:
    """Main class that orchestrates the comic panel extraction process."""
    
    def __init__(self, config: Config, reset: bool = True):
        self.config = config
        if reset:
            if Path(self.config.output_folder).exists():
                shutil.rmtree(self.config.output_folder)
            Path(self.config.output_folder).mkdir(exist_ok=True)
        
        self.image_processor = ImageProcessor(self.config)
        self.panel_extractor = PanelExtractor(self.config)
    
    def extract_panels_from_comic(self) -> Tuple[List[np.ndarray], List[PanelData]]:
        """Complete pipeline to extract panels from a comic image."""
        print(f"Starting panel extraction for: {self.config.input_path}")

        processed_image_path = basic_panel_segmentation(self.config.output_folder, self.config.input_path, self.config.input_path)
        self.config.black_overlay_input_path = processed_image_path

        _, _, processed_image_path = self.image_processor.preprocess_image(processed_image_path)

        processed_image_path = self.image_processor.thin_image_borders(processed_image_path)

        processed_image_path = self.image_processor.remove_dangling_lines(processed_image_path)

        processed_image_path = self.image_processor.remove_diagonal_only_cells(processed_image_path)

        processed_image_path = self.image_processor.remove_small_continuity_components(processed_image_path)

        processed_image_path = self.image_processor.thick_black(processed_image_path)

        processed_image_path = self.image_processor.remove_small_regions(processed_image_path)

        processed_image_path = self.image_processor.remove_diagonal_lines(processed_image_path)

        processed_image_path = self.image_processor.remove_small_regions(processed_image_path)

        processed_image_path = self.image_processor.connect_horizontal_vertical_gaps(processed_image_path)

        processed_image_path = self.image_processor.thin_image_borders(processed_image_path)

        panel_images, panel_data, all_panel_path = self.panel_extractor.extract_panels(
            processed_image_path
        )
        
        return panel_images, panel_data, all_panel_path
    
    def cleanup(self):
        """Clean up temporary files if needed."""
        # Add cleanup logic here if needed
        pass