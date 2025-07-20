from .text_detector import TextDetector
from .config import Config
from .image_processor import ImageProcessor
from .panel_extractor import PanelData
from .panel_extractor import PanelExtractor

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
        
        # Step 1: Detect and mask text regions
        text_bubbles = self._detect_text_bubbles()
        masked_image_path = self.image_processor.mask_text_regions([bubble["bbox"] for bubble in text_bubbles])
        
        # Step 2: Preprocess image
        _, _, processed_image_path = self.image_processor.preprocess_image(masked_image_path)

        # Step 3: Thin border line
        processed_image_path = self.image_processor.thin_image_borders(processed_image_path)
        # Step 3: Clean dilated image
        # processed_image_path = self.image_processor.clean_dilated_image(processed_image_path)
        
        # Step 4: Extract panels
        panel_images, panel_data, all_panel_path = self.panel_extractor.extract_panels(
            processed_image_path, min_width_ratio=0.1
        )
        
        return panel_images, panel_data, all_panel_path
    
    def _detect_text_bubbles(self) -> List[dict]:
        """Detect text bubbles in the comic image."""
        with TextDetector(self.config) as text_detector:
            bubbles_path = text_detector.detect_and_group_text()
        
        with open(bubbles_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def cleanup(self):
        """Clean up temporary files if needed."""
        # Add cleanup logic here if needed
        pass