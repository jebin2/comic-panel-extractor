# from .text_detector import TextDetector
from .config import Config, load_config
from .image_processor import ImageProcessor
from .panel_extractor import PanelData
from .panel_extractor import PanelExtractor
from .panel_segmentation import main as basic_panel_segmentation

from typing import List, Tuple
from pathlib import Path
import numpy as np
from .border_panel_extractor import BorderPanelExtractor
import shutil
from . import utils
import traceback

class ComicPanelExtractor:
    """Main class that orchestrates the comic panel extraction process."""
    
    def __init__(self, config: Config, reset: bool = True):
        self.config = config
        self.reset = reset
        if reset:
            if Path(self.config.output_folder).exists():
                shutil.rmtree(self.config.output_folder)
            Path(self.config.output_folder).mkdir(exist_ok=True)
        
        self.image_processor = ImageProcessor(self.config)
        self.panel_extractor = PanelExtractor(self.config)
    
    def extract_panels_from_comic(self) -> Tuple[List[np.ndarray], List[PanelData]]:
        """Complete pipeline to extract panels from a comic image."""
        print(f"Starting panel extraction for: {self.config.input_path}")
        try:
            # Get original image dimensions
            from PIL import Image
            with Image.open(self.config.input_path) as original_image:
                original_width, original_height = original_image.size
            from .llm_panel_extractor import extract_panel_via_llm
            all_path, detected_boxes, all_processed_boxes = extract_panel_via_llm(self.config.input_path, self.config, self.reset)
            print("LLM Done.")
            if utils.box_covered_ratio(all_processed_boxes, (original_width, original_height)) < 0.95:
                print("LLM failed.")
            return None, None, all_path
        except Exception as e:
            print(f'{str(e)} {traceback.format_exc()}')
            raise
    
    def cleanup(self):
        """Clean up temporary files if needed."""
        # Add cleanup logic here if needed
        pass