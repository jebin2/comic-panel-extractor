from .main import ComicPanelExtractor
from .config import Config
from .text_detector import TextDetector, TextDetection
from .image_processor import ImageProcessor
from .panel_extractor import PanelExtractor, PanelData

__version__ = "0.1.0"
__all__ = [
    "ComicPanelExtractor",
    "Config", 
    "TextDetector",
    "TextDetection",
    "ImageProcessor",
    "PanelExtractor", 
    "PanelData"
]