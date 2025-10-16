from dataclasses import dataclass
import os
import toml
from dotenv import load_dotenv

load_dotenv()

CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
CONFIG_FILE = os.path.join(CURRENT_PATH, "config.toml")


@dataclass
class Config:
    """Configuration settings for the comic-to-video pipeline."""

    # Paths
    current_path: str = CURRENT_PATH
    config_path: str = CONFIG_FILE

    # Core settings
    EPOCH: int = 200
    DEFAULT_IMAGE_SIZE: int = 640
    BATCH: int = 10
    RESUME_TRAIN: bool = True
    RECREATE_DATASET: bool = True

    # YOLO models
    YOLO_BASE_MODEL_NAME: str = "yolo11s-seg"
    YOLO_MODEL_NAME: str = ""  # will be derived if empty
    IMAGE_SOURCE_PATH: str = ""
    YOLO_MODEL_REMOTE_URL: str = ""

    # Derived paths
    yolo_base_model_path: str = ""
    yolo_trained_model_path: str = ""

    # Pipeline parameters
    org_input_path: str = ""
    input_path: str = ""
    black_overlay_input_path: str = ""
    output_folder: str = "temp_dir"
    distance_threshold: int = 70
    vertical_threshold: int = 30
    text_cood_file_name: str = "detect_and_group_text.json"
    min_text_length: int = 2
    min_area_ratio: float = 0.05
    min_width_ratio: float = 0.15
    min_height_ratio: float = 0.15

    # BorderPanelExtractor
    panel_filename_pattern: str = r"panel_\d+_\((\d+), (\d+), (\d+), (\d+)\)\.jpg"

    # Constants
    SUPPORTED_EXTENSIONS: tuple = ('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')

    def __post_init__(self):
        # Ensure absolute IMAGE_SOURCE_PATH
        if self.IMAGE_SOURCE_PATH:
            if not os.path.isabs(self.IMAGE_SOURCE_PATH):
                self.IMAGE_SOURCE_PATH = os.path.join(self.current_path, self.IMAGE_SOURCE_PATH)

        # Derive YOLO_MODEL_NAME if empty
        if not self.YOLO_MODEL_NAME:
            self.YOLO_MODEL_NAME = f"comic_panel_{self.YOLO_BASE_MODEL_NAME}"

        # Derived paths
        self.yolo_base_model_path = os.path.join(self.current_path, f"{self.YOLO_BASE_MODEL_NAME}.pt")
        self.yolo_trained_model_path = os.path.join(self.current_path, f"{self.YOLO_MODEL_NAME}.pt")


def load_config(file_path=CONFIG_FILE) -> Config:
    """Load the latest config from TOML file and return a Config instance."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    data = toml.load(file_path)

    # Convert boolean strings to actual bool
    def to_bool(val):
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("1", "true", "yes")

    return Config(
        EPOCH=int(data.get("EPOCH", 200)),
        DEFAULT_IMAGE_SIZE=int(data.get("DEFAULT_IMAGE_SIZE", 640)),
        BATCH=int(data.get("BATCH", 10)),
        RESUME_TRAIN=to_bool(data.get("RESUME_TRAIN", True)),
        RECREATE_DATASET=to_bool(data.get("RECREATE_DATASET", True)),
        YOLO_BASE_MODEL_NAME=data.get("YOLO_BASE_MODEL_NAME", "yolo11s-seg"),
        YOLO_MODEL_NAME=data.get("YOLO_MODEL_NAME", ""),  # derived in __post_init__
        IMAGE_SOURCE_PATH=data.get("IMAGE_SOURCE_PATH", ""),
        YOLO_MODEL_REMOTE_URL=data.get("YOLO_MODEL_REMOTE_URL", "")
    )


def update_toml_key(key: str, value, file_path=CONFIG_FILE) -> Config:
    """Update a key in the TOML file and reload config."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    data = toml.load(file_path)
    data[key] = value
    with open(file_path, "w") as f:
        toml.dump(data, f)

    # Reload and return new Config
    return load_config(file_path)


def get_text_cood_file_path(config: Config) -> str:
    """Return full path to text coordinate file."""
    return os.path.join(config.output_folder, config.text_cood_file_name)


# Example usage:
if __name__ == "__main__":
    # Load config
    config = load_config()
    print("EPOCH:", config.EPOCH)

    # Update TOML key and reload
    config = update_toml_key("EPOCH", 500)
    print("Updated EPOCH:", config.EPOCH)

    # Get text coord file path
    print("Text coord path:", get_text_cood_file_path(config))
