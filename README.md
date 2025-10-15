---
title: Comic Panel Extractor
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

https://jebin2-comic-panel-extractor.hf.space/

# 📚 Comic Panel Extractor

Automatically extract panels from comic pages using YOLO segmentation and image processing.
Meanwhile currently using - "https://huggingface.co/mosesb/best-comic-panel-detection/resolve/main/best.pt"

## Installation

```bash
git clone https://github.com/jebin2/comic-panel-extractor.git
cd comic-panel-extractor
pip install -e .
```

## Usage

### Extract Panels

**Web Interface:**
```bash
serve-comic-panel-extractor
# Visit http://localhost:7860
```

**CLI:**
```bash
comic-panel-extractor path/to/comic.jpg
```

**Python:**
```python
from comic_panel_extractor.main import ComicPanelExtractor
from comic_panel_extractor.config import Config

config = Config()
config.input_path = "comic.jpg"
extractor = ComicPanelExtractor(config)
panels, data, paths = extractor.extract_panels_from_comic()
```

### Annotate Data

Visit `http://localhost:7860/annotate`

**Shortcuts:**
- Click/drag for boxes
- `S` = Save
- `D` = Auto-detect
- `Delete` = Remove

### Train Model

**1. Setup Dataset:**
```bash
# Add to .env
SOURCE_PATH=/path/to/images

# Create dataset (80/10/10 split)
python -m comic_panel_extractor.create_dataset
```

**2. Configure Training (.env):**
```env
EPOCH=200
YOLO_BASE_MODEL_NAME=yolo11s-seg
YOLO_MODEL_NAME=comic_panel
```

**3. Train:**
```bash
python -m comic_panel_extractor.train
```

### Run Inference

```bash
python -m comic_panel_extractor.inference
```

```python
from comic_panel_extractor.yolo_manager import YOLOManager

with YOLOManager() as yolo:
    yolo.load_model('weights.pt')
    yolo.annotate_images(['image.jpg'], 'output')
```

## Configuration

```python
config = Config()
config.min_width_ratio = 0.15    # Min panel width (% of image)
config.min_height_ratio = 0.15   # Min panel height (% of image)
config.min_area_ratio = 0.05     # Min panel area (% of image)
```

## Docker

```bash
docker build -t comic-panel-extractor .
docker run -p 7860:7860 comic-panel-extractor
```

## API Endpoints

**Extract:**
- `POST /api/extract/convert` - Upload & extract panels

**Annotate:**
- `GET /api/annotate/images` - List images
- `GET /api/annotate/annotations/{image}` - Get annotations
- `GET /api/annotate/detect_annotations/{image}` - Auto-detect
- `POST /api/annotate/annotations` - Save annotations

## Author

Jebin Einstein E - jebineinstein@gmail.com