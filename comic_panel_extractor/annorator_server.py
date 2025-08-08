from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from typing import List
from PIL import Image
import os
import base64
from io import BytesIO
import shutil

current_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

app = APIRouter()

# === Configuration ===
IMAGE_ROOT = os.path.join(current_path, "dataset/images")
LABEL_ROOT = os.path.join(current_path, "dataset/labels")
IMAGE_LABEL_ROOT = os.path.join(current_path, "image_labels")

CLASS_ID = 0

# === Pydantic Models ===
class Box(BaseModel):
    left: int
    top: int
    width: int
    height: int
    type: str = "rect"
    stroke: str = "#00ff00"
    strokeWidth: int = 3
    fill: str = "rgba(0, 255, 0, 0.2)"
    saved: bool = True

    @field_validator("left", "top", "width", "height", mode="before")
    def round_floats(cls, v):
        return round(v)

class SaveAnnotationsRequest(BaseModel):
    boxes: List[Box]
    image_name: str  # Relative path like train/image1.jpg
    original_width: int
    original_height: int

class ImageInfo(BaseModel):
    name: str  # Relative path like train/image1.jpg
    width: int
    height: int
    has_annotations: bool

# === Helpers ===
def get_image_path(image_name: str) -> str:
    return os.path.join(IMAGE_ROOT, image_name)

def get_label_path(image_name: str) -> str:
    return os.path.join(LABEL_ROOT, os.path.splitext(image_name)[0] + ".txt")

# === Core Functions ===
def load_yolo_boxes(image_path: str, label_path: str):
    try:
        img = Image.open(image_path)
        w, h = img.size
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) != 5:
                        continue
                    _, xc, yc, bw, bh = parts
                    left = int((xc - bw / 2) * w)
                    top = int((yc - bh / 2) * h)
                    width = int(bw * w)
                    height = int(bh * h)
                    boxes.append({
                        "type": "rect",
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                        "stroke": "#00ff00",
                        "strokeWidth": 3,
                        "fill": "rgba(0, 255, 0, 0.2)",
                        "saved": True
                    })
        return boxes, (w, h)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

def save_yolo_annotations(boxes: List[Box], original_size: tuple, label_path: str):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    w, h = original_size
    try:
        with open(label_path, "w") as f:
            for box in boxes:
                left, top, width, height = box.left, box.top, box.width, box.height
                xc = (left + width / 2) / w
                yc = (top + height / 2) / h
                bw = width / w
                bh = height / h
                f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        shutil.copy2(label_path, f"{IMAGE_LABEL_ROOT}/{os.path.basename(label_path)}")
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving annotations: {str(e)}")

# === API Routes ===

@app.get("/api/annotate/images", response_model=List[ImageInfo])
async def list_all_images():
    image_info_list = []
    for root, _, files in os.walk(IMAGE_ROOT):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                rel_path = os.path.relpath(image_path, IMAGE_ROOT)
                label_path = get_label_path(rel_path)

                img = Image.open(image_path)
                width, height = img.size

                image_info_list.append(ImageInfo(
                    name=rel_path.replace("\\", "/"),
                    width=width,
                    height=height,
                    has_annotations=os.path.exists(label_path)
                ))
    return image_info_list

@app.get("/api/annotate/image/{image_name:path}")
async def get_image(image_name: str):
    image_path = get_image_path(image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return {
            "image_data": f"data:image/jpeg;base64,{img_data}",
            "width": img.width,
            "height": img.height
        }

@app.get("/api/annotate/annotations/{image_name:path}")
async def get_annotations(image_name: str):
    image_path = get_image_path(image_name)
    label_path = get_label_path(image_name)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    boxes, (width, height) = load_yolo_boxes(image_path, label_path)
    return {
        "boxes": boxes,
        "original_width": width,
        "original_height": height
    }

@app.post("/api/annotate/annotations")
async def save_annotations(request: SaveAnnotationsRequest):
    label_path = get_label_path(request.image_name)
    success = save_yolo_annotations(
        request.boxes,
        (request.original_width, request.original_height),
        label_path
    )
    return {"message": f"Saved {len(request.boxes)} annotations successfully"}

@app.delete("/api/annotate/annotations/{image_name:path}")
async def delete_annotations(image_name: str):
    label_path = get_label_path(image_name)
    if os.path.exists(label_path):
        os.remove(label_path)
        return {"message": "Annotations deleted"}
    return {"message": "No annotations to delete"}

@app.get("/api/annotate/annotations/{image_name:path}/download")
async def download_annotations(image_name: str):
    label_path = get_label_path(image_name)
    if not os.path.exists(label_path):
        raise HTTPException(status_code=404, detail="Annotations not found")
    return FileResponse(
        label_path,
        media_type="text/plain",
        filename=os.path.basename(label_path)
    )

@app.post("/api/annotate/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_path = os.path.join(IMAGE_ROOT, "train", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"Uploaded {file.filename} to train set"}