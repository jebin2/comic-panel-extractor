from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from .ws_manager import manager
from pydantic import BaseModel, field_validator
from typing import List
from PIL import Image, UnidentifiedImageError
import os
import base64
from io import BytesIO
import shutil
from .config import Config
from typing import List, Optional, Union, Dict, Any
from . import utils
import copy
import traceback
import asyncio
import sys, signal
import psutil
import subprocess
from . import common
import fcntl

app = APIRouter()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any websocket messages if needed
    except WebSocketDisconnect:
        print("Client disconnected:", websocket.client)
        manager.disconnect(websocket)

# === Configuration ===
IMAGE_ROOT = os.path.join(Config.current_path, "images")
IMAGE_LABEL_ROOT = os.path.join(Config.current_path, "image_labels")

CLASS_ID = 0

# === Pydantic Models ===
class Point(BaseModel):
    x: float
    y: float

class Box(BaseModel):
    type: str = "bbox"  # "bbox" or "segmentation"
    # For bbox
    left: Optional[int] = None
    top: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    # For segmentation
    points: Optional[List[Point]] = None
    # Common fields
    classId: int = CLASS_ID
    stroke: str = "#00ff00"
    strokeWidth: int = 3
    fill: str = "rgba(0, 255, 0, 0.2)"
    saved: bool = True

    @field_validator("left", "top", "width", "height", mode="before")
    def round_floats(cls, v):
        return round(v) if v is not None else None

class SaveAnnotationsRequest(BaseModel):
    annotations: List[Box]  # Changed from 'boxes' to 'annotations'
    image_name: str
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
    return os.path.join(IMAGE_LABEL_ROOT, os.path.splitext(image_name)[0] + ".txt")

# === Core Functions ===
def load_yolo_annotations(image_path: str, label_path: str, detect: bool = False):
    """Load both bbox and segmentation annotations from YOLO format"""
    try:
        img = Image.open(image_path)
        w, h = img.size
        annotations = []

        # Auto-detect if needed
        normalise = False
        if detect and not os.path.exists(label_path):
            from .yolo_manager import YOLOManager
            with YOLOManager() as yolo_manager:
                weights_path = Config.yolo_trained_model_path
                yolo_manager.load_model(weights_path)
                yolo_manager.annotate_images(
                    image_paths=[image_path],
                    output_dir=IMAGE_LABEL_ROOT,
                    save_image=False,
                    label_path=label_path
                )
                normalise = True

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])

                    if len(parts) == 5:  # Bounding box format
                        _, xc, yc, bw, bh = parts
                        left = int((xc - bw / 2) * w)
                        top = int((yc - bh / 2) * h)
                        width = int(bw * w)
                        height = int(bh * h)

                        annotations.append({
                            "type": "bbox",
                            "left": left,
                            "top": top,
                            "width": width,
                            "height": height,
                            "classId": class_id,
                            "stroke": "#00ff00",
                            "strokeWidth": 3,
                            "fill": "rgba(0, 255, 0, 0.2)",
                            "saved": True
                        })

                    elif len(parts) > 5 and len(parts) % 2 == 1:  # Segmentation format
                        # Skip class_id, then pairs of x,y coordinates
                        coords = parts[1:]
                        if len(coords) >= 6:  # At least 3 points
                            points = []
                            for i in range(0, len(coords), 2):
                                if i + 1 < len(coords):
                                    x = coords[i] * w
                                    y = coords[i + 1] * h
                                    points.append({"x": x, "y": y})

                            annotations.append({
                                "type": "segmentation",
                                "points": points,
                                "classId": class_id,
                                "stroke": "#00ff00",
                                "strokeWidth": 3,
                                "fill": "rgba(0, 255, 0, 0.2)",
                                "saved": True
                            })
            if normalise:
                annotations = utils.normalize_segmentation(annotations)
                save_yolo_annotations(
                    copy.deepcopy(annotations),
                    (w, h),
                    label_path
                )
        return annotations, (w, h)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading annotations: {str(e)} {traceback.format_exc()}")

def normalize_annotations(annotations: List[Union[Box, dict]]) -> List[Box]:
    """Convert all annotations to Box objects."""
    normalized = []
    for ann in annotations:
        if isinstance(ann, Box):
            normalized.append(ann)
        elif isinstance(ann, dict):
            normalized.append(Box(**ann))
        else:
            raise TypeError(f"Unsupported annotation type: {type(ann)}")
    return normalized

def save_yolo_annotations(annotations: List[Box], original_size: tuple, label_path: str):
    """Save annotations in YOLO format (both bbox and segmentation)"""
    annotations = normalize_annotations(annotations)
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    w, h = original_size

    try:
        with open(label_path, "w") as f:
            # Generate YOLO format from annotations
            for annotation in annotations:
                if annotation.type == "bbox":
                    left, top, width, height = annotation.left, annotation.top, annotation.width, annotation.height
                    xc = (left + width / 2) / w
                    yc = (top + height / 2) / h
                    bw = width / w
                    bh = height / h
                    f.write(f"{annotation.classId} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                elif annotation.type == "segmentation" and annotation.points:
                    # Convert points to normalized coordinates
                    normalized_points = []
                    for point in annotation.points:
                        normalized_points.extend([point.x / w, point.y / h])

                    coords_str = " ".join(f"{coord:.6f}" for coord in normalized_points)
                    f.write(f"{annotation.classId} {coords_str}\n")

        # Copy to image_labels directory
        shutil.copy2(label_path, f"{IMAGE_LABEL_ROOT}/{os.path.basename(label_path)}")
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving annotations: {str(e)} {traceback.format_exc()}")

def parse_yolo_line(line: str, image_width: int, image_height: int) -> Dict[str, Any]:
    """Parse a single YOLO format line and return annotation dict"""
    parts = list(map(float, line.strip().split()))
    if len(parts) < 5:
        return None

    class_id = int(parts[0])

    if len(parts) == 5:  # Bounding box
        _, xc, yc, bw, bh = parts
        left = int((xc - bw / 2) * image_width)
        top = int((yc - bh / 2) * image_height)
        width = int(bw * image_width)
        height = int(bh * image_height)

        return {
            "type": "bbox",
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "classId": class_id,
            "stroke": "#00ff00",
            "strokeWidth": 3,
            "fill": "rgba(0, 255, 0, 0.2)",
            "saved": True
        }

    elif len(parts) > 5 and len(parts) % 2 == 1:  # Segmentation
        coords = parts[1:]
        if len(coords) >= 6:  # At least 3 points
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i] * image_width
                    y = coords[i + 1] * image_height
                    points.append({"x": x, "y": y})

            return {
                "type": "segmentation",
                "points": points,
                "classId": class_id,
                "stroke": "#00ff00",
                "strokeWidth": 3,
                "fill": "rgba(0, 255, 0, 0.2)",
                "saved": True
            }

    return None

# === API Routes ===

@app.get("/api/annotate/images", response_model=List[ImageInfo])
async def list_all_images():
    image_info_list = []
    for root, _, files in os.walk(IMAGE_ROOT):
        for file in sorted(files):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
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
                except UnidentifiedImageError:
                    print(f"Cannot identify image file: {image_path}")

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

    annotations, (width, height) = load_yolo_annotations(image_path, label_path)

    return {
        "annotations": annotations,
        "original_width": width,
        "original_height": height
    }

@app.get("/api/annotate/detect_annotations/{image_name:path}")
async def get_detected_annotations(image_name: str):
    image_path = get_image_path(image_name)
    label_path = get_label_path(image_name)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    annotations, (width, height) = load_yolo_annotations(image_path, label_path, True)
    return {
        "annotations": annotations,
        "original_width": width,
        "original_height": height
    }

@app.post("/api/annotate/annotations")
async def save_annotations(request: SaveAnnotationsRequest):
    label_path = get_label_path(request.image_name)
    success = save_yolo_annotations(
        request.annotations,
        (request.original_width, request.original_height),
        label_path
    )
    return {"message": f"Saved {len(request.annotations)} annotations successfully"}

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

    file_path = os.path.join(Config.IMAGE_SOURCE_PATH, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"Uploaded {file.filename} to train set"}

####################### ----train---- #############################


current_process = {}

def reset_current_process():
    global current_process
    current_process = {
        "process": None
    }

reset_current_process()

# Define a function to handle cleanup
def handle_exit(signal_received, frame):
    if current_process["process"]:
        os.killpg(os.getpgid(current_process['process'].pid), signal.SIGKILL)
    sys.exit(0)

# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, handle_exit)

@app.get("/api/annotate/train")
async def upload_image(recreate_dataset: bool = False):
    os.environ['PYTHONUNBUFFERED'] = "1"
    # Skip if the training process is already running
    if is_process_running("comic_panel_extractor.train"):
        return {"status": "ignored", "message": "Training already in progress."}
    reset_current_process()
    cmd_to_run=""
    if recreate_dataset:
        cmd_to_run = "python -m comic_panel_extractor.create_dataset && "
    cmd_to_run += "python -m comic_panel_extractor.train"

    async def run_and_stream_output():
        process = None
        try:
            process = subprocess.Popen(
                cmd_to_run,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid,
		        env={**os.environ, 'PYTHONUNBUFFERED': '1', 'CUDA_LAUNCH_BLOCKING': '1', 'USE_CPU_IF_POSSIBLE': str(common.get_device() == "cpu")}
            )
            
            # Set non-blocking I/O
            fd = process.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            current_process['process'] = process

            # Stream the output and send it via WebSocket in real-time
            while True:
                try:
                    output = process.stdout.readline()
                    if output:
                        print(output.strip())
                        print("Active connections:", len(manager.active_connections))
                        asyncio.create_task(manager.broadcast({
                            'type': 'command_output',
                            'data': output.strip()
                        }))
                        sys.stdout.flush()
                    
                    if process.poll() is not None:
                        break
                        
                    # Small delay to prevent CPU spinning
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    print(f"Error reading process output: {e}")
                    break

            # Process finished
            return_code = process.returncode if process else -1
            asyncio.create_task(manager.broadcast({
                'type': 'command_finished',
                'return_code': return_code
            }))
            
        except Exception as e:
            print(f"Error in run_and_stream_output: {e}")
            asyncio.create_task(manager.broadcast({
                'type': 'command_error',
                'error': str(e)
            }))
        finally:
            current_process['process'] = None

    # Start the command execution in a separate task
    asyncio.create_task(run_and_stream_output())
    return {"message": "Command started!", "status": "started"}


@app.get("/api/annotate/stopTrain")
async def stop_train():
    try:
        # Check if there's actually a process to stop
        if current_process['process'] is None:
            return {'message': 'No command is currently running.', 'status': 'no_process'}

        # Check if process has already terminated naturally
        if current_process['process'].poll() is not None:
            # Process already finished, just clean up
            reset_current_process()
            return {'message': 'Command has already finished.', 'status': 'already_finished'}

        try:
            # Get the process group ID before attempting to kill
            pgid = os.getpgid(current_process['process'].pid)
            
            # Kill the entire process group
            os.killpg(pgid, signal.SIGTERM)  # Try SIGTERM first
            
            # Wait a bit for graceful shutdown
            await asyncio.sleep(1)

            # If still running, force kill
            if current_process['process'] and current_process['process'].poll() is None:
                os.killpg(pgid, signal.SIGKILL)

        except ProcessLookupError:
            # Process already dead
            print("Process already terminated")
        except OSError as e:
            # Handle permission errors or other OS-level issues
            print(f"Error terminating process: {e}")
            # Try to kill just the main process if group kill fails
            try:
                current_process['process'].terminate()
                await asyncio.sleep(0.5)
                if current_process['process'].poll() is None:
                    current_process['process'].kill()
            except:
                pass
        
        # Always reset the process state
        reset_current_process()
        
        # Notify connected clients
        await manager.broadcast({
            'type': 'command_stopped',
            'message': 'Command terminated by user'
        })
        
        return {'message': 'Command terminated successfully.', 'status': 'terminated'}
        
    except Exception as e:
        print(f"Error in stop_command: {str(e)}")
        # Force reset even if there was an error
        reset_current_process()
        raise HTTPException(status_code=500, detail=f'Error stopping command: {str(e)}')


def is_process_running(name: str) -> bool:
    """
    Check if a process containing 'name' in its command line is running.
    """
    for proc in psutil.process_iter(['cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
            if name in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False