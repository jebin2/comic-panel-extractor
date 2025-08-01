from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
from typing import List
import uuid
from .config import Config
from .main import ComicPanelExtractor
from jinja2 import Environment, FileSystemLoader, select_autoescape
import traceback
from pathlib import Path
import shutil
import time
import mimetypes

current_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

base_output_folder = "api_outputs"
static_folder = "./static"
output_folder = os.path.join(current_path, base_output_folder)
static_folder = os.path.join(current_path, static_folder)

# Create directories for uploads and outputs
os.makedirs(output_folder, exist_ok=True)
os.makedirs(static_folder, exist_ok=True)

# Templates
template_dirs = [static_folder]
env = Environment(
    loader=FileSystemLoader(template_dirs),
    autoescape=select_autoescape(['html', 'xml'])
)

app = FastAPI(title="Comic Panel Extractor", version="1.0.0")

# Mount static files
app.mount(static_folder, StaticFiles(directory=static_folder), name="static")
# app.mount(output_folder, StaticFiles(directory=output_folder), name="api_outputs")

def delete_folder_if_old_or_empty(parent_folder, age_days=1):
    """
    Delete subfolders inside `parent_folder` if they are empty
    or older than `age_days`.

    Args:
        parent_folder (str): Path to the parent directory.
        age_days (int): Number of days before a folder is considered old.
    """
    try:
        current_time = time.time()
        age_seconds = age_days * 24 * 60 * 60

        # Loop through all items in the parent folder
        for entry in os.scandir(parent_folder):
            if entry.is_dir():
                folder_path = entry.path
                # Check if folder is empty
                if not os.listdir(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Deleted empty folder: {folder_path}")
                    continue

                # Check if folder is older than age_days
                folder_mtime = os.path.getmtime(folder_path)
                if current_time - folder_mtime > age_seconds:
                    shutil.rmtree(folder_path)
                    print(f"Deleted old folder (>{age_days} day): {folder_path}")

    except Exception as e:
        print(f"Error cleaning subfolders in {parent_folder}: {e}")

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    delete_folder_if_old_or_empty(output_folder)
    template = env.get_template("index.html")  # From tool/
    html_content = template.render(request=request)
    return HTMLResponse(content=html_content)

@app.post("/convert")
async def convert_comic(file: UploadFile = File(...)):
    """
    Upload a comic page and extract panels
    """
    # Generate unique filename
    file_id = os.path.splitext(file.filename)[0]
    specific_output_folder = f'{output_folder}/{file_id}'

    shutil.rmtree(specific_output_folder, ignore_errors=True)
    Path(specific_output_folder).mkdir(exist_ok=True)
    file_path = f'{specific_output_folder}/{file.filename}'
    
    # Save uploaded file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # 🔍 DEBUG: Log file info
        print("======== DEBUG: Upload Info ========")
        print(f"Working Dir: {os.getcwd()}")
        print(f"Saved file path: {file_path}")
        print(f"Output folder: {specific_output_folder}")
        print(f"List of files in output folder: {os.listdir(specific_output_folder)}")
        print("====================================")

        # Extract panels
        config = Config()
        config.input_path = file_path
        config.output_folder = specific_output_folder

        print(f"[DEBUG] Setting config.input_path to: {config.input_path}")
        print(f"[DEBUG] Setting config.output_folder to: {config.output_folder}")

        _, _, all_panel_path = ComicPanelExtractor(config, reset=False).extract_panels_from_comic()
        all_panel_path = [f'/{"/".join(path.split("/")[-3:])}' for path in all_panel_path]

        return {
            "success": True,
            "message": f"Extracted {len(all_panel_path)} panels",
            "panels": all_panel_path
        }
        
    except Exception as e:
        print(f"Error processing image: {str(e)} {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)} {traceback.format_exc()}")

@app.get("/api_outputs/{folder}/{filename}")
async def get_output_file(folder: str, filename: str):
    file_path = f'{output_folder}/{folder}/{filename}'
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    mime_type, _ = mimetypes.guess_type(file_path)
    return FileResponse(file_path, media_type=mime_type, filename=filename)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()