from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
from .config import load_config
from .main import ComicPanelExtractor
import traceback
from pathlib import Path
import shutil
import time
import mimetypes

config = load_config()

base_output_folder = "api_outputs"
output_folder = os.path.join(config.current_path, base_output_folder)

app = APIRouter()

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

@app.post("/api/extract/convert")
async def convert_comic(file: UploadFile = File(...)):
    """
    Upload a comic page and extract panels
    """
    # Generate unique filename
    file_id = os.path.splitext(file.filename)[0]
    specific_output_folder = f'{output_folder}/{file_id}'

    shutil.rmtree(specific_output_folder, ignore_errors=True)
    Path(specific_output_folder).mkdir(parents=True, exist_ok=True)
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
        all_panel_path = [f'/api/extract/{base_output_folder}/{file_id}/{os.path.basename(path)}' for path in all_panel_path]

        return {
            "success": True,
            "message": f"Extracted {len(all_panel_path)} panels",
            "panels": all_panel_path
        }
        
    except Exception as e:
        print(f"Error processing image: {str(e)} {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)} {traceback.format_exc()}")

@app.get("/api/extract/api_outputs/{folder}/{filename}")
async def get_output_file(folder: str, filename: str):
    file_path = f'{output_folder}/{folder}/{filename}'
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    mime_type, _ = mimetypes.guess_type(file_path)
    return FileResponse(file_path, media_type=mime_type, filename=filename)