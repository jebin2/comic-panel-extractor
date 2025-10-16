from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .extractor_server import app as extractor_app, delete_folder_if_old_or_empty, output_folder
from .annorator_server import app as annotator_app
import os, json
from .config import Config, load_config

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

fast_api = FastAPI()
config = load_config()

# Mount static files ONCE
static_folder = os.path.join(config.current_path, "static")
fast_api.mount("/static", StaticFiles(directory=static_folder), name="static")

fast_api.include_router(extractor_app)
fast_api.include_router(annotator_app)


# Add CORS middleware
fast_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
template_dirs = [static_folder]
env = Environment(
    loader=FileSystemLoader(template_dirs),
    autoescape=select_autoescape(['html', 'xml'])
)

# Routes
@fast_api.get("/", response_class=HTMLResponse)
async def index(request: Request):
    delete_folder_if_old_or_empty(output_folder)
    template = env.get_template("index.html")  # From tool/
    html_content = template.render(request=request)
    return HTMLResponse(content=html_content)

@fast_api.get("/annotate", response_class=HTMLResponse)
async def index(request: Request):
    template = env.get_template("annotator.html")  # From tool/
    html_content = template.render(request=request)
    return HTMLResponse(content=html_content)

def main():
    import uvicorn
    uvicorn.run(
        fast_api,
        host="0.0.0.0",  # Or "0.0.0.0" to allow access from other machines
        port=7860,         # Change to any available port, e.g., 8080
        # reload=True        # Enables auto-reload for development (like --reload in CLI)
    )

if __name__ == "__main__":
    main()
