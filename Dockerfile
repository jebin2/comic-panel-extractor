# Use official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user

# Switch to user early
USER user

# Set working directory
WORKDIR /home/user/app

# Set environment variables for Hugging Face Spaces
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    BASE_PATH=/home/user/app \
    IS_DOCKER=True \
    SERVER_PORT=7860

# COPY --chown=user requirements.txt .
# RUN pip install --no-cache-dir --user -r requirements.txt

# Copy app code
COPY --chown=user . .

# Install the package in editable mode
RUN pip install --no-cache-dir --user -e .

# Entry point
CMD ["uvicorn", "comic_panel_extractor.server:main", "--host", "0.0.0.0", "--port", "7860"]