# Comic Panel Extractor

A web application that automatically extracts individual panels from comic book pages.

## Features

- Drag & drop or click to upload comic images
- Automatic panel detection and extraction
- Preview images in full-screen modal
- Download individual panels
- Mobile-friendly responsive design

## How to Run

1. Set up a backend server with these endpoints:
   - `POST /convert` - Process images and extract panels
   - `DELETE /clear` - Clear stored panels

2. Open `index.html` in a web browser

## How to Use

1. Upload a comic page image (drag & drop or click)
2. Wait for automatic panel extraction
3. View extracted panels in the grid
4. Click panels to view full-screen
5. Download individual panels as needed