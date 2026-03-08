# Crop Disease Detector for Render

This package is ready to upload to GitHub and deploy on Render.

## What it does
- Uses your uploaded TensorFlow Lite models on the server
- Includes all 17 disease labels already
- Opens camera in the browser
- Supports predict once and realtime classification

## Important limitation
Your current model is a **classifier**, not an object detector.
It predicts the class for the **whole image/frame**.
It does **not** output bounding boxes.

## Deploy on Render
1. Create a new GitHub repository.
2. Upload all files from this folder to the repo root.
3. In Render, click **New +** -> **Web Service**.
4. Connect the GitHub repo.
5. Render should detect the included `Dockerfile` automatically.
6. Deploy.
7. Open the Render URL and start the camera.

## Local test
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Files
- `app.py` - FastAPI backend
- `static/` - frontend UI
- `models/` - your `.tflite` files
- `requirements.txt` - Python packages
- `Dockerfile` - Render build file
- `render.yaml` - optional Render config
