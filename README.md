# Crop Disease Detector

A web app that uses your browser camera to classify crop leaf diseases in real
time. A FastAPI backend serves TensorFlow Lite models covering 17 classes across
five crops.

| Crop | Classes |
| --- | --- |
| Corn | Common rust, gray leaf spot, northern leaf blight, healthy |
| Potato | Early blight, late blight, healthy |
| Rice | Brown spot, leaf blast, neck blast, healthy |
| Wheat | Brown rust, yellow rust, healthy |
| Sugarcane | Red rot, bacterial blight, healthy |

## Features

- Live camera capture in the browser, with single-shot or continuous prediction.
- Float and int8-quantised TFLite models; the int8 model is smaller and faster on CPU.
- Ships as a Docker image, with a Render config included.

> The model is an image **classifier**: it labels the whole frame and does not
> draw bounding boxes.

## Run locally

```bash
cd crop-disease-render-final
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Or with Docker:

```bash
cd crop-disease-render-final
docker build -t crop-disease .
docker run -p 8000:8000 crop-disease
```

## Tech

Python, FastAPI, TensorFlow Lite, Pillow and NumPy, with a vanilla JS frontend
and Docker.
