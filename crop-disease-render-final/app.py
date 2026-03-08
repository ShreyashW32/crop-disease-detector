from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import os

LABELS = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust",
    "Sugarcane__Red_Rot",
    "Sugarcane__Healthy",
    "Sugarcane__Bacterial_Blight",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL = "crop_disease.tflite"

app = FastAPI(title="Crop Disease Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

_interpreters = {}


def get_interpreter(model_name: str):
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError(f"TensorFlow backend is not installed correctly: {e}")

    safe_name = os.path.basename(model_name or DEFAULT_MODEL)
    model_path = os.path.join(MODELS_DIR, safe_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {safe_name}")
    if safe_name not in _interpreters:
        interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=1)
        interpreter.allocate_tensors()
        _interpreters[safe_name] = interpreter
    return _interpreters[safe_name]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    denom = np.sum(e)
    return e / denom if denom != 0 else np.ones_like(e) / len(e)


def preprocess(image_bytes: bytes, input_shape, input_dtype):
    _, h, w, c = input_shape
    if c != 3:
        raise ValueError(f"Expected 3 channels, got {c}")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((w, h))
    arr = np.array(image)
    arr = np.expand_dims(arr, axis=0)

    if input_dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    elif input_dtype == np.int8:
        arr = arr.astype(np.int8)
    elif input_dtype == np.uint8:
        arr = arr.astype(np.uint8)
    elif input_dtype == np.int32:
        arr = arr.astype(np.int32)
    else:
        arr = arr.astype(input_dtype)
    return arr


def run_inference(image_bytes: bytes, model_name: str):
    interpreter = get_interpreter(model_name)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = preprocess(image_bytes, input_details["shape"], input_details["dtype"])
    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details["index"])[0]
    y = np.array(y).reshape(-1)

    # Normalize robustly for logits / raw scores / quantized outputs.
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    probs = softmax(y)

    ranked = np.argsort(-probs)
    top = []
    for idx in ranked[:5]:
        label = LABELS[idx] if idx < len(LABELS) else f"Class_{idx}"
        top.append({"label": label, "score": float(probs[idx])})

    return {
        "top_label": top[0]["label"],
        "confidence": top[0]["score"],
        "top_predictions": top,
        "input_dtype": str(input_details["dtype"]),
        "input_shape": list(map(int, input_details["shape"])),
        "model": os.path.basename(model_name),
        "labels": LABELS,
        "type": "classification"
    }


@app.get("/api/health")
def health():
    return {"ok": True, "models": os.listdir(MODELS_DIR), "labels": len(LABELS)}


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL)
):
    try:
        image_bytes = await image.read()
        result = run_inference(image_bytes, model_name)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
