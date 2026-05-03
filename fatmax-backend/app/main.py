from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import math

app = FastAPI()

# ── CORS ──────────────────────────────────────────────────────────────────────
# This allows your browser frontend to talk to this server.
# Without it, the browser would block all requests for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, lock this to your actual domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load YOLOv8 model ─────────────────────────────────────────────────────────
# YOLOv8n = nano, the smallest/fastest version.
# On first run this downloads the model weights (~6MB) automatically.
model = YOLO("yolov8n.pt")

# ── Reference object real-world dimensions (mm) ───────────────────────────────
# These are hardcoded constants — universally standardized sizes.
REFERENCE_DIMS = {
    "cell phone":    {"mmWidth": 146.7, "mmHeight": 71.5},  # iPhone 14 avg
    "book":          {"mmWidth": 297.0, "mmHeight": 210.0}, # A4 approx
    "keyboard":      {"mmWidth": 450.0, "mmHeight": 150.0},
    "remote":        {"mmWidth": 180.0, "mmHeight": 50.0},
    "mouse":         {"mmWidth": 120.0, "mmHeight": 65.0},
    "bottle":        {"mmWidth": 65.0,  "mmHeight": 230.0},
    "cup":           {"mmWidth": 95.0,  "mmHeight": 95.0},
    "scissors":      {"mmWidth": 200.0, "mmHeight": 70.0},
    "toothbrush":    {"mmWidth": 190.0, "mmHeight": 15.0},
    "banana":        {"mmWidth": 180.0, "mmHeight": 35.0},  # useful fallback
}

# ── Health check endpoint ─────────────────────────────────────────────────────
# A simple endpoint to confirm the server is running.
# Visit http://localhost:8000/health in your browser to test.
@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8n"}

# ── Detection endpoint ────────────────────────────────────────────────────────
@app.post("/detect-reference")
async def detect_reference(file: UploadFile = File(...)):
    """
    Receives a photo, runs YOLOv8, finds the best reference object,
    and returns its bounding box + real-world dimensions.
    """

    # 1. Read the uploaded image bytes into a PIL Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_width, img_height = image.size

    # 2. Run YOLOv8 detection
    results = model(image, verbose=False)

    # 3. Parse detections — find all objects we recognise as references
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name in REFERENCE_DIMS and confidence > 0.3:
                # bounding box in pixel coords [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "box": {
                        "x1": round(x1), "y1": round(y1),
                        "x2": round(x2), "y2": round(y2),
                    },
                    "dims": REFERENCE_DIMS[class_name],
                    "img_width": img_width,
                    "img_height": img_height,
                })

    # 4. If nothing found, return a clear message
    if not detections:
        return {
            "found": False,
            "message": "No reference object detected. Please mark manually.",
            "detections": []
        }

    # 5. Return the highest-confidence detection first
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    best = detections[0]

    # 6. Calculate px/mm scale from the best detection
    box = best["box"]
    px_width = box["x2"] - box["x1"]
    px_height = box["y2"] - box["y1"]
    mm_width = best["dims"]["mmWidth"]
    mm_height = best["dims"]["mmHeight"]

    # use whichever axis gives us more pixels = more accurate scale
    if px_width >= px_height:
        scale_mm_per_px = mm_width / px_width
    else:
        scale_mm_per_px = mm_height / px_height

    return {
        "found": True,
        "reference": best["class_name"],
        "confidence": best["confidence"],
        "box": best["box"],
        "dims": best["dims"],
        "scale_mm_per_px": round(scale_mm_per_px, 6),
        "img_width": img_width,
        "img_height": img_height,
        "all_detections": detections
    }