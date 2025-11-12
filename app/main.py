import io
import os
import threading
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .face_service import FaceService

app = FastAPI(title="Face API", version="1.0.0")

# CORS: allow mobile/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defer creating the heavy FaceService until first request (lazy instantiation)
_service_lock = threading.Lock()
_service: Optional[FaceService] = None


def get_service() -> FaceService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                # Create the service lazily; models inside FaceService are also lazy-loaded
                _service = FaceService(threshold=0.7)
    return _service


@app.get("/health")
def health():
    # Keep health cheap (don't create models)
    return {"ok": True}


@app.post("/register")
async def register(
    sweeperId: str = Form(...),
    name: str = Form(...),
    images: List[UploadFile] = File(...)
):
    pil_images: List[Image.Image] = []
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")
    for uf in images:
        try:
            content = await uf.read()
            img = Image.open(io.BytesIO(content))
            pil_images.append(img)
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"Invalid image uploaded: {uf.filename}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading image {uf.filename}: {e}")

    service = get_service()
    res = service.register(sweeperId=sweeperId, name=name, images=pil_images)
    return res


@app.post("/recognize")
async def recognize(
    image: UploadFile = File(...),
    sweeperId: Optional[str] = Form(None)
):
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail=f"Invalid image uploaded: {image.filename}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image {image.filename}: {e}")

    service = get_service()
    res = service.recognize(img, sweeperId=sweeperId)
    return res