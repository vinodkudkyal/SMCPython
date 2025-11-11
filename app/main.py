import io
import os
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .face_service import FaceService

app = FastAPI(title="Face API", version="1.0.0")

app = FastAPI(title="Face API", version="1.0.0", redirect_slashes=True)
# CORS: allow mobile/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



service = FaceService(threshold=0.6)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/register")
async def register(
    sweeperId: str = Form(...),
    name: str = Form(...),
    images: List[UploadFile] = File(...)
):
    pil_images: List[Image.Image] = []
    for uf in images:
        content = await uf.read()
        img = Image.open(io.BytesIO(content))
        pil_images.append(img)
    res = service.register(sweeperId=sweeperId, name=name, images=pil_images)
    return res

@app.post("/recognize")
async def recognize(
    image: UploadFile = File(...),
    sweeperId: Optional[str] = Form(None)
):
    content = await image.read()
    img = Image.open(io.BytesIO(content))
    res = service.recognize(img, sweeperId=sweeperId)
    return res