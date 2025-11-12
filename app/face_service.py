import io
import os
import threading
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from .db import face_embeddings

logger = logging.getLogger(__name__)


def to_rgb_pil(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms


class FaceService:
    def __init__(self, threshold: float = 0.7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mtcnn = MTCNN(keep_all=True, image_size=160, margin=20, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.threshold = threshold
        self._lock = threading.Lock()

    def _embed(self, pil_img: Image.Image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            pil_img = to_rgb_pil(pil_img)
            faces = self.mtcnn(pil_img)  # [N,3,160,160] or None
            boxes, _ = self.mtcnn.detect(pil_img)  # [N,4] or None
            if faces is None or boxes is None:
                return None, None
            with torch.no_grad():
                emb = self.resnet(faces.to(self.device)).cpu().numpy().astype(np.float32)
            return boxes, emb
        except Exception as e:
            logger.exception("Embedding failed: %s", e)
            return None, None

    def register(self, sweeperId: str, name: str, images: List[Image.Image]) -> Dict[str, Any]:
        if not images:
            return {"registered": 0, "message": "No images provided."}

        collected: List[np.ndarray] = []
        try:
            for img in images:
                boxes, emb = self._embed(img)
                if emb is None or len(emb) == 0:
                    continue
                # choose largest face
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = int(np.argmax(areas))
                collected.append(emb[idx])

            if not collected:
                return {"registered": 0, "message": "No faces detected in provided images."}

            timestamp = datetime.utcnow().isoformat() + "Z"
            with self._lock:
                col = face_embeddings()
                # upsert one document per person
                doc = col.find_one({"sweeperId": sweeperId, "name": name})
                collected_list = [x.astype(np.float32).tolist() for x in collected]
                if doc:
                    col.update_one(
                        {"_id": doc["_id"]},
                        {
                            "$push": {"embeddings": {"$each": collected_list}},
                            "$set": {"updatedAt": timestamp}
                        }
                    )
                else:
                    col.insert_one({
                        "sweeperId": sweeperId,
                        "name": name,
                        "embeddings": collected_list,
                        "createdAt": timestamp
                    })

            return {"registered": len(collected), "name": name, "sweeperId": sweeperId}
        except Exception as e:
            logger.exception("Registration error for %s/%s: %s", sweeperId, name, e)
            return {"registered": 0, "message": f"Registration error: {str(e)}"}

    def recognize(self, image: Image.Image, sweeperId: Optional[str] = None, top_k: int = 1) -> Dict[str, Any]:
        try:
            boxes, emb = self._embed(image)
            if emb is None or len(emb) == 0:
                # No face detected
                return {
                    "detections": [],
                    "best": {"identity": "Unknown", "confidence": 0.0}
                }

            # Only largest face for now
            if boxes is not None and len(boxes) > 0:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = int(np.argmax(areas))
                query = emb[idx:idx+1]  # [1,512]
                box = boxes[idx]
            else:
                query = emb[:1]
                box = [0, 0, 0, 0]

            # Load DB embeddings (filtered by sweeperId if provided)
            col = face_embeddings()
            cursor = col.find({"sweeperId": sweeperId}) if sweeperId else col.find({})

            names: List[str] = []
            embs: List[np.ndarray] = []
            for doc in cursor:
                person_name = doc.get("name", "Unknown")
                for e in (doc.get("embeddings") or []):
                    arr = np.asarray(e, dtype=np.float32)
                    if arr.shape and arr.shape[-1] == 512:
                        names.append(person_name)
                        embs.append(arr)

            if not embs:
                # No stored embeddings to compare against
                return {
                    "detections": [],
                    "best": {"identity": "Unknown", "confidence": 0.0}
                }

            db_mat = np.vstack(embs).astype(np.float32)
            db_mat = normalize_embeddings(db_mat)
            q = normalize_embeddings(query.astype(np.float32))  # [1,512]

            sims = np.dot(q, db_mat.T)[0]  # cosine similarity
            max_idx = int(np.argmax(sims))
            max_sim = float(sims[max_idx])
            identity = names[max_idx] if max_sim >= self.threshold else "Unknown"

            det = {
                "box": [int(x) for x in box],
                "identity": identity,
                "confidence": max_sim
            }
            return {
                "detections": [det],
                "best": {"identity": identity, "confidence": max_sim}
            }
        except Exception as e:
            logger.exception("Recognition error: %s", e)
            return {
                "detections": [],
                "best": {"identity": "Unknown", "confidence": 0.0},
                "error": str(e)
            }