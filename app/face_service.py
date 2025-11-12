# import io
# import os
# import threading
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# from PIL import Image
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from .db import face_embeddings

# def to_rgb_pil(img: Image.Image) -> Image.Image:
#     return img.convert("RGB") if img.mode != "RGB" else img

# def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
#     return emb / norms

# class FaceService:
#     def __init__(self, threshold: float = 0.6):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.mtcnn = MTCNN(keep_all=True, image_size=160, margin=20, device=self.device)
#         self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
#         self.threshold = threshold
#         self._lock = threading.Lock()

#     def _embed(self, pil_img: Image.Image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
#         pil_img = to_rgb_pil(pil_img)
#         faces = self.mtcnn(pil_img)  # [N,3,160,160] or None
#         boxes, _ = self.mtcnn.detect(pil_img)  # [N,4] or None
#         if faces is None or boxes is None:
#             return None, None
#         with torch.no_grad():
#             emb = self.resnet(faces.to(self.device)).cpu().numpy().astype(np.float32)
#         return boxes, emb

#     def register(self, sweeperId: str, name: str, images: List[Image.Image]) -> Dict[str, Any]:
#         collected: List[np.ndarray] = []

#         for img in images:
#             boxes, emb = self._embed(img)
#             if emb is None or len(emb) == 0:
#                 continue
#             # choose largest face
#             areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#             idx = int(np.argmax(areas))
#             collected.append(emb[idx])

#         if not collected:
#             return {"registered": 0, "message": "No faces detected in provided images."}

#         with self._lock:
#             col = face_embeddings()
#             # upsert one document per person
#             doc = col.find_one({"sweeperId": sweeperId, "name": name})
#             collected_list = [x.astype(np.float32).tolist() for x in collected]
#             if doc:
#                 col.update_one(
#                     {"_id": doc["_id"]},
#                     {
#                         "$push": {"embeddings": {"$each": collected_list}},
#                         "$set": {"updatedAt": torch.tensor([]).new_tensor([]).cpu().numpy().tolist() if False else None}
#                     }
#                 )
#             else:
#                 col.insert_one({
#                     "sweeperId": sweeperId,
#                     "name": name,
#                     "embeddings": collected_list,
#                     "createdAt": np.datetime64("now").astype("datetime64[ms]").tolist()
#                 })

#         return {"registered": len(collected), "name": name, "sweeperId": sweeperId}

#     def recognize(self, image: Image.Image, sweeperId: Optional[str] = None, top_k: int = 1) -> Dict[str, Any]:
#         boxes, emb = self._embed(image)
#         if emb is None or len(emb) == 0:
#             return {"detections": []}

#         # Only largest face for now
#         areas = None
#         if boxes is not None and len(boxes) > 0:
#             areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#             idx = int(np.argmax(areas))
#             query = emb[idx:idx+1]  # [1,512]
#             box = boxes[idx]
#         else:
#             query = emb[:1]
#             box = [0,0,0,0]

#         # Load DB embeddings
#         col = face_embeddings()
#         if sweeperId:
#             cursor = col.find({"sweeperId": sweeperId})
#         else:
#             cursor = col.find({})

#         names: List[str] = []
#         embs: List[np.ndarray] = []
#         for doc in cursor:
#             person_name = doc["name"]
#             for e in (doc.get("embeddings") or []):
#                 arr = np.asarray(e, dtype=np.float32)
#                 if arr.shape[-1] == 512:
#                     names.append(person_name)
#                     embs.append(arr)

#         if not embs:
#             return {"detections": [{"box": [int(x) for x in box], "identity": "Unknown", "confidence": 0.0}]}

#         db_mat = np.vstack(embs).astype(np.float32)
#         db_mat = normalize_embeddings(db_mat)
#         q = normalize_embeddings(query.astype(np.float32))  # [1,512]

#         sims = np.dot(q, db_mat.T)[0]  # cosine similarity
#         max_idx = int(np.argmax(sims))
#         max_sim = float(sims[max_idx])
#         identity = names[max_idx] if max_sim >= self.threshold else "Unknown"

#         det = {
#             "box": [int(x) for x in box],
#             "identity": identity,
#             "confidence": max_sim
#         }
#         return {
#             "detections": [det],
#             "best": {"identity": identity, "confidence": max_sim}
#         }

import os
import threading
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- Utility: Convert image to RGB safely ---------------- #
def to_rgb_pil(pil_img):
    """Ensure image is RGB mode before processing."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img


# ---------------- Core Face Service ---------------- #
class FaceService:
    """
    Face recognition service with lazy-loaded models for memory efficiency.
    """

    def __init__(self, threshold: float = 0.6):
        self.device = "cpu"  # Force CPU on Render (no GPU)
        self.mtcnn = None
        self.resnet = None
        self.threshold = threshold
        self._lock = threading.Lock()
        print("[INIT] FaceService initialized (models lazy-loaded).")

    # ---------------- Lazy load models only when needed ---------------- #
    def _load_models(self):
        """Load models only when required."""
        if self.mtcnn is None or self.resnet is None:
            print("[MODEL] Loading MTCNN and ResNet models...")
            self.mtcnn = MTCNN(keep_all=True, image_size=160, margin=20, device=self.device)
            self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            print("[MODEL] Models loaded successfully.")

    # ---------------- Extract Embeddings ---------------- #
    def _embed(self, pil_img: Image.Image):
        """Detect faces and return their embeddings."""
        self._load_models()
        pil_img = to_rgb_pil(pil_img)
        faces = self.mtcnn(pil_img)
        boxes, _ = self.mtcnn.detect(pil_img)

        if faces is None or boxes is None:
            print("[EMBED] No face detected.")
            return None, None

        with torch.no_grad():
            embeddings = self.resnet(faces.to(self.device)).cpu().numpy().astype(np.float32)

        print(f"[EMBED] Generated embeddings for {len(embeddings)} faces.")
        return boxes, embeddings

    # ---------------- Register a new face ---------------- #
    def register_face(self, image_path: str):
        """Register a new face and store its embedding."""
        try:
            pil_img = Image.open(image_path)
        except Exception as e:
            print(f"[ERROR] Failed to open image: {e}")
            return None

        _, embedding = self._embed(pil_img)
        if embedding is None:
            return None

        embedding = embedding[0]
        print("[REGISTER] Face registered successfully.")
        return embedding

    # ---------------- Compare embeddings ---------------- #
    def recognize_face(self, registered_embeddings: list, test_image_path: str):
        """
        Compare the test image with registered embeddings and return the most similar match.
        """
        try:
            pil_img = Image.open(test_image_path)
        except Exception as e:
            print(f"[ERROR] Failed to open image: {e}")
            return None, 0.0

        _, test_emb = self._embed(pil_img)
        if test_emb is None:
            return None, 0.0

        test_emb = test_emb[0]
        similarities = cosine_similarity([test_emb], registered_embeddings)[0]
        best_match_index = int(np.argmax(similarities))
        best_score = similarities[best_match_index]

        print(f"[RECOGNIZE] Best match score: {best_score:.3f}")
        if best_score >= self.threshold:
            print("[RECOGNIZE] Face recognized successfully.")
            return best_match_index, float(best_score)
        else:
            print("[RECOGNIZE] No matching face found.")
            return None, float(best_score)

    # ---------------- Optional: Release Models (For Low Memory Plans) ---------------- #
    def unload_models(self):
        """Manually release model memory (for free Render plans)."""
        try:
            del self.mtcnn
            del self.resnet
            self.mtcnn = None
            self.resnet = None
            torch.cuda.empty_cache()
            print("[CLEANUP] Models unloaded and memory cleared.")
        except Exception as e:
            print(f"[CLEANUP ERROR] {e}")
