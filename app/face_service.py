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

import cv2
import numpy as np
import face_recognition
import os
from io import BytesIO
from PIL import Image

class FaceService:
    def __init__(self, base_dir="faces"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.known_encodings = []
        self.known_ids = []
        self.load_known_faces()

    def load_known_faces(self):
        """Loads all saved faces into memory."""
        self.known_encodings = []
        self.known_ids = []

        for user_id in os.listdir(self.base_dir):
            user_folder = os.path.join(self.base_dir, user_id)
            if not os.path.isdir(user_folder):
                continue

            for img_name in os.listdir(user_folder):
                path = os.path.join(user_folder, img_name)
                try:
                    img = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_ids.append(user_id)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

        print(f"Loaded {len(self.known_encodings)} known faces.")

    def register_face(self, image_bytes: bytes, sweeperId: str):
        """Registers a new sweeper's face."""
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(img)
            encodings = face_recognition.face_encodings(img_np)
            if not encodings:
                return {"status": "failed", "message": "No face detected."}

            user_folder = os.path.join(self.base_dir, str(sweeperId))
            os.makedirs(user_folder, exist_ok=True)
            img_path = os.path.join(user_folder, f"{len(os.listdir(user_folder))}.jpg")
            img.save(img_path)

            self.known_encodings.append(encodings[0])
            self.known_ids.append(str(sweeperId))

            return {"status": "success", "message": "Face registered successfully."}

        except Exception as e:
            return {"status": "failed", "message": str(e)}

    def recognize(self, image_bytes: bytes, sweeperId: str = None):
        """Recognizes the sweeper's face from an image."""
        try:
            if not self.known_encodings:
                self.load_known_faces()

            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(img)

            face_locations = face_recognition.face_locations(img_np)
            encodings = face_recognition.face_encodings(img_np, face_locations)

            if not encodings:
                return {"status": "failed", "message": "No face detected."}

            for face_encoding in encodings:
                matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

                if best_match_index is not None and matches[best_match_index]:
                    matched_id = self.known_ids[best_match_index]
                    return {"status": "success", "matchedId": matched_id}

            return {"status": "failed", "message": "No match found."}

        except Exception as e:
            return {"status": "error", "message": str(e)}
