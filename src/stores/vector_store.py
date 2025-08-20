import os
import json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.vector_meta: List[Dict[str, Any]] = []

    # -------- internal --------
    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    # -------- build / save --------
    def build_from_records(self, records: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        self._ensure_model()
        texts = [content for _, content, _ in records]
        embeddings = self.model.encode(texts)
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")

        index = faiss.IndexFlatL2(arr.shape[1])
        index.add(arr)
        self.index = index

        # Row-aligned vector metadata (same order as embeddings)
        self.vector_meta = []
        for (_, content, meta) in records:
            self.vector_meta.append({
                "file_name": meta["file_name"],
                "global_uuid": meta["global_uuid"],
                "content": content,
            })

    def save(self, index_path: str, meta_path: str) -> None:
        if self.index is None:
            raise RuntimeError("No index to save. Build or load first.")
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.vector_meta, f, indent=2, ensure_ascii=False)
        print(f"Saved FAISS index to {index_path}")
        print(f"Saved vector metadata to {meta_path}")

    # -------- load / search --------
    def load(self, index_path: str, meta_path: str) -> None:
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Loaded FAISS index from {index_path}")
        else:
            raise FileNotFoundError(f"Index file not found at {index_path}")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.vector_meta = json.load(f)
            print(f"Loaded vector metadata from {meta_path}")
        else:
            raise FileNotFoundError(f"Vector metadata file not found at {meta_path}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self._ensure_model()
        if self.index is None:
            raise RuntimeError("FAISS index not loaded or built.")
        embedding = self.model.encode([query])[0].astype(np.float32)
        distances, indices = self.index.search(np.array([embedding]), top_k)
        results: List[Dict[str, Any]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.vector_meta):
                row = self.vector_meta[idx].copy()
                row["distance"] = float(dist)
                results.append(row)
        return results
