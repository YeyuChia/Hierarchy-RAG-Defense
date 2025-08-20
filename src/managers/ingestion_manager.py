import os
import json
import uuid
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from tqdm import tqdm
from src.stores.vector_store import VectorStore

def sanitize_filename(title: str) -> str:
    title = title.lower().replace(" ", "_")
    return re.sub(r"[^\w\d_]", "", title)

@dataclass
class IngestionConfig:
    storage_map_path: str = "file_to_storage_info.json"
    hotpot_file_path: str = "hotpot_dev_distractor_v1.json"
    output_dir: str = "artifacts"
    faiss_index_name: str = "faiss.index"
    faiss_meta_name: str = "faiss_metadata.json"
    metadata_name: str = "metadata.json"
    embedding_model: str = "all-MiniLM-L6-v2"
    gcp_endpoint: str = "https://storage.googleapis.com"
    aws_endpoint: str = "https://s3.ap-northeast-2.amazonaws.com"
    deterministic_uuid: bool = True

class IngestionManager:
    def __init__(self, cfg: IngestionConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.vector_store = VectorStore(self.cfg.embedding_model)

    def ingest(self) -> Dict[str, str]:
        storage_map = self._load_json(self.cfg.storage_map_path)
        hotpot_data = self._load_json(self.cfg.hotpot_file_path)

        print(f"Loading data from HotpotQA file: {self.cfg.hotpot_file_path}")
        records = self._collect_records_from_hotpot(hotpot_data, storage_map)

        print("Building FAISS index from records...")
        self.vector_store.build_from_records(records)

        index_path = os.path.join(self.cfg.output_dir, self.cfg.faiss_index_name)
        vecmeta_path = os.path.join(self.cfg.output_dir, self.cfg.faiss_meta_name)
        self.vector_store.save(index_path, vecmeta_path)

        access_meta = [meta for _, _, meta in records]
        meta_path = os.path.join(self.cfg.output_dir, self.cfg.metadata_name)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(access_meta, f, indent=2, ensure_ascii=False)
        print(f"Saved access metadata to {meta_path}")
        print("\nIngestion complete.")

        return {
            "faiss_index": index_path,
            "faiss_metadata": vecmeta_path,
            "metadata": meta_path,
        }

    def _load_json(self, path: str) -> Any:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _collect_records_from_hotpot(
        self,
        hotpot_data: List[Dict[str, Any]],
        storage_map: Dict[str, Any]
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        doc_map = {}
        print("Extracting unique documents from HotpotQA data...")
        for item in hotpot_data:
            for title, _ in item["supporting_facts"]:
                if title not in doc_map:
                    for ctx_title, sents in item["context"]:
                        if ctx_title == title:
                            doc_map[title] = " ".join(sents)
                            break
        
        print(f"Total unique docs to ingest: {len(doc_map)}")
        
        records: List[Tuple[str, str, Dict[str, Any]]] = []
        for title, content in tqdm(doc_map.items(), desc="Processing documents"):
            file_name = sanitize_filename(title) + ".txt"
            
            storage_info = storage_map.get(file_name, {})
            provider = storage_info.get("provider", "")
            bucket = storage_info.get("bucket", "")
            
            if self.cfg.deterministic_uuid:
                key = f"{provider}:{bucket}:{file_name}"
                global_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
            else:
                global_uuid = str(uuid.uuid4())

            endpoint = self.cfg.gcp_endpoint if provider == "gcp" else self.cfg.aws_endpoint

            meta = {
                "file_name": file_name,
                "global_uuid": global_uuid,
                "provider": provider,
                "endpoint": endpoint,
                "parameters": {
                    "iam": storage_info.get("iam", ""),
                    "region": storage_info.get("region", ""),
                    "bucket": bucket,
                },
            }
            records.append((file_name, content, meta))

        if not records:
            raise RuntimeError("No records were collected. Check the HotpotQA data and storage map.")
        return records