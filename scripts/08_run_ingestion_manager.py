# 07_run_ingestion_manager.py
# Purpose: Run ingestion pipeline to process HotpotQA data and produce artifacts.

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.managers.ingestion_manager import IngestionManager, IngestionConfig

ARTIFACTS_DIR = ROOT / "artifacts"
FAISS_ASCII = Path(os.environ.get("FAISS_ASCII_DIR", r"C:\temp\ref_rag_artifacts"))
FAISS_ASCII.mkdir(parents=True, exist_ok=True)

config = IngestionConfig(
    hotpot_file_path=str(ARTIFACTS_DIR / "hotpot_dev_distractor_v1.json"),
    storage_map_path=str(ARTIFACTS_DIR / "file_to_storage_info.json"),
    output_dir=str(FAISS_ASCII),
    faiss_index_name="faiss.index",
    faiss_meta_name="faiss_metadata.json",
    metadata_name="metadata.json",
)

print(f"[INFO] Writing FAISS/metadata to {FAISS_ASCII}")
manager = IngestionManager(config)
artifact_paths = manager.ingest()

# Keep copies under artifacts/ (metadata must be the working copy)
shutil.copy2(FAISS_ASCII / "metadata.json", ARTIFACTS_DIR / "metadata.json")
shutil.copy2(FAISS_ASCII / "faiss_metadata.json", ARTIFACTS_DIR / "faiss_metadata.json")
try:
    shutil.copy2(FAISS_ASCII / "faiss.index", ARTIFACTS_DIR / "faiss.index")
except OSError as e:
    print(f"[WARN] could not copy faiss.index into artifacts: {e}")

print("\n[INFO] Generated artifacts:")
for name, path in artifact_paths.items():
    print(f"  - {name}: {path}")
print(f"[INFO] Working metadata: {ARTIFACTS_DIR / 'metadata.json'}")
