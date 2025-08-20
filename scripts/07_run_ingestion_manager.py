# 07_run_ingestion_manager.py
# Purpose: Run ingestion pipeline to process HotpotQA data and produce artifacts.
# Note: Business logic preserved. Comments normalized; [INFO] logs only.

import sys
import os
from pathlib import Path
from src.managers.ingestion_manager import IngestionManager, IngestionConfig

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

# 1. Add project root to sys.path for src imports
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"[INFO] Added project root to sys.path: {project_root}")

# 2. Define artifacts directory path
ARTIFACTS_DIR = Path("artifacts")
print(f"[INFO] Artifacts directory: {os.path.abspath(ARTIFACTS_DIR)}")

# ---------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------

# Configure ingestion
config = IngestionConfig(
    hotpot_file_path=os.path.join(ARTIFACTS_DIR, "hotpot_dev_distractor_v1.json"),
    storage_map_path=os.path.join(ARTIFACTS_DIR, "file_to_storage_info.json"),
    output_dir=ARTIFACTS_DIR,
)

# Initialize and run manager
manager = IngestionManager(config)
artifact_paths = manager.ingest()

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

print("\n[INFO] Generated artifacts:")
for name, path in artifact_paths.items():
    print(f"  - {name}: {path}")
