import json
from typing import Dict, Any, List, Optional

class MetadataStore:
    def __init__(self):
        self._by_uuid: Dict[str, Dict[str, Any]] = {}
        self._by_file: Dict[str, Dict[str, Any]] = {}
        self._all: List[Dict[str, Any]] = []

    def load(self, metadata_path: str) -> None:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._all = data
        self._by_uuid = {m["global_uuid"]: m for m in data}
        self._by_file = {m["file_name"]: m for m in data}

    def get_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        return self._by_uuid.get(uuid)

    def get_by_file(self, file_name: str) -> Optional[Dict[str, Any]]:
        return self._by_file.get(file_name)

    def all(self) -> List[Dict[str, Any]]:
        return self._all
