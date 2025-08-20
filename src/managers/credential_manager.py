import os
import json
from typing import Dict, Any, Optional
from src.adapters.iam.adapter_factory import AdapterFactory

class CredentialManager:
    def __init__(self, base_path: str = "./artifacts", file_name: str = "test_credential.txt"):
        self.base_path = base_path
        self.env_file = os.path.join(self.base_path, file_name)

        self.users: Dict[str, Dict[str, Any]] = {}
        self.ac_managers: Dict[str, AdapterFactory] = {}

        self._load_users_and_factories()

    def _load_users_and_factories(self) -> None:
        if not os.path.exists(self.env_file):
            print(f"[INFO] {self.env_file} does not exist.")
            return

        with open(self.env_file, "r", encoding="utf-8") as f:
            users_list = json.load(f)
        self.users = {u["name"]: u for u in users_list}

        for user_name, creds in self.users.items():
            factory = AdapterFactory(creds)
            factory.user_name = user_name
            if not hasattr(factory, "get_access_controller"):
                factory.get_access_controller = factory.get
            self.ac_managers[user_name] = factory

    def get_user_credentials(self, username: str) -> Dict[str, Any]:
        return self.users.get(username, {})

    def get_user_accessible_files(self, user_name: str):
        path = os.path.join(self.base_path, "user_accessible_files.json")
        with open(path, "r", encoding="utf-8") as f:
            access_map = json.load(f)
        return access_map.get(user_name, {})

    def get_ac_manager(self, user_name: str) -> Optional[AdapterFactory]:
        return self.ac_managers.get(user_name)
