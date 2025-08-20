from typing import Dict, Any
from google.cloud import storage
from google.api_core.exceptions import Forbidden, NotFound
from .base import AccessControlBase

class GCPAccessControl(AccessControlBase):
    def __init__(self, user_credential: Dict[str, Any]):
        super().__init__(user_credential)
        self.gcp_client = self._create_gcp_client()

    def _create_gcp_client(self):
        cred = self.user_credential.get("gcp_credential")
        if not cred:
            return None
        try:
            return storage.Client.from_service_account_info(cred)
        except Exception as e:
            print(f"[GCP][ERROR] Failed to create storage client: {e}")
            return None

    def check_access(self, resource_metadata: Dict[str, Any]) -> bool:
        if not self.gcp_client:
            return False
        bucket = (resource_metadata.get("parameters") or {}).get("bucket")
        name   = resource_metadata.get("file_name")
        if not (bucket and name):
            return False
        try:
            blob = self.gcp_client.bucket(bucket).blob(name)
            needed  = ['storage.objects.get']
            allowed = blob.test_iam_permissions(needed)
            if needed[0] not in allowed:
                return False
            return blob.exists()
        except (Forbidden, NotFound):
            return False
        except Exception:
            return False
