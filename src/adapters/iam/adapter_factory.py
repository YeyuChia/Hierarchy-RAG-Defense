from typing import Dict, Type, Optional
from .base import AccessControlBase
from .gcp import GCPAccessControl
from .aws import AWSAccessControl
from .keycloak import KeycloakAccessControl

class AdapterFactory:
    def __init__(self, credentials: Dict):
        self.credentials = credentials or {}
        self._registry: Dict[str, Type[AccessControlBase]] = {
            "gcp": GCPAccessControl,
            "aws": AWSAccessControl,
            "keycloak": KeycloakAccessControl,
            "on-premise": KeycloakAccessControl,
        }
        self._cache: Dict[str, AccessControlBase] = {}

    def normalize(self, provider: Optional[str]) -> str:
        # only strip/lower; no alias expansion
        return (provider or "").strip().lower()

    def get(self, provider: Optional[str]) -> Optional[AccessControlBase]:
        key = self.normalize(provider)
        adapter_cls = self._registry.get(key)
        if adapter_cls is None:
            return None
        if key not in self._cache:
            self._cache[key] = adapter_cls(self.credentials)
        return self._cache[key]

    def get_access_controller(self, provider: Optional[str]):
        return self.get(provider)
