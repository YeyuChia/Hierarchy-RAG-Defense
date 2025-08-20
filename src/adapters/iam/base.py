from abc import ABC, abstractmethod
from typing import Dict, Any

class AccessControlBase(ABC):
    """Abstract base class for IAM adapters (single-item API)."""
    def __init__(self, user_credential: Dict[str, Any]):
        self.user_credential = user_credential or {}

    @abstractmethod
    def check_access(self, resource_metadata: Dict[str, Any]) -> bool:
        """Return True if the user can access the given resource."""
        pass
