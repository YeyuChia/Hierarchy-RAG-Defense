# src/permission_retriever.py
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, time

@dataclass
class PermissionRetrieverConfig:
    max_workers: int = 10
    use_cache: bool = False
    per_item_timeout: Optional[float] = None  # seconds

class PermissionRetriever:
    """
    Filter retrieved docs by checking access via IAM adapters.
    - ac_manager.get_access_controller(provider) -> adapter
    - adapter.check_access(resource_metadata) -> bool
    - metadata_store.get_by_uuid(uuid) -> metadata dict
    """
    def __init__(self, metadata_store, config: PermissionRetrieverConfig = PermissionRetrieverConfig()):
        self.ms = metadata_store
        self.config = config
        self._cache: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def clear_cache(self): 
        with self._lock: self._cache.clear()

    def prime_cache(self, uuid_to_allowed: Dict[str, bool]): 
        if not uuid_to_allowed: return
        with self._lock: self._cache.update(uuid_to_allowed)

    def _check_one(self, ac_manager, uuid: str) -> Tuple[str, bool, float]:
        
        t0 = time.monotonic()
        meta = self.ms.get_by_uuid(uuid)
        if not meta:
            return uuid, False, time.monotonic() - t0
        provider = (meta.get("provider") or "").lower()
        ctrl = ac_manager.get_access_controller(provider) if ac_manager else None
        if not ctrl:
            return uuid, False, time.monotonic() - t0
        try:
            allowed = bool(ctrl.check_access(meta))
        except Exception:
            allowed = False
        return uuid, allowed, time.monotonic() - t0

    def filter_docs(
        self,
        retrieved_docs: List[Dict[str, Any]],
        ac_manager,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, bool], Dict[str, float], float]:
        """
        Returns:
          filtered_docs, decisions{uuid->bool}, per_uuid_durations{uuid->sec}, total_duration
        """
        t_all = time.monotonic()
        decisions: Dict[str, bool] = {}
        per_uuid_dur: Dict[str, float] = {}
        
        user_name = ac_manager.user_name
        
        uuids = [d["global_uuid"] for d in retrieved_docs if d.get("global_uuid")]
        to_check: List[str] = []
        if self.config.use_cache:
            with self._lock:
                for u in uuids:
                    cache_key = f"{user_name}:{u}"
                    if cache_key in self._cache:
                        decisions[u] = self._cache[cache_key]
                    else:
                        to_check.append(u)
        else:
            to_check = uuids[:]

        if to_check:
            mw = max(1, min(self.config.max_workers, len(to_check)))
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = {ex.submit(self._check_one, ac_manager, u): u for u in to_check}
                for fut in as_completed(futs):
                    u = futs[fut]
                    try:
                        uid, ok, dur = fut.result(timeout=self.config.per_item_timeout)
                    except Exception:
                        uid, ok, dur = u, False, 0.0
                    decisions[uid] = ok
                    per_uuid_dur[uid] = dur
            if self.config.use_cache:
                with self._lock:
                    for u in to_check:
                        cache_key = f"{user_name}:{u}"
                        self._cache[cache_key] = decisions[u]

        filtered = [d for d in retrieved_docs if decisions.get(d.get("global_uuid"), False)]
        return filtered, decisions, per_uuid_dur, (time.monotonic() - t_all)
