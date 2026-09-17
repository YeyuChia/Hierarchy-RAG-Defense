# Local smoke test: retrieval + mock IAM filter only (no OpenAI, no cloud).
import json
import os
import random
from pathlib import Path

from src.managers.credential_manager import CredentialManager
from src.retrievers.permission_retriever import PermissionRetriever, PermissionRetrieverConfig
from src.stores.metadata_store import MetadataStore
from src.stores.vector_store import VectorStore

ARTIFACTS = Path("artifacts")
TOP_K = 5
SAMPLE_SIZE = 50
TARGET_USERS = ["user-1", "user-2", "user-3", "user-7", "baseline-admin"]
SEED = 42


def load_expected(user_name: str, access_map: dict) -> dict:
    truth = access_map.get(user_name) or {}
    if isinstance(truth, dict) and truth.get("full_access"):
        return {"__FULL_ACCESS__": True}
    if isinstance(truth, dict) and "files" in truth:
        return {fn: True for fn in truth.get("files", [])}
    return {}


def expect_allow(exp_map: dict, file_name: str) -> bool:
    if exp_map.get("__FULL_ACCESS__"):
        return True
    return bool(exp_map.get(file_name, False))


def main() -> None:
    os.environ.setdefault("LOCAL_MOCK_IAM", "0")

    vs = VectorStore(model_name="all-MiniLM-L6-v2")
    vs.load(str(ARTIFACTS / "faiss.index"), str(ARTIFACTS / "faiss_metadata.json"))

    ms = MetadataStore()
    ms.load(str(ARTIFACTS / "metadata.json"))

    with open(ARTIFACTS / "question_list.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(ARTIFACTS / "user_accessible_files.json", "r", encoding="utf-8") as f:
        access_map = json.load(f)

    random.seed(SEED)
    sampled = random.sample(questions, min(SAMPLE_SIZE, len(questions)))

    user_manager = CredentialManager(base_path=str(ARTIFACTS), file_name="test_credential.txt")
    pr = PermissionRetriever(ms, PermissionRetrieverConfig(max_workers=8, use_cache=True))

    rows = []
    for user_name in TARGET_USERS:
        ac = user_manager.get_ac_manager(user_name)
        if ac is None:
            print(f"[WARN] no credentials for {user_name}")
            continue
        exp = load_expected(user_name, access_map)

        total_retrieved = 0
        total_expected_auth = 0
        TP = FP = FN = TN = 0

        for q in sampled:
            retrieved = vs.search(q["question"], top_k=TOP_K)
            _, decisions, _, _ = pr.filter_docs(retrieved, ac)
            for row in retrieved:
                meta = ms.get_by_uuid(row["global_uuid"]) or {}
                fname = meta.get("file_name", "")
                e = expect_allow(exp, fname)
                a = bool(decisions.get(row["global_uuid"], False))
                total_retrieved += 1
                if e:
                    total_expected_auth += 1
                if a and e:
                    TP += 1
                elif a and not e:
                    FP += 1
                elif (not a) and e:
                    FN += 1
                else:
                    TN += 1

        permcov = (total_expected_auth / total_retrieved * 100.0) if total_retrieved else 0.0
        match = "PASS" if (FP + FN) == 0 else "FAIL"
        row = {
            "user": user_name,
            "sample_size": len(sampled),
            "top_k": TOP_K,
            "PermCov": round(permcov, 2),
            "Match": match,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "total_retrieved": total_retrieved,
            "total_authorized": total_expected_auth,
        }
        rows.append(row)
        print(
            f"[RESULT] {user_name}: PermCov={row['PermCov']:.2f}% | "
            f"Match={match} | TP/FP/FN/TN={TP}/{FP}/{FN}/{TN}"
        )

    out = ARTIFACTS / "results"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "permission_smoke_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Saved {out_path}")


if __name__ == "__main__":
    main()
