import os
import json, time
from src.stores.vector_store import VectorStore
from src.stores.metadata_store import MetadataStore
from src.retrievers.permission_retriever import PermissionRetriever, PermissionRetrieverConfig
from src.managers.credential_manager import CredentialManager
from src.processors.question_processor import QuestionProcessor


from dotenv import load_dotenv
load_dotenv()

def main():
    ARTIFACTS = "./artifacts"
    QUESTION_FILE = os.path.join(ARTIFACTS, "question_list.json")
    METADATA_FILE = os.path.join(ARTIFACTS, "metadata.json")
    FAISS_INDEX = os.path.join(ARTIFACTS, "faiss.index")
    FAISS_META  = os.path.join(ARTIFACTS, "faiss_metadata.json")
    RESULTS_DIR = os.path.join(ARTIFACTS, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    SAMPLE_SIZE = 50

    user_manager = CredentialManager(base_path=ARTIFACTS, file_name="test_credential.txt")
    vs = VectorStore(model_name="all-MiniLM-L6-v2")
    vs.load(FAISS_INDEX, FAISS_META)

    ms = MetadataStore()
    ms.load(METADATA_FILE)

    pr = PermissionRetriever(ms, PermissionRetrieverConfig(max_workers=10, use_cache=False))
    qp = QuestionProcessor(QUESTION_FILE, METADATA_FILE, vs, user_manager, permission_retriever=pr)
    qp.sample_questions(sample_size=SAMPLE_SIZE, seed=42)

    experiment_configs = [
        {'top_k': 10, 'max_workers': 1},
        {'top_k': 10, 'max_workers': 2},
        {'top_k': 10, 'max_workers': 4},
        {'top_k': 10, 'max_workers': 8},
        {'top_k': 10, 'max_workers': 10},
        {'top_k': 10, 'max_workers': 15},
        {'top_k': 10, 'max_workers': 20},

        {'top_k': 1,  'max_workers': 20},
        {'top_k': 3,  'max_workers': 20},
        {'top_k': 5,  'max_workers': 20},
        {'top_k': 10, 'max_workers': 20},
        {'top_k': 15, 'max_workers': 20},
        {'top_k': 20, 'max_workers': 20},
    ]
    
    target_users = {"baseline-admin"}

    summary_rows = []
    for cfg in experiment_configs:
        top_k = cfg["top_k"]
        max_workers = cfg["max_workers"]
        print(f"\n=== Start: top_k={top_k}, max_workers={max_workers} ===")

        for user_name in user_manager.users.keys():
            if user_name not in target_users:
                continue

            ac_manager = user_manager.get_ac_manager(user_name)
            t0 = time.time()
            result = qp.run_for_latency(ac_manager, top_k=top_k, max_workers=max_workers)
            t1 = time.time()
            print(f"[INFO] run_for_latency elapsed: {t1 - t0:.2f}s")

            iam_latencies = []
            aws_durs, gcp_durs = [], []
            num_q = len(result["questions"]) or 1

            for q in result["questions"]:
                iam_latencies.append(q["total_access_check_duration"])
                uuid_to_provider = {d["uuid"]: (d.get("metadata") or {}).get("provider") for d in q["doc_results"]}
                for uuid, dur in q["uuid_access_check_durations"].items():
                    prov = uuid_to_provider.get(uuid)
                    if prov == "aws":
                        aws_durs.append(dur)
                    elif prov == "gcp":
                        gcp_durs.append(dur)

            avg_iam_ms = (sum(iam_latencies) / num_q) * 1000
            avg_e2e_ms = ((t1 - t0) / num_q) * 1000
            avg_aws_ms = (sum(aws_durs) / len(aws_durs)) * 1000 if aws_durs else 0.0
            avg_gcp_ms = (sum(gcp_durs) / len(gcp_durs)) * 1000 if gcp_durs else 0.0

            print(f"- IAM: {avg_iam_ms:.2f} ms | E2E: {avg_e2e_ms:.2f} ms | AWS: {avg_aws_ms:.2f} ms | GCP: {avg_gcp_ms:.2f} ms")

            summary_rows.append({
                "top_k": top_k, "max_workers": max_workers, "user": user_name,
                "iam_latency_ms": round(avg_iam_ms, 2),
                "e2e_latency_ms": round(avg_e2e_ms, 2),
                "avg_aws_response_ms": round(avg_aws_ms, 2),
                "avg_gcp_response_ms": round(avg_gcp_ms, 2),
            })

    with open(os.path.join(RESULTS_DIR, "latency_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print("[DONE] Saved summary.")


if __name__ == "__main__":
    main()
