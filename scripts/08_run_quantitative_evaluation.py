import os, sys, json, time
from dotenv import load_dotenv

from src.stores.vector_store import VectorStore
from src.stores.metadata_store import MetadataStore
from src.retrievers.permission_retriever import PermissionRetriever, PermissionRetrieverConfig
from src.managers.credential_manager import CredentialManager
from src.processors.question_processor import QuestionProcessor

from artifacts.hotpot_evaluate_with_em_admin import eval as hp_eval

load_dotenv()

def main():
    ARTIFACTS = "./artifacts"
    QUESTION_FILE = os.path.join(ARTIFACTS, "question_list.json")
    METADATA_FILE = os.path.join(ARTIFACTS, "metadata.json")
    FAISS_INDEX = os.path.join(ARTIFACTS, "faiss.index")
    FAISS_META  = os.path.join(ARTIFACTS, "faiss_metadata.json")
    RESULTS_DIR = os.path.join(ARTIFACTS, "results")
    PRED_DIR = os.path.join(RESULTS_DIR, "preds")
    GOLD_FILE = os.path.join(ARTIFACTS, "hotpot_dev_distractor_v1.json")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    # ----- Experiment setting -----
    TOP_K = 5
    MAX_WORKERS = 10
    QUESTION_CONCURRENCY = 2
    USE_CACHE = True
    TARGET_USERS = {"user-1", "user-2", "user-3", "user-4", "user-5", "user-6", "user-7", "baseline-admin"}
    SAMPLE_SIZE = 1000
    
    # ----- Managers / Stores -----
    user_manager = CredentialManager(base_path=ARTIFACTS, file_name="test_credential.txt")
    vs = VectorStore(model_name="all-MiniLM-L6-v2")
    vs.load(FAISS_INDEX, FAISS_META)

    ms = MetadataStore()
    ms.load(METADATA_FILE)

    # ----- Processor + PermissionRetriever -----
    pr = PermissionRetriever(ms, PermissionRetrieverConfig(max_workers=1, use_cache=USE_CACHE))
    qp = QuestionProcessor(QUESTION_FILE, METADATA_FILE, vs, user_manager, permission_retriever=pr)
    qp.sample_questions(sample_size=SAMPLE_SIZE, seed=42)

    summary_rows = []
    details_all = []
    summary_index = {}
    
    t_perm_check_start = time.time()

    for user_name in user_manager.users.keys():
        if user_name not in TARGET_USERS:
            continue

        ac_manager = user_manager.get_ac_manager(user_name)

        t0 = time.time()

        result = qp.run_for_quantitative(
            ac_manager,
            user_name=user_name,
            top_k=TOP_K,
            max_workers=MAX_WORKERS,
            question_concurrency=QUESTION_CONCURRENCY,
            use_permission_cache=USE_CACHE,
        )
        t1 = time.time()
        print(f"[INFO] run_for_qualitative elapsed: {t1 - t0:.2f}s")

        agg = result["aggregate"]
        print(f"- PermCov: {agg['PermissionCoverage_pct']:.2f}% | "
              f"AccessMatch: {agg['AccessMatch']} | "
              f"TP/FP/FN/TN: {agg['TP']}/{agg['FP']}/{agg['FN']}/{agg['TN']} | "
              f"Docs: {agg['total_authorized']}/{agg['total_retrieved']} auth/retr")

        row = {
            "user": user_name,
            "top_k": TOP_K,
            "max_workers": MAX_WORKERS,
            "use_cache": USE_CACHE,
            "PermissionCoverage_pct": agg["PermissionCoverage_pct"],
            "AccessMatch": agg["AccessMatch"],
            "TP": agg["TP"], "FP": agg["FP"], "FN": agg["FN"], "TN": agg["TN"],
            "total_retrieved": agg["total_retrieved"],
            "total_authorized": agg["total_authorized"],
            "elapsed_sec": round(t1 - t0, 2),
            "em": None,
            "f1": None,
            "em_admin": None,
        }
        summary_rows.append(row)
        summary_index[user_name] = row

        details_all.append({
            "config": {"user": user_name, "top_k": TOP_K, "max_workers": MAX_WORKERS, "use_cache": USE_CACHE},
            "result": result,
        })

        pred_map = {}
        for q in result["by_question"]:
            qid = q["question_id"]
            pred_answer = (q.get("predicted_answer") or "").strip()
            if not pred_answer:
                pred_answer = "I don't know."
            pred_map[qid] = pred_answer

        pred_obj = {
            "answer": pred_map,
            "sp": {}
        }
        with open(os.path.join(PRED_DIR, f"preds_{user_name}.json"), "w", encoding="utf-8") as f:
            json.dump(pred_obj, f, indent=2, ensure_ascii=False)

    t_perm_check_end = time.time()
    print(f"\n[TIMER] Permission checking and prediction generation took: {t_perm_check_end - t_perm_check_start:.2f}s")

    t_eval_start = time.time()

    admin_file = None
    for fname in os.listdir(PRED_DIR):
        if fname.endswith(".json") and "baseline-admin" in fname:
            admin_file = os.path.join(PRED_DIR, fname)
            break

    for fname in sorted(os.listdir(PRED_DIR)):
        if not fname.endswith(".json"):
            continue
        pred_path = os.path.join(PRED_DIR, fname)
        base = os.path.splitext(fname)[0]
        user_from_file = base.replace("preds_", "", 1)

        try:
            metrics = hp_eval(
                prediction_file=pred_path,
                gold_file=GOLD_FILE,
                admin_prediction_file=admin_file
            )
            if user_from_file in summary_index:
                summary_index[user_from_file]["em"] = round(metrics.get("em", 0.0), 4)
                summary_index[user_from_file]["f1"] = round(metrics.get("f1", 0.0), 4)
                if "em_admin" in metrics:
                    summary_index[user_from_file]["em_admin"] = round(metrics["em_admin"], 4)

            print(f"[INFO] evaluated {fname} -> em={metrics.get('em'):.4f}, "
                  f"f1={metrics.get('f1'):.4f}, "
                  f"em_admin={metrics.get('em_admin', 0.0):.4f}")
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {fname}: {e}")

    t_eval_end = time.time()
    print(f"[TIMER] Final evaluation (EM, F1, EM@Admin) took: {t_eval_end - t_eval_start:.2f}s\n")
    
    # ----- Save outputs -----
    quick = [
    {
        'user': r.get('user'),
        'PermCov': round(r.get('PermissionCoverage_pct', 0.0), 2),
        'Match': 'PASS' if r.get('AccessMatch') else 'FAIL',
        'EM': r.get('em'),
        'F1': r.get('f1'),
        'EM@Admin': r.get('em_admin')
    } for r in summary_rows
    ]
    print('[SUMMARY]', quick)
    
    with open(os.path.join(RESULTS_DIR, "quantitative_summary.json"), "w", encoding="utf-8") as f:
        json.dump(quick, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS_DIR, "quantitative_details.json"), "w", encoding="utf-8") as f:
        json.dump(details_all, f, indent=2, ensure_ascii=False)
    print("[DONE] Saved qualitative summary & details including EM/F1/EM@Admin.")

if __name__ == "__main__":
    main()