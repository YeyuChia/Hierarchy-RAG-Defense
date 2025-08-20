# hotpot_evaluate_with_em_admin.py
# Thin wrapper: reuse primitives from hotpot_evaluate_v1.py and compute
# EM, F1, and EM@Admin. Returns a dict without printing.

from typing import Dict, Any, Optional, Set
import json

# Reuse the original helpers to ensure identical behavior
from artifacts.hotpot_evaluate_v1 import (
    normalize_answer,
    f1_score,          # returns (f1, prec, recall)
    exact_match_score, # bool
)

def _answer_dict(pred_obj: Dict[str, Any]) -> Dict[str, str]:
    # supports {"answer": {...}} or flat {"qid": "answer"}
    if isinstance(pred_obj, dict) and "answer" in pred_obj:
        return pred_obj["answer"]
    return pred_obj

def _load_gold_answers(gold_file: str):
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_list = json.load(f)
    # qid -> gold answer
    gold_answers = {item["_id"]: item["answer"] for item in gold_list}
    return gold_answers, gold_list

def _compute_admin_correct_ids(admin_pred_obj: Dict[str, Any],
                               gold_answers: Dict[str, str]) -> Set[str]:
    ans = _answer_dict(admin_pred_obj)
    correct: Set[str] = set()
    for qid, pred in ans.items():
        g = gold_answers.get(qid)
        if g is None:
            continue
        if normalize_answer(str(pred)) == normalize_answer(str(g)):
            correct.add(qid)
    return correct

def _compute_af_em(user_ans: Dict[str, str],
                   gold_answers: Dict[str, str],
                   filtered_qids: Set[str]) -> float:
    if not filtered_qids:
        return 0.0
    good = 0
    for qid in filtered_qids:
        if qid in user_ans:
            if normalize_answer(str(user_ans[qid])) == normalize_answer(str(gold_answers.get(qid, ""))):
                good += 1
    return good / float(len(filtered_qids))

def eval(prediction_file: str,
         gold_file: str,
         admin_prediction_file: Optional[str] = None) -> Dict[str, float]:
    """
    Compute EM, F1 (HotpotQA-normalized) and EM@Admin (Admin-Filtered EM).

    Returns:
      {
        "em": <float in [0,1]>,
        "f1": <float in [0,1]>,
        "em_admin": <float in [0,1]>   # only if admin_prediction_file provided
      }
    """
    # Load inputs
    with open(prediction_file, "r", encoding="utf-8") as f:
        pred_obj = json.load(f)
    user_ans = _answer_dict(pred_obj)

    gold_answers, gold_list = _load_gold_answers(gold_file)
    N = len(gold_list)

    # Aggregate EM and F1 exactly like the original logic:
    # - Missing answers contribute 0 (denominator is N)
    em_sum = 0.0
    f1_sum = 0.0
    for dp in gold_list:
        qid = dp["_id"]
        gold_a = dp["answer"]
        if qid in user_ans:
            # original uses exact_match_score and f1_score on normalized strings
            em_bool = exact_match_score(user_ans[qid], gold_a)
            f1_val, _, _ = f1_score(user_ans[qid], gold_a)
            em_sum += float(em_bool)
            f1_sum += f1_val
        else:
            # missing -> add 0.0 (keeps behavior consistent with original)
            pass

    em = em_sum / float(N) if N > 0 else 0.0
    f1 = f1_sum / float(N) if N > 0 else 0.0

    out = {"em": em, "f1": f1}

    # Optional EM@Admin
    if admin_prediction_file:
        with open(admin_prediction_file, "r", encoding="utf-8") as f:
            admin_obj = json.load(f)
        admin_correct = _compute_admin_correct_ids(admin_obj, gold_answers)
        em_admin = _compute_af_em(user_ans, gold_answers, admin_correct)
        out["em_admin"] = em_admin

    return out
