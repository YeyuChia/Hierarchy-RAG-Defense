# 06_generate_question_list.py
# Purpose: Extract question/answer pairs from HotpotQA distractor set.
# Note: Business logic preserved. Comments normalized; [INFO] logs only.

import json
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
INPUT_FILE = ARTIFACTS_DIR / "hotpot_dev_distractor_v1.json"
OUTPUT_FILE = ARTIFACTS_DIR / "question_list.json"


def _ensure_parent_dir(path_str: str) -> None:
    """Create parent directory if it does not exist."""
    p = Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


def generate_question_list() -> None:
    # Load dataset
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_list = []
    for item in data:
        question_id = item["_id"]
        question = item["question"]
        gold_answer = item["answer"]

        question_list.append(
            {
                "id": question_id,
                "question": question,
                "gold_answer": gold_answer,
            }
        )

    # Save output
    _ensure_parent_dir(OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(question_list, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Extracted {len(question_list)} questions to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_question_list()
