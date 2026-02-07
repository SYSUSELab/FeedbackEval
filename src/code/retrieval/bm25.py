import os
import json
import re
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


_test_missing_data = None
_bm25_instance = None


def preprocess_text(text):
    """Preprocess text for BM25 matching."""
    if not text:
        return []

    # Convert to lowercase and remove special characters
    text = re.sub(r"[^\w\s]", " ", text.lower())

    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [
            token for token in tokens if token not in stop_words and len(token) > 1
        ]
        return tokens
    except Exception:
        # simple whitespace split if NLTK data is unavailable
        return text.split()


def load_test_missing_data(dataset):
    """Load data and initialize BM25."""
    global _test_missing_data, _bm25_instance

    if _test_missing_data is not None:
        return _test_missing_data, _bm25_instance

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(
        project_root, "dataset", dataset, f"{dataset}_feedback_test.jsonl"
    )

    if not os.path.exists(data_path):
        return [], None

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except Exception:
                continue

    if not data:
        return [], None

    documents = []
    for item in data:
        doc_text = ""
        if item.get("false_results"):
            false_result = item["false_results"][0]
            doc_text += false_result.get("generate_code", "")
        documents.append(preprocess_text(doc_text))

    bm25 = BM25Okapi(documents) if documents else None

    _test_missing_data = data
    _bm25_instance = bm25

    return data, bm25


def find_best_example(current_task, dataset="CoderEval"):
    """Use BM25 to find the most similar example."""
    data, bm25 = load_test_missing_data(dataset)

    if not data or bm25 is None:
        return None

    # Build query text from current task
    query_text = ""
    if current_task.get("false_results"):
        false_result = current_task["false_results"][0]
        query_text += false_result.get("generate_code", "")

    if not query_text.strip():
        return None

    # Preprocess query text
    query_tokens = preprocess_text(query_text)
    if not query_tokens:
        return None

    # Compute BM25 similarity scores
    scores = bm25.get_scores(query_tokens)
    best_idx = scores.argmax()

    # Return the most similar example
    best_example = data[best_idx]

    if best_example.get("false_results"):
        false_result = best_example["false_results"][0]
        return {
            "erroneous_code": false_result.get("generate_code", ""),
            "feedback": false_result.get("test_feedback", ""),
            "fixed_code": best_example.get("correct_code", ""),
        }

    return None

