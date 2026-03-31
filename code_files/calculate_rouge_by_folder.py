import json
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path("output/openai/summary")
MAX_FILES_PER_FOLDER = 1000


def tokenize(text):
    if not text:
        return []
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def get_ngrams(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(reference, candidate, n):
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)

    reference_ngrams = get_ngrams(reference_tokens, n)
    candidate_ngrams = get_ngrams(candidate_tokens, n)

    if not reference_ngrams or not candidate_ngrams:
        return 0.0

    overlap = sum((reference_ngrams & candidate_ngrams).values())
    reference_total = sum(reference_ngrams.values())
    candidate_total = sum(candidate_ngrams.values())

    recall = overlap / reference_total if reference_total else 0.0
    precision = overlap / candidate_total if candidate_total else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def score_folder(folder_path):
    rouge_1_scores = []
    rouge_2_scores = []
    processed_files = 0
    skipped_files = 0

    for index in range(MAX_FILES_PER_FOLDER):
        file_path = folder_path / f"{index}.json"
        if not file_path.exists():
            skipped_files += 1
            continue

        try:
            data = load_json(file_path)
        except (json.JSONDecodeError, OSError):
            skipped_files += 1
            continue

        reference_summary = data.get("summary", "")
        generated_summary = data.get("generated_summary", "")

        if not reference_summary or not generated_summary:
            skipped_files += 1
            continue

        rouge_1_scores.append(rouge_n_f1(reference_summary, generated_summary, 1))
        rouge_2_scores.append(rouge_n_f1(reference_summary, generated_summary, 2))
        processed_files += 1

    average_rouge_1 = sum(rouge_1_scores) / processed_files if processed_files else 0.0
    average_rouge_2 = sum(rouge_2_scores) / processed_files if processed_files else 0.0

    return {
        "folder": folder_path.name,
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "rouge_1_f1": average_rouge_1,
        "rouge_2_f1": average_rouge_2,
    }


def main():
    if not BASE_DIR.exists():
        print(f"Base directory not found: {BASE_DIR}")
        return

    folders = sorted(path for path in BASE_DIR.iterdir() if path.is_dir())

    if not folders:
        print(f"No folders found inside: {BASE_DIR}")
        return

    print("ROUGE scores by folder")
    print("-" * 60)

    for folder in folders:
        result = score_folder(folder)
        print(f"Folder: {result['folder']}")
        print(f"Processed files: {result['processed_files']}")
        print(f"Skipped files: {result['skipped_files']}")
        print(f"ROUGE-1 F1: {result['rouge_1_f1']:.4f}")
        print(f"ROUGE-2 F1: {result['rouge_2_f1']:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
