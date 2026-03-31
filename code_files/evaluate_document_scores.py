import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUT_DIR = Path("output/openai/claims/nli_score/claims/hindi_1")
DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Threshold document support scores and optionally compute balanced accuracy."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing NLI-scored JSON files.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: file,label where label is 0 or 1.",
    )
    return parser.parse_args()


def load_labels(path):
    labels = {}
    if path is None:
        return labels

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_name = (row.get("file") or "").strip()
            label_text = (row.get("label") or "").strip()
            if not file_name or label_text not in {"0", "1"}:
                continue
            labels[file_name] = int(label_text)

    return labels


def load_scores(input_dir):
    records = []
    for path in sorted(input_dir.glob("*.json"), key=lambda p: int(p.stem)):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, ValueError):
            continue

        score = data.get("document_support_score")
        if not isinstance(score, (int, float)):
            continue

        records.append({"file": path.name, "score": float(score)})

    return records


def confusion_counts(records, labels, threshold):
    tp = fp = tn = fn = 0

    for record in records:
        file_name = record["file"]
        if file_name not in labels:
            continue

        pred = 1 if record["score"] >= threshold else 0
        gold = labels[file_name]

        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        else:
            fn += 1

    return tp, fp, tn, fn


def balanced_accuracy(tp, fp, tn, fn):
    tpr = tp / (tp + fn) if (tp + fn) else None
    tnr = tn / (tn + fp) if (tn + fp) else None

    if tpr is None or tnr is None:
        return None, tpr, tnr

    return (tpr + tnr) / 2, tpr, tnr


def summarize_threshold(records, labels, threshold):
    factual = sum(1 for record in records if record["score"] >= threshold)
    non_factual = len(records) - factual

    summary = {
        "threshold": threshold,
        "total_documents": len(records),
        "predicted_factual": factual,
        "predicted_non_factual": non_factual,
    }

    if labels:
        tp, fp, tn, fn = confusion_counts(records, labels, threshold)
        bal_acc, tpr, tnr = balanced_accuracy(tp, fp, tn, fn)
        summary.update(
            {
                "labeled_documents": tp + fp + tn + fn,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "tpr": tpr,
                "tnr": tnr,
                "balanced_accuracy": bal_acc,
            }
        )

    return summary


def main():
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    records = load_scores(args.input_dir)
    labels = load_labels(args.labels_csv)

    if not records:
        print(f"No valid document_support_score values found in {args.input_dir}")
        return

    print(f"Input directory: {args.input_dir}")
    print(f"Documents with scores: {len(records)}")
    print()

    if labels:
        print(f"Loaded labels: {len(labels)}")
    else:
        print("No labels provided, so balanced accuracy cannot be computed.")
        print("This run will only show threshold-based factual/non-factual counts.")
    print()

    for threshold in DEFAULT_THRESHOLDS:
        result = summarize_threshold(records, labels, threshold)
        print(f"Threshold: {result['threshold']:.1f}")
        print(f"Total documents: {result['total_documents']}")
        print(f"Predicted factual: {result['predicted_factual']}")
        print(f"Predicted non-factual: {result['predicted_non_factual']}")

        if labels:
            print(f"Labeled documents: {result['labeled_documents']}")
            print(f"TP: {result['tp']}")
            print(f"FP: {result['fp']}")
            print(f"TN: {result['tn']}")
            print(f"FN: {result['fn']}")
            print(f"TPR: {result['tpr']:.4f}" if result["tpr"] is not None else "TPR: N/A")
            print(f"TNR: {result['tnr']:.4f}" if result["tnr"] is not None else "TNR: N/A")
            print(
                f"Balanced Accuracy: {result['balanced_accuracy']:.4f}"
                if result["balanced_accuracy"] is not None
                else "Balanced Accuracy: N/A"
            )
        print("-" * 60)


if __name__ == "__main__":
    main()
