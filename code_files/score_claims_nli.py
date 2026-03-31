import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_CLAIMS_ROOTS = [
    Path("output/openai/claims"),
    Path("output/claims/indicbartss"),
]
DEFAULT_OUTPUT_ROOT = Path("output/openai/claims/nli_score")
DEFAULT_FOLDER_GLOB = "hindi*"
DEFAULT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score Hindi summary claims against source sentences with multilingual NLI."
    )
    parser.add_argument(
        "--claims-roots",
        type=Path,
        nargs="+",
        default=DEFAULT_CLAIMS_ROOTS,
        help="One or more roots containing JSON files with extracted_claims.",
    )
    parser.add_argument(
        "--folder-glob",
        default=DEFAULT_FOLDER_GLOB,
        help="Folder pattern inside each claims root. Example: 'hindi_3'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where NLI-scored JSON files will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face NLI model name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files processed per folder.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Optional cap on source sentences per document.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for sentence-claim NLI scoring.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def iter_target_folders(claims_root, folder_glob):
    if not claims_root.exists():
        return []
    return sorted(path for path in claims_root.glob(folder_glob) if path.is_dir())


def iter_json_files(folder_path, limit=None):
    files = sorted(folder_path.glob("*.json"), key=lambda path: path.stem)
    if limit is not None:
        files = files[:limit]
    return files


def split_into_sentences(text):
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[।!?\.])\s+", cleaned)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def normalize_claims(claims):
    normalized = []
    seen = set()

    for claim in claims:
        if not isinstance(claim, str):
            continue
        clean_claim = re.sub(r"\s+", " ", claim).strip()
        if not clean_claim:
            continue
        key = clean_claim.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(clean_claim)

    return normalized


def get_label_id_map(model):
    raw_map = model.config.id2label
    normalized_map = {int(key): value.lower() for key, value in raw_map.items()}

    entailment_id = next(
        key for key, value in normalized_map.items() if "entail" in value
    )
    contradiction_id = next(
        key for key, value in normalized_map.items() if "contrad" in value
    )
    neutral_id = next(key for key, value in normalized_map.items() if "neutral" in value)

    return {
        "entailment": entailment_id,
        "contradiction": contradiction_id,
        "neutral": neutral_id,
    }


def score_claim_against_sentences(
    claim,
    sentences,
    tokenizer,
    model,
    label_ids,
    device,
    batch_size,
):
    best_result = None

    for start in range(0, len(sentences), batch_size):
        batch_sentences = sentences[start : start + batch_size]
        encoded = tokenizer(
            batch_sentences,
            [claim] * len(batch_sentences),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu()

        for sentence, probs in zip(batch_sentences, probabilities):
            entailment = float(probs[label_ids["entailment"]])
            contradiction = float(probs[label_ids["contradiction"]])
            neutral = float(probs[label_ids["neutral"]])
            support_score = entailment - contradiction

            candidate = {
                "claim": claim,
                "best_supporting_sentence": sentence,
                "entailment_probability": entailment,
                "contradiction_probability": contradiction,
                "neutral_probability": neutral,
                "support_score": support_score,
            }

            if best_result is None or support_score > best_result["support_score"]:
                best_result = candidate

    return best_result


def score_document_claims(
    claims,
    source_text,
    tokenizer,
    model,
    label_ids,
    device,
    batch_size,
    max_sentences,
):
    sentences = split_into_sentences(source_text)
    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    claim_scores = []
    for claim in normalize_claims(claims):
        if not sentences:
            claim_scores.append(
                {
                    "claim": claim,
                    "best_supporting_sentence": None,
                    "entailment_probability": None,
                    "contradiction_probability": None,
                    "neutral_probability": None,
                    "support_score": None,
                }
            )
            continue

        claim_scores.append(
            score_claim_against_sentences(
                claim=claim,
                sentences=sentences,
                tokenizer=tokenizer,
                model=model,
                label_ids=label_ids,
                device=device,
                batch_size=batch_size,
            )
        )

    valid_scores = [
        item["support_score"]
        for item in claim_scores
        if item["support_score"] is not None
    ]
    document_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return sentences, claim_scores, document_score


def process_folder(
    folder_path,
    output_root,
    root_name,
    tokenizer,
    model,
    label_ids,
    device,
    limit,
    max_sentences,
    batch_size,
):
    output_folder = output_root / root_name / folder_path.name
    processed = 0
    skipped = 0

    for file_path in iter_json_files(folder_path, limit=limit):
        try:
            data = load_json(file_path)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        claims = data.get("extracted_claims") or []
        source_text = (data.get("text") or "").strip()
        if not claims or not source_text:
            skipped += 1
            continue

        sentences, claim_scores, document_score = score_document_claims(
            claims=claims,
            source_text=source_text,
            tokenizer=tokenizer,
            model=model,
            label_ids=label_ids,
            device=device,
            batch_size=batch_size,
            max_sentences=max_sentences,
        )

        enriched = dict(data)
        enriched["nli_model"] = model.name_or_path
        enriched["source_sentences"] = sentences
        enriched["claim_nli_scores"] = claim_scores
        enriched["document_support_score"] = document_score

        save_json(output_folder / file_path.name, enriched)
        processed += 1
        print(f"Processed {file_path} -> {output_folder / file_path.name}")

    return processed, skipped


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    label_ids = get_label_id_map(model)

    total_processed = 0
    total_skipped = 0

    for claims_root in args.claims_roots:
        folders = iter_target_folders(claims_root, args.folder_glob)
        if not folders:
            print(f"No folders matched '{args.folder_glob}' inside {claims_root}")
            continue

        root_name = claims_root.name
        for folder in folders:
            processed, skipped = process_folder(
                folder_path=folder,
                output_root=args.output_root,
                root_name=root_name,
                tokenizer=tokenizer,
                model=model,
                label_ids=label_ids,
                device=device,
                limit=args.limit,
                max_sentences=args.max_sentences,
                batch_size=args.batch_size,
            )
            total_processed += processed
            total_skipped += skipped
            print(
                f"{claims_root}/{folder.name}: processed={processed} skipped={skipped} "
                f"output={(args.output_root / root_name / folder.name)}"
            )

    print(
        f"Done. total_processed={total_processed} total_skipped={total_skipped} "
        f"model={args.model_name}"
    )


if __name__ == "__main__":
    main()
