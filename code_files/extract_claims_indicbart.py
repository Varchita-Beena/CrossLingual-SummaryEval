import argparse
import json
import re
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_INPUT_ROOT = Path("output/openai/summary")
DEFAULT_OUTPUT_ROOT = Path("output/claims/indicbartss")
DEFAULT_FOLDER_GLOB = "hindi*"
DEFAULT_MODEL_NAME = "ai4bharat/IndicBARTSS"
DEFAULT_PROMPT = "extract claims from generated summary"
DEFAULT_FALLBACK_MODE = "sentence_split"
DEFAULT_MAX_CLAIMS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract claims from generated Hindi summaries with IndicBARTSS."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing summary folders.",
    )
    parser.add_argument(
        "--folder-glob",
        default=DEFAULT_FOLDER_GLOB,
        help="Folder pattern inside input-root. Example: 'hindi*' or 'hindi_3'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where enriched JSON files with extracted claims will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt prefix used before the generated summary.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files processed per folder.",
    )
    parser.add_argument(
        "--fallback-mode",
        choices=["none", "sentence_split"],
        default=DEFAULT_FALLBACK_MODE,
        help="Fallback used when model output appears corrupted.",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=DEFAULT_MAX_CLAIMS,
        help="Maximum number of claims to keep per summary.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def normalize_claims(text, max_claims):
    cleaned = text.replace("\r", "\n").strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    lines = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        line = re.sub(r"^[\-\*\u2022\d\.\)\s]+", "", line).strip()
        if line:
            lines.append(line)

    if len(lines) > 1:
        return deduplicate_claims(lines, max_claims=max_claims)

    sentence_chunks = re.split(r"(?<=[।!?])\s+", cleaned)
    sentence_chunks = [chunk.strip(" -•\t") for chunk in sentence_chunks if chunk.strip()]
    return deduplicate_claims(sentence_chunks, max_claims=max_claims)


def split_summary_into_claims(summary, max_claims):
    sentence_chunks = re.split(r"(?<=[।!?])\s+", summary.strip())
    sentence_chunks = [chunk.strip() for chunk in sentence_chunks if chunk.strip()]
    return deduplicate_claims(sentence_chunks, max_claims=max_claims)


def deduplicate_claims(claims, max_claims):
    seen = set()
    unique_claims = []

    for claim in claims:
        normalized = re.sub(r"\s+", " ", claim).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_claims.append(normalized)
        if len(unique_claims) >= max_claims:
            break

    return unique_claims


def build_model_input(prompt, summary):
    return f"{prompt}\n\n{summary.strip()}"


def looks_corrupted_output(raw_output):
    text = raw_output.strip()
    if not text:
        return True

    lowered = text.lower()
    if "<s>" in text or "extract claims from generated summary" in lowered:
        return True

    if re.search(r"\b(extra|extract claims)\b", lowered):
        return True

    repeated_noise = re.search(r"(\b\S+\b)(?:\s+\1){3,}", text)
    if repeated_noise:
        return True

    dev_chars = re.findall(r"[\u0900-\u097F]", text)
    vowel_marks = re.findall(r"[\u093E-\u094C\u0962\u0963]", text)
    devanagari_ratio = len(dev_chars) / max(len(text), 1)
    vowel_mark_ratio = len(vowel_marks) / max(len(dev_chars), 1)

    if devanagari_ratio < 0.25:
        return True

    if vowel_mark_ratio < 0.08:
        return True

    return False


def extract_claims(model, tokenizer, summary, prompt, max_new_tokens, max_claims):
    model_input = build_model_input(prompt, summary)
    encoded = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
    )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    return decoded, normalize_claims(decoded, max_claims=max_claims)


def iter_target_folders(input_root, folder_glob):
    return sorted(path for path in input_root.glob(folder_glob) if path.is_dir())


def iter_json_files(folder_path, limit=None):
    files = sorted(folder_path.glob("*.json"), key=lambda path: path.stem)
    if limit is not None:
        files = files[:limit]
    return files


def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=False,
    )


def process_folder(
    folder_path,
    output_root,
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    limit,
    fallback_mode,
    max_claims,
):
    output_folder = output_root / folder_path.name
    processed = 0
    skipped = 0
    fallback_used = 0

    for file_path in iter_json_files(folder_path, limit=limit):
        try:
            data = load_json(file_path)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        generated_summary = (data.get("generated_summary") or "").strip()
        if not generated_summary:
            skipped += 1
            continue

        raw_output, claims = extract_claims(
            model=model,
            tokenizer=tokenizer,
            summary=generated_summary,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            max_claims=max_claims,
        )

        extraction_mode = "model"
        if fallback_mode == "sentence_split" and looks_corrupted_output(raw_output):
            claims = split_summary_into_claims(generated_summary, max_claims=max_claims)
            extraction_mode = "sentence_split_fallback"
            fallback_used += 1

        enriched = dict(data)
        enriched["claim_extraction_model"] = tokenizer.name_or_path
        enriched["claim_extraction_prompt"] = prompt
        enriched["claim_extraction_raw_output"] = raw_output
        enriched["extracted_claims"] = claims
        enriched["claim_extraction_mode"] = extraction_mode
        enriched["claim_extraction_max_claims"] = max_claims

        save_json(output_folder / file_path.name, enriched)
        processed += 1

    return processed, skipped, fallback_used


def main():
    args = parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    folders = iter_target_folders(args.input_root, args.folder_glob)
    if not folders:
        raise FileNotFoundError(
            f"No folders matched '{args.folder_glob}' inside {args.input_root}"
        )

    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    total_processed = 0
    total_skipped = 0
    total_fallback_used = 0

    for folder in folders:
        processed, skipped, fallback_used = process_folder(
            folder_path=folder,
            output_root=args.output_root,
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            fallback_mode=args.fallback_mode,
            max_claims=args.max_claims,
        )
        total_processed += processed
        total_skipped += skipped
        total_fallback_used += fallback_used
        print(
            f"{folder.name}: processed={processed} skipped={skipped} fallback_used={fallback_used} "
            f"output={(args.output_root / folder.name)}"
        )

    print(
        f"Done. total_processed={total_processed} total_skipped={total_skipped} "
        f"total_fallback_used={total_fallback_used} "
        f"model={args.model_name}"
    )


if __name__ == "__main__":
    main()
