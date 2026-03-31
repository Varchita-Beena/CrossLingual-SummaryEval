import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI


DEFAULT_INPUT_ROOT = Path("output/openai/summary")
DEFAULT_OUTPUT_ROOT = Path("output/openai/claims")
DEFAULT_FOLDER_GLOB = "hindi_1*"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_MAX_CLAIMS = 1001


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract claims from generated Hindi summaries using OpenAI."
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
        help="Folder pattern inside input-root. Example: 'hindi_3'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where enriched JSON files will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Optional cap on number of files processed per folder.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=600,
        help="Maximum completion tokens for each request.",
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


def iter_target_folders(input_root, folder_glob):
    return sorted(path for path in input_root.glob(folder_glob) if path.is_dir())


def iter_json_files(folder_path, limit=None):
    files = sorted(folder_path.glob("*.json"), key=lambda path: path.stem)
    if limit is not None:
        files = files[:limit]
    return files


def extract_json_block(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    object_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if object_match:
        return json.loads(object_match.group(1))

    raise ValueError("Could not parse JSON from model response.")


def extract_claims_from_text(text, max_claims):
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[\-\*\u2022\d\.\)\s]+", "", line).strip()
        if line:
            lines.append(line)

    if not lines:
        sentence_chunks = re.split(r"(?<=[।!?])\s+", text.strip())
        lines = [chunk.strip() for chunk in sentence_chunks if chunk.strip()]

    return normalize_claims(lines, max_claims=max_claims)


def build_messages(summary, max_claims):
    system_message = (
        "You extract atomic factual claims from Hindi news summaries. "
        "Return only valid JSON."
    )
    user_message = (
        "Extract claims from generated summary.\n"
        "Return JSON with this shape: {\"claims\": [\"claim 1\", \"claim 2\"]}.\n"
        "Rules:\n"
        "- Keep claims in Hindi.\n"
        "- Keep claims factual and concise.\n"
        "- Split combined facts into separate atomic claims when possible.\n"
        f"- Return at most {max_claims} claims.\n"
        "- Do not add information not present in the summary.\n"
        "- Return only the JSON object.\n\n"
        f"Generated summary:\n{summary.strip()}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def normalize_claims(claims, max_claims):
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
        if len(normalized) >= max_claims:
            break

    return normalized


def extract_claims(client, model_name, summary, max_completion_tokens, max_claims):
    response = client.chat.completions.create(
        model=model_name,
        messages=build_messages(summary, max_claims=max_claims),
        temperature=0,
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content or ""
    try:
        parsed = extract_json_block(raw_text)
        claims = normalize_claims(parsed.get("claims", []), max_claims=max_claims)
    except (ValueError, json.JSONDecodeError):
        claims = extract_claims_from_text(raw_text, max_claims=max_claims)
    usage = response.usage.model_dump() if response.usage else None
    return raw_text, claims, usage


def process_folder(
    folder_path,
    output_root,
    client,
    model_name,
    limit,
    max_completion_tokens,
    max_claims,
):
    output_folder = output_root / folder_path.name
    processed = 0
    skipped = 0
    skipped_existing = 0

    for file_path in iter_json_files(folder_path, limit=limit):
        output_file_path = output_folder / file_path.name
        if output_file_path.exists():
            skipped_existing += 1
            continue

        try:
            data = load_json(file_path)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        generated_summary = (data.get("generated_summary") or "").strip()
        if not generated_summary:
            skipped += 1
            continue

        raw_output, claims, usage = extract_claims(
            client=client,
            model_name=model_name,
            summary=generated_summary,
            max_completion_tokens=max_completion_tokens,
            max_claims=max_claims,
        )

        enriched = dict(data)
        enriched["claim_extraction_model"] = model_name
        enriched["claim_extraction_raw_output"] = raw_output
        enriched["extracted_claims"] = claims
        enriched["claim_extraction_max_claims"] = max_claims
        enriched["openai_usage"] = usage

        save_json(output_file_path, enriched)
        processed += 1
        print(f"Processed {file_path.name} -> {output_file_path}")

    return processed, skipped, skipped_existing


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    folders = iter_target_folders(args.input_root, args.folder_glob)
    if not folders:
        raise FileNotFoundError(
            f"No folders matched '{args.folder_glob}' inside {args.input_root}"
        )

    client = OpenAI(api_key=api_key)

    total_processed = 0
    total_skipped = 0

    for folder in folders:
        processed, skipped, skipped_existing = process_folder(
            folder_path=folder,
            output_root=args.output_root,
            client=client,
            model_name=args.model_name,
            limit=args.limit,
            max_completion_tokens=args.max_completion_tokens,
            max_claims=args.max_claims,
        )
        total_processed += processed
        total_skipped += skipped
        print(
            f"{folder.name}: processed={processed} skipped={skipped} skipped_existing={skipped_existing} "
            f"output={(args.output_root / folder.name)}"
        )

    print(
        f"Done. total_processed={total_processed} total_skipped={total_skipped} "
        f"model={args.model_name}"
    )


if __name__ == "__main__":
    main()
