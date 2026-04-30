import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI


SOURCE_ROOTS = {
    "openai": Path("output/openai/nearest"),
    "sarvam": Path("output/sarvam/nearest"),
}
DEFAULT_OUTPUT_ROOT = Path("output/derived/nearest_event_extraction")
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_FOLDER_GLOB = "*_1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract canonical nearest-event labels from nearest-task outputs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where extracted label JSON files will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="OpenAI model name for extraction.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--folder-glob",
        default=DEFAULT_FOLDER_GLOB,
        help="Folder pattern inside each nearest root. Default processes only *_1 folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files processed per folder.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=350,
        help="Maximum completion tokens for each extraction request.",
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
    files = sorted(folder_path.glob("*.json"), key=lambda path: int(path.stem))
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


def build_messages(nearest_text, language):
    language_name = "Hindi" if language == "hindi" else "English"
    system_message = (
        "You extract the main recalled historical analogy from a model-generated response. "
        "Return only valid JSON."
    )
    user_message = (
        "You will receive a nearest-event explanation generated from a news article.\n"
        f"The response language is {language_name}.\n"
        "Extract the single strongest recalled analogy.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "reference_kind": "specific_event" | "broader_pattern" | "mixed" | "unclear",\n'
        '  "event_label_original": "short phrase in the original response language, or empty string",\n'
        '  "canonical_event_label_english": "short canonical English event name, or empty string",\n'
        '  "canonical_broader_pattern_english": "short canonical English broader pattern, or empty string",\n'
        '  "event_type": "war_conflict" | "election_politics" | "disaster" | "terrorism" | "economic_crisis" | "protest_social_unrest" | "pandemic_health" | "institutional_breakdown" | "other",\n'
        '  "year_or_period": "year or period if clearly mentioned, else empty string",\n'
        '  "country_or_region": "country or region if clearly implied, else empty string",\n'
        '  "short_rationale": "one short sentence explaining the extracted analogy"\n'
        "}\n\n"
        "Rules:\n"
        "- Choose the primary analogy only.\n"
        "- If the response names a specific event, fill canonical_event_label_english.\n"
        "- If the response mainly refers to a broad pattern, fill canonical_broader_pattern_english.\n"
        "- If both appear, choose the dominant one and set reference_kind to mixed.\n"
        "- Keep fields concise.\n"
        "- Do not invent missing information.\n"
        "- Return only the JSON object.\n\n"
        f"Nearest-event response:\n{nearest_text.strip()}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def normalize_extraction(parsed):
    fields = {
        "reference_kind": "",
        "event_label_original": "",
        "canonical_event_label_english": "",
        "canonical_broader_pattern_english": "",
        "event_type": "",
        "year_or_period": "",
        "country_or_region": "",
        "short_rationale": "",
    }
    normalized = {}
    for key, default in fields.items():
        value = parsed.get(key, default)
        normalized[key] = value.strip() if isinstance(value, str) else default
    return normalized


def extract_event_label(client, model_name, nearest_text, language, max_completion_tokens):
    response = client.chat.completions.create(
        model=model_name,
        messages=build_messages(nearest_text, language),
        temperature=0,
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content or ""
    parsed = extract_json_block(raw_text)
    usage = response.usage.model_dump() if response.usage else None
    return raw_text, normalize_extraction(parsed), usage


def process_folder(
    source_model,
    folder_path,
    output_root,
    client,
    model_name,
    limit,
    max_completion_tokens,
):
    language = folder_path.name.split("_", 1)[0]
    output_folder = output_root / source_model / folder_path.name
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

        nearest_text = (data.get("nearest") or "").strip()
        if not nearest_text:
            skipped += 1
            continue

        raw_output, extracted, usage = extract_event_label(
            client=client,
            model_name=model_name,
            nearest_text=nearest_text,
            language=language,
            max_completion_tokens=max_completion_tokens,
        )

        enriched = dict(data)
        enriched["source_model"] = source_model
        enriched["nearest_event_extraction_model"] = model_name
        enriched["nearest_event_extraction_raw_output"] = raw_output
        enriched["nearest_event_extraction"] = extracted
        enriched["openai_usage"] = usage

        save_json(output_file_path, enriched)
        processed += 1
        print(f"Processed {file_path} -> {output_file_path}")

    return processed, skipped, skipped_existing


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    total_processed = 0
    total_skipped = 0

    for source_model, input_root in SOURCE_ROOTS.items():
        if not input_root.exists():
            raise FileNotFoundError(f"Input root does not exist: {input_root}")
        folders = iter_target_folders(input_root, args.folder_glob)
        if not folders:
            raise FileNotFoundError(
                f"No folders matched '{args.folder_glob}' inside {input_root}"
            )

        for folder in folders:
            processed, skipped, skipped_existing = process_folder(
                source_model=source_model,
                folder_path=folder,
                output_root=args.output_root,
                client=client,
                model_name=args.model_name,
                limit=args.limit,
                max_completion_tokens=args.max_completion_tokens,
            )
            total_processed += processed
            total_skipped += skipped
            print(
                f"{source_model}/{folder.name}: processed={processed} skipped={skipped} "
                f"skipped_existing={skipped_existing} output={(args.output_root / source_model / folder.name)}"
            )

    print(
        f"Done. total_processed={total_processed} total_skipped={total_skipped} "
        f"model={args.model_name}"
    )


if __name__ == "__main__":
    main()
