import csv
import json
import re
from pathlib import Path
from statistics import mean


INPUT_ROOT = Path("output/derived/nearest_event_extraction")
OUTPUT_DIR = Path("output/analysis/nearest_event_label_consistency_run1")
SOURCE_MODELS = ("openai", "sarvam")
LANGUAGES = ("english", "hindi")
RUN = "1"


def safe_mean(values):
    return mean(values) if values else 0.0


def normalize_label(text):
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return re.findall(r"\w+", normalize_label(text), flags=re.UNICODE)


def set_jaccard(left_set, right_set):
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def load_run(source_model, language):
    folder = INPUT_ROOT / source_model / f"{language}_{RUN}"
    items = {}
    for path in sorted(folder.glob("*.json"), key=lambda p: int(p.stem)):
        data = json.loads(path.read_text())
        extraction = data.get("nearest_event_extraction", {})
        primary_label = (
            extraction.get("canonical_event_label_english")
            or extraction.get("canonical_broader_pattern_english")
            or ""
        )
        items[data["id"]] = {
            "file_name": path.name,
            "title": data.get("title", ""),
            "nearest": data.get("nearest", ""),
            "reference_kind": extraction.get("reference_kind", ""),
            "event_label_original": extraction.get("event_label_original", ""),
            "canonical_event_label_english": extraction.get("canonical_event_label_english", ""),
            "canonical_broader_pattern_english": extraction.get("canonical_broader_pattern_english", ""),
            "event_type": extraction.get("event_type", ""),
            "year_or_period": extraction.get("year_or_period", ""),
            "country_or_region": extraction.get("country_or_region", ""),
            "short_rationale": extraction.get("short_rationale", ""),
            "primary_label": primary_label,
            "normalized_primary_label": normalize_label(primary_label),
        }
    return items


def compare_records(left, right):
    left_label = left["normalized_primary_label"]
    right_label = right["normalized_primary_label"]
    left_tokens = set(tokenize(left_label))
    right_tokens = set(tokenize(right_label))

    return {
        "exact_primary_label_match": int(left_label == right_label and left_label != ""),
        "primary_label_jaccard": set_jaccard(left_tokens, right_tokens),
        "exact_event_label_match": int(
            normalize_label(left["canonical_event_label_english"]) ==
            normalize_label(right["canonical_event_label_english"]) and
            normalize_label(left["canonical_event_label_english"]) != ""
        ),
        "exact_broader_pattern_match": int(
            normalize_label(left["canonical_broader_pattern_english"]) ==
            normalize_label(right["canonical_broader_pattern_english"]) and
            normalize_label(left["canonical_broader_pattern_english"]) != ""
        ),
        "same_reference_kind": int(left["reference_kind"] == right["reference_kind"]),
        "same_event_type": int(left["event_type"] == right["event_type"]),
        "same_year_or_period": int(
            normalize_label(left["year_or_period"]) == normalize_label(right["year_or_period"]) and
            normalize_label(left["year_or_period"]) != ""
        ),
        "same_country_or_region": int(
            normalize_label(left["country_or_region"]) == normalize_label(right["country_or_region"]) and
            normalize_label(left["country_or_region"]) != ""
        ),
        "left_label_word_count": len(tokenize(left["primary_label"])),
        "right_label_word_count": len(tokenize(right["primary_label"])),
        "left_nearest_word_count": len(tokenize(left["nearest"])),
        "right_nearest_word_count": len(tokenize(right["nearest"])),
        "label_word_count_delta": abs(len(tokenize(left["primary_label"])) - len(tokenize(right["primary_label"]))),
        "nearest_word_count_delta": abs(len(tokenize(left["nearest"])) - len(tokenize(right["nearest"]))),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        source_model: {
            language: load_run(source_model, language)
            for language in LANGUAGES
        }
        for source_model in SOURCE_MODELS
    }

    summary_rows = []
    detail_rows = []

    for language in LANGUAGES:
        common_ids = sorted(
            set(data["openai"][language]).intersection(set(data["sarvam"][language]))
        )
        pair_rows = []

        for article_id in common_ids:
            left = data["openai"][language][article_id]
            right = data["sarvam"][language][article_id]
            metrics = compare_records(left, right)
            pair_rows.append(metrics)
            detail_rows.append({
                "language": language,
                "article_id": article_id,
                "title": left["title"],
                "openai_primary_label": left["primary_label"],
                "sarvam_primary_label": right["primary_label"],
                "openai_event_type": left["event_type"],
                "sarvam_event_type": right["event_type"],
                "openai_reference_kind": left["reference_kind"],
                "sarvam_reference_kind": right["reference_kind"],
                **metrics,
            })

        summary_rows.append({
            "language": language,
            "article_count": len(common_ids),
            "exact_primary_label_match_rate": safe_mean([row["exact_primary_label_match"] for row in pair_rows]),
            "primary_label_jaccard_mean": safe_mean([row["primary_label_jaccard"] for row in pair_rows]),
            "exact_event_label_match_rate": safe_mean([row["exact_event_label_match"] for row in pair_rows]),
            "exact_broader_pattern_match_rate": safe_mean([row["exact_broader_pattern_match"] for row in pair_rows]),
            "same_reference_kind_rate": safe_mean([row["same_reference_kind"] for row in pair_rows]),
            "same_event_type_rate": safe_mean([row["same_event_type"] for row in pair_rows]),
            "same_year_or_period_rate": safe_mean([row["same_year_or_period"] for row in pair_rows]),
            "same_country_or_region_rate": safe_mean([row["same_country_or_region"] for row in pair_rows]),
            "label_word_count_delta_mean": safe_mean([row["label_word_count_delta"] for row in pair_rows]),
            "nearest_word_count_delta_mean": safe_mean([row["nearest_word_count_delta"] for row in pair_rows]),
        })

    write_csv(
        OUTPUT_DIR / "summary.csv",
        summary_rows,
        [
            "language", "article_count", "exact_primary_label_match_rate",
            "primary_label_jaccard_mean", "exact_event_label_match_rate",
            "exact_broader_pattern_match_rate", "same_reference_kind_rate",
            "same_event_type_rate", "same_year_or_period_rate",
            "same_country_or_region_rate", "label_word_count_delta_mean",
            "nearest_word_count_delta_mean",
        ],
    )

    write_csv(
        OUTPUT_DIR / "pairwise_details.csv",
        detail_rows,
        [
            "language", "article_id", "title", "openai_primary_label",
            "sarvam_primary_label", "openai_event_type", "sarvam_event_type",
            "openai_reference_kind", "sarvam_reference_kind",
            "exact_primary_label_match", "primary_label_jaccard",
            "exact_event_label_match", "exact_broader_pattern_match",
            "same_reference_kind", "same_event_type", "same_year_or_period",
            "same_country_or_region", "left_label_word_count",
            "right_label_word_count", "left_nearest_word_count",
            "right_nearest_word_count", "label_word_count_delta",
            "nearest_word_count_delta",
        ],
    )

    report_lines = [
        "# Nearest Event Label Consistency Report (Run 1 Only)",
        "",
        "This report compares extracted canonical nearest-event labels for `english_1` and `hindi_1` between OpenAI and Sarvam.",
        "",
        "## Summary Snapshot",
        "",
    ]
    for row in summary_rows:
        report_lines.append(
            f"- {row['language']}: "
            f"exact_primary_label_match_rate={row['exact_primary_label_match_rate']:.4f}, "
            f"same_event_type_rate={row['same_event_type_rate']:.4f}, "
            f"primary_label_jaccard_mean={row['primary_label_jaccard_mean']:.4f}"
        )
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote outputs to: {OUTPUT_DIR}")
    print("Created files:")
    print(f"  - {OUTPUT_DIR / 'summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'pairwise_details.csv'}")
    print(f"  - {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
