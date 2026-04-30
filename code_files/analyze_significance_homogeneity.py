import csv
import json
import math
import re
from itertools import combinations, product
from pathlib import Path
from statistics import mean, median


BASE_DIR = Path("output")
INPUT_ROOTS = {
    "openai": BASE_DIR / "openai" / "embedding" / "significance",
    "sarvam": BASE_DIR / "sarvam" / "embedding" / "significance",
}
LANGUAGES = ("english", "hindi")
RUNS = ("1", "2", "3")
OUTPUT_DIR = BASE_DIR / "analysis" / "significance_homogeneity"
TEXT_FIELD = "significance"
TEXT_LABEL = "significance"


def tokenize(text):
    text = text or ""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return None
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_distance(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def safe_mean(values):
    return mean(values) if values else 0.0


def safe_median(values):
    return median(values) if values else 0.0


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def average_vector(vectors):
    valid_vectors = [vector for vector in vectors if vector is not None]
    if not valid_vectors:
        return None
    dimension = len(valid_vectors[0])
    return [
        sum(vector[i] for vector in valid_vectors) / len(valid_vectors)
        for i in range(dimension)
    ]


def load_run(model, language, run):
    folder = INPUT_ROOTS[model] / f"{language}_{run}"
    items = {}
    for path in sorted(folder.glob("*.json"), key=lambda p: int(p.stem)):
        data = json.loads(path.read_text())
        items[data["id"]] = {
            "file_name": path.name,
            "text_value": data.get(TEXT_FIELD),
            "embedding": data.get("embedding"),
            "title": data.get("title", ""),
            "summary": data.get("summary", ""),
            "url": data.get("url", ""),
        }
    return items


def lexical_metrics(text):
    tokens = tokenize(text)
    unique_tokens = set(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    return {
        "token_count": len(tokens),
        "unique_token_count": len(unique_tokens),
        "type_token_ratio": safe_ratio(len(unique_tokens), len(tokens)),
        "repeated_token_count": len(tokens) - len(unique_tokens),
        "bigram_count": len(bigrams),
        "unique_bigram_count": len(set(bigrams)),
        "bigram_diversity": safe_ratio(len(set(bigrams)), len(bigrams)),
        "trigram_count": len(trigrams),
        "unique_trigram_count": len(set(trigrams)),
        "trigram_diversity": safe_ratio(len(set(trigrams)), len(trigrams)),
        "char_count": len(text or ""),
    }


def compare_records(left, right):
    left_text = left["text_value"] or ""
    right_text = right["text_value"] or ""
    left_tokens = tokenize(left_text)
    right_tokens = tokenize(right_text)
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    union = left_set | right_set
    overlap = left_set & right_set

    return {
        "cosine_similarity": cosine_similarity(left["embedding"], right["embedding"]),
        "word_overlap_count": len(overlap),
        "word_jaccard_similarity": (len(overlap) / len(union)) if union else 1.0,
        "exact_text_match": int(left_text == right_text),
        "left_word_count": len(left_tokens),
        "right_word_count": len(right_tokens),
        "left_char_count": len(left_text),
        "right_char_count": len(right_text),
        "word_count_delta": abs(len(left_tokens) - len(right_tokens)),
        "char_count_delta": abs(len(left_text) - len(right_text)),
    }


def build_group_metrics(records_by_label):
    labels = sorted(records_by_label)
    pair_rows = []

    for left_label, right_label in combinations(labels, 2):
        metrics = compare_records(records_by_label[left_label], records_by_label[right_label])
        metrics["left_label"] = left_label
        metrics["right_label"] = right_label
        pair_rows.append(metrics)

    all_word_counts = {
        label: len(tokenize(record["text_value"]))
        for label, record in records_by_label.items()
    }
    all_char_counts = {
        label: len(record["text_value"] or "")
        for label, record in records_by_label.items()
    }

    longest_word_label = max(all_word_counts, key=all_word_counts.get)
    shortest_word_label = min(all_word_counts, key=all_word_counts.get)
    longest_char_label = max(all_char_counts, key=all_char_counts.get)
    shortest_char_label = min(all_char_counts, key=all_char_counts.get)

    return {
        "pair_rows": pair_rows,
        "word_count_range": max(all_word_counts.values()) - min(all_word_counts.values()),
        "char_count_range": max(all_char_counts.values()) - min(all_char_counts.values()),
        "longest_word_label": longest_word_label,
        "shortest_word_label": shortest_word_label,
        "longest_word_count": all_word_counts[longest_word_label],
        "shortest_word_count": all_word_counts[shortest_word_label],
        "longest_char_label": longest_char_label,
        "shortest_char_label": shortest_char_label,
        "longest_char_count": all_char_counts[longest_char_label],
        "shortest_char_count": all_char_counts[shortest_char_label],
    }


def summarize_pair_rows(pair_rows):
    cosine_scores = [row["cosine_similarity"] for row in pair_rows if row["cosine_similarity"] is not None]
    jaccard_scores = [row["word_jaccard_similarity"] for row in pair_rows]
    word_deltas = [row["word_count_delta"] for row in pair_rows]
    char_deltas = [row["char_count_delta"] for row in pair_rows]
    exact_matches = [row["exact_text_match"] for row in pair_rows]

    return {
        "pair_count": len(pair_rows),
        "cosine_pair_count": len(cosine_scores),
        "cosine_mean": safe_mean(cosine_scores),
        "cosine_median": safe_median(cosine_scores),
        "cosine_min": min(cosine_scores) if cosine_scores else 0.0,
        "cosine_max": max(cosine_scores) if cosine_scores else 0.0,
        "jaccard_mean": safe_mean(jaccard_scores),
        "jaccard_median": safe_median(jaccard_scores),
        "word_count_delta_mean": safe_mean(word_deltas),
        "char_count_delta_mean": safe_mean(char_deltas),
        "exact_match_rate": safe_mean(exact_matches),
    }


def summarize_variation_rows(rows):
    word_ranges = [row["word_count_range"] for row in rows]
    char_ranges = [row["char_count_range"] for row in rows]
    centroid_distances = [
        row["centroid_distance_mean"]
        for row in rows
        if row["centroid_distance_mean"] is not None
    ]
    centroid_cosines = [
        row["centroid_cosine_mean"]
        for row in rows
        if row["centroid_cosine_mean"] is not None
    ]
    return {
        "article_count": len(rows),
        "mean_word_count_range": safe_mean(word_ranges),
        "mean_char_count_range": safe_mean(char_ranges),
        "mean_centroid_distance": safe_mean(centroid_distances),
        "mean_centroid_cosine": safe_mean(centroid_cosines),
    }


def summarize_lexical_rows(rows):
    return {
        "text_count": len(rows),
        "token_count_mean": safe_mean([row["token_count"] for row in rows]),
        "unique_token_count_mean": safe_mean([row["unique_token_count"] for row in rows]),
        "type_token_ratio_mean": safe_mean([row["type_token_ratio"] for row in rows]),
        "repeated_token_count_mean": safe_mean([row["repeated_token_count"] for row in rows]),
        "bigram_diversity_mean": safe_mean([row["bigram_diversity"] for row in rows]),
        "trigram_diversity_mean": safe_mean([row["trigram_diversity"] for row in rows]),
        "char_count_mean": safe_mean([row["char_count"] for row in rows]),
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
        model: {
            language: {
                run: load_run(model, language, run)
                for run in RUNS
            }
            for language in LANGUAGES
        }
        for model in INPUT_ROOTS
    }

    summary_rows = []
    detail_rows = []
    variation_rows = []
    centroid_rows = []
    lexical_rows = []
    lexical_summary_rows = []
    boxplot_rows = []
    example_rows = []

    for language in LANGUAGES:
        for model in INPUT_ROOTS:
            for run in RUNS:
                for article_id, record in data[model][language][run].items():
                    metrics = lexical_metrics(record["text_value"] or "")
                    lexical_rows.append({
                        "language": language,
                        "model": model,
                        "run": run,
                        "article_id": article_id,
                        "title": record["title"],
                        TEXT_LABEL: record["text_value"] or "",
                        **metrics,
                    })

            model_rows = [
                row for row in lexical_rows
                if row["language"] == language and row["model"] == model
            ]
            lexical_summary_rows.append({
                "language": language,
                "model": model,
                **summarize_lexical_rows(model_rows),
            })

    for language in LANGUAGES:
        for model in INPUT_ROOTS:
            run_pairs = list(combinations(RUNS, 2))
            per_pair_rows = []
            per_article_variation_rows = []
            common_ids = sorted(
                set.intersection(*(set(data[model][language][run]) for run in RUNS))
            )

            for article_id in common_ids:
                records = {
                    f"{model}_{run}": data[model][language][run][article_id]
                    for run in RUNS
                }
                group_metrics = build_group_metrics(records)
                valid_embeddings = [
                    records[f"{model}_{run}"]["embedding"]
                    for run in RUNS
                    if records[f"{model}_{run}"]["embedding"] is not None
                ]
                centroid = average_vector(valid_embeddings)
                per_run_centroid_distances = []
                per_run_centroid_cosines = []
                for run in RUNS:
                    label = f"{model}_{run}"
                    embedding = records[label]["embedding"]
                    centroid_distance = euclidean_distance(embedding, centroid)
                    centroid_cosine = cosine_similarity(embedding, centroid)
                    centroid_rows.append({
                        "comparison_scope": "intra_model",
                        "language": language,
                        "model_or_pair": model,
                        "article_id": article_id,
                        "title": records[f"{model}_1"]["title"],
                        "label": label,
                        "centroid_distance": centroid_distance,
                        "centroid_cosine": centroid_cosine,
                    })
                    if centroid_distance is not None:
                        per_run_centroid_distances.append(centroid_distance)
                    if centroid_cosine is not None:
                        per_run_centroid_cosines.append(centroid_cosine)

                variation_entry = {
                    "comparison_scope": "intra_model",
                    "language": language,
                    "model_or_pair": model,
                    "article_id": article_id,
                    "title": records[f"{model}_1"]["title"],
                    "word_count_range": group_metrics["word_count_range"],
                    "char_count_range": group_metrics["char_count_range"],
                    "longest_word_label": group_metrics["longest_word_label"],
                    "longest_word_count": group_metrics["longest_word_count"],
                    "shortest_word_label": group_metrics["shortest_word_label"],
                    "shortest_word_count": group_metrics["shortest_word_count"],
                    "longest_char_label": group_metrics["longest_char_label"],
                    "longest_char_count": group_metrics["longest_char_count"],
                    "shortest_char_label": group_metrics["shortest_char_label"],
                    "shortest_char_count": group_metrics["shortest_char_count"],
                    "centroid_distance_mean": safe_mean(per_run_centroid_distances) if per_run_centroid_distances else None,
                    "centroid_distance_min": min(per_run_centroid_distances) if per_run_centroid_distances else None,
                    "centroid_distance_max": max(per_run_centroid_distances) if per_run_centroid_distances else None,
                    "centroid_cosine_mean": safe_mean(per_run_centroid_cosines) if per_run_centroid_cosines else None,
                    "centroid_cosine_min": min(per_run_centroid_cosines) if per_run_centroid_cosines else None,
                    "centroid_cosine_max": max(per_run_centroid_cosines) if per_run_centroid_cosines else None,
                }
                variation_rows.append(variation_entry)
                per_article_variation_rows.append(variation_entry)

                for pair_row in group_metrics["pair_rows"]:
                    detail_rows.append({
                        "comparison_scope": "intra_model",
                        "language": language,
                        "model_or_pair": model,
                        "article_id": article_id,
                        "title": records[f"{model}_1"]["title"],
                        "left_label": pair_row["left_label"],
                        "right_label": pair_row["right_label"],
                        f"left_{TEXT_LABEL}": records[pair_row["left_label"]]["text_value"],
                        f"right_{TEXT_LABEL}": records[pair_row["right_label"]]["text_value"],
                        **pair_row,
                    })
                    for metric_name in (
                        "cosine_similarity",
                        "word_jaccard_similarity",
                        "word_count_delta",
                        "char_count_delta",
                    ):
                        boxplot_rows.append({
                            "comparison_scope": "intra_model",
                            "language": language,
                            "model_or_pair": model,
                            "article_id": article_id,
                            "metric": metric_name,
                            "value": pair_row[metric_name],
                        })
                    per_pair_rows.append(pair_row)

            summary_rows.append({
                "comparison_scope": "intra_model",
                "language": language,
                "model_or_pair": model,
                "article_count": len(common_ids),
                "run_pairs_considered": len(run_pairs),
                **summarize_pair_rows(per_pair_rows),
                **summarize_variation_rows(per_article_variation_rows),
            })

            repetitive_sorted = sorted(
                per_article_variation_rows,
                key=lambda row: (
                    -(row["centroid_cosine_mean"] if row["centroid_cosine_mean"] is not None else -1),
                    row["word_count_range"],
                    row["char_count_range"],
                ),
            )[:20]
            variable_sorted = sorted(
                per_article_variation_rows,
                key=lambda row: (
                    row["centroid_cosine_mean"] if row["centroid_cosine_mean"] is not None else float("inf"),
                    -row["word_count_range"],
                    -row["char_count_range"],
                ),
            )[:20]

            for rank, row in enumerate(repetitive_sorted, start=1):
                example_rows.append({
                    "example_type": "most_repetitive",
                    "comparison_scope": "intra_model",
                    "language": language,
                    "model_or_pair": model,
                    "rank": rank,
                    **row,
                })
            for rank, row in enumerate(variable_sorted, start=1):
                example_rows.append({
                    "example_type": "most_variable",
                    "comparison_scope": "intra_model",
                    "language": language,
                    "model_or_pair": model,
                    "rank": rank,
                    **row,
                })

        model_pair_name = "openai_vs_sarvam"
        common_ids = sorted(
            set.intersection(
                *(set(data["openai"][language][run]) for run in RUNS),
                *(set(data["sarvam"][language][run]) for run in RUNS),
            )
        )
        inter_pair_rows = []
        inter_article_variation_rows = []

        for article_id in common_ids:
            records = {
                **{f"openai_{run}": data["openai"][language][run][article_id] for run in RUNS},
                **{f"sarvam_{run}": data["sarvam"][language][run][article_id] for run in RUNS},
            }
            group_metrics = build_group_metrics(records)
            valid_embeddings = [
                records[label]["embedding"]
                for label in records
                if records[label]["embedding"] is not None
            ]
            centroid = average_vector(valid_embeddings)
            per_run_centroid_distances = []
            per_run_centroid_cosines = []
            for label, record in records.items():
                centroid_distance = euclidean_distance(record["embedding"], centroid)
                centroid_cosine = cosine_similarity(record["embedding"], centroid)
                centroid_rows.append({
                    "comparison_scope": "inter_model",
                    "language": language,
                    "model_or_pair": model_pair_name,
                    "article_id": article_id,
                    "title": records["openai_1"]["title"],
                    "label": label,
                    "centroid_distance": centroid_distance,
                    "centroid_cosine": centroid_cosine,
                })
                if centroid_distance is not None:
                    per_run_centroid_distances.append(centroid_distance)
                if centroid_cosine is not None:
                    per_run_centroid_cosines.append(centroid_cosine)

            variation_entry = {
                "comparison_scope": "inter_model",
                "language": language,
                "model_or_pair": model_pair_name,
                "article_id": article_id,
                "title": records["openai_1"]["title"],
                "word_count_range": group_metrics["word_count_range"],
                "char_count_range": group_metrics["char_count_range"],
                "longest_word_label": group_metrics["longest_word_label"],
                "longest_word_count": group_metrics["longest_word_count"],
                "shortest_word_label": group_metrics["shortest_word_label"],
                "shortest_word_count": group_metrics["shortest_word_count"],
                "longest_char_label": group_metrics["longest_char_label"],
                "longest_char_count": group_metrics["longest_char_count"],
                "shortest_char_label": group_metrics["shortest_char_label"],
                "shortest_char_count": group_metrics["shortest_char_count"],
                "centroid_distance_mean": safe_mean(per_run_centroid_distances) if per_run_centroid_distances else None,
                "centroid_distance_min": min(per_run_centroid_distances) if per_run_centroid_distances else None,
                "centroid_distance_max": max(per_run_centroid_distances) if per_run_centroid_distances else None,
                "centroid_cosine_mean": safe_mean(per_run_centroid_cosines) if per_run_centroid_cosines else None,
                "centroid_cosine_min": min(per_run_centroid_cosines) if per_run_centroid_cosines else None,
                "centroid_cosine_max": max(per_run_centroid_cosines) if per_run_centroid_cosines else None,
            }
            variation_rows.append(variation_entry)
            inter_article_variation_rows.append(variation_entry)

            for openai_run, sarvam_run in product(RUNS, RUNS):
                left_label = f"openai_{openai_run}"
                right_label = f"sarvam_{sarvam_run}"
                pair_row = compare_records(records[left_label], records[right_label])
                pair_row["left_label"] = left_label
                pair_row["right_label"] = right_label
                inter_pair_rows.append(pair_row)
                detail_rows.append({
                    "comparison_scope": "inter_model",
                    "language": language,
                    "model_or_pair": model_pair_name,
                    "article_id": article_id,
                    "title": records["openai_1"]["title"],
                    "left_label": left_label,
                    "right_label": right_label,
                    f"left_{TEXT_LABEL}": records[left_label]["text_value"],
                    f"right_{TEXT_LABEL}": records[right_label]["text_value"],
                    **pair_row,
                })
                for metric_name in (
                    "cosine_similarity",
                    "word_jaccard_similarity",
                    "word_count_delta",
                    "char_count_delta",
                ):
                    boxplot_rows.append({
                        "comparison_scope": "inter_model",
                        "language": language,
                        "model_or_pair": model_pair_name,
                        "article_id": article_id,
                        "metric": metric_name,
                        "value": pair_row[metric_name],
                    })

        summary_rows.append({
            "comparison_scope": "inter_model",
            "language": language,
            "model_or_pair": model_pair_name,
            "article_count": len(common_ids),
            "run_pairs_considered": 9,
            **summarize_pair_rows(inter_pair_rows),
            **summarize_variation_rows(inter_article_variation_rows),
        })

        repetitive_sorted = sorted(
            inter_article_variation_rows,
            key=lambda row: (
                -(row["centroid_cosine_mean"] if row["centroid_cosine_mean"] is not None else -1),
                row["word_count_range"],
                row["char_count_range"],
            ),
        )[:20]
        variable_sorted = sorted(
            inter_article_variation_rows,
            key=lambda row: (
                row["centroid_cosine_mean"] if row["centroid_cosine_mean"] is not None else float("inf"),
                -row["word_count_range"],
                -row["char_count_range"],
            ),
        )[:20]
        for rank, row in enumerate(repetitive_sorted, start=1):
            example_rows.append({
                "example_type": "most_repetitive",
                "comparison_scope": "inter_model",
                "language": language,
                "model_or_pair": model_pair_name,
                "rank": rank,
                **row,
            })
        for rank, row in enumerate(variable_sorted, start=1):
            example_rows.append({
                "example_type": "most_variable",
                "comparison_scope": "inter_model",
                "language": language,
                "model_or_pair": model_pair_name,
                "rank": rank,
                **row,
            })

    write_csv(
        OUTPUT_DIR / "summary.csv",
        summary_rows,
        [
            "comparison_scope",
            "language",
            "model_or_pair",
            "article_count",
            "run_pairs_considered",
            "pair_count",
            "cosine_pair_count",
            "cosine_mean",
            "cosine_median",
            "cosine_min",
            "cosine_max",
            "jaccard_mean",
            "jaccard_median",
            "word_count_delta_mean",
            "char_count_delta_mean",
            "exact_match_rate",
            "mean_word_count_range",
            "mean_char_count_range",
            "mean_centroid_distance",
            "mean_centroid_cosine",
        ],
    )

    write_csv(
        OUTPUT_DIR / "pairwise_details.csv",
        detail_rows,
        [
            "comparison_scope",
            "language",
            "model_or_pair",
            "article_id",
            "title",
            "left_label",
            "right_label",
            f"left_{TEXT_LABEL}",
            f"right_{TEXT_LABEL}",
            "cosine_similarity",
            "word_overlap_count",
            "word_jaccard_similarity",
            "exact_text_match",
            "left_word_count",
            "right_word_count",
            "left_char_count",
            "right_char_count",
            "word_count_delta",
            "char_count_delta",
        ],
    )

    write_csv(
        OUTPUT_DIR / "text_variation.csv",
        variation_rows,
        [
            "comparison_scope",
            "language",
            "model_or_pair",
            "article_id",
            "title",
            "word_count_range",
            "char_count_range",
            "longest_word_label",
            "longest_word_count",
            "shortest_word_label",
            "shortest_word_count",
            "longest_char_label",
            "longest_char_count",
            "shortest_char_label",
            "shortest_char_count",
            "centroid_distance_mean",
            "centroid_distance_min",
            "centroid_distance_max",
            "centroid_cosine_mean",
            "centroid_cosine_min",
            "centroid_cosine_max",
        ],
    )

    write_csv(
        OUTPUT_DIR / "centroid_analysis.csv",
        centroid_rows,
        [
            "comparison_scope",
            "language",
            "model_or_pair",
            "article_id",
            "title",
            "label",
            "centroid_distance",
            "centroid_cosine",
        ],
    )

    write_csv(
        OUTPUT_DIR / "lexical_diversity.csv",
        lexical_rows,
        [
            "language",
            "model",
            "run",
            "article_id",
            "title",
            TEXT_LABEL,
            "token_count",
            "unique_token_count",
            "type_token_ratio",
            "repeated_token_count",
            "bigram_count",
            "unique_bigram_count",
            "bigram_diversity",
            "trigram_count",
            "unique_trigram_count",
            "trigram_diversity",
            "char_count",
        ],
    )

    write_csv(
        OUTPUT_DIR / "lexical_summary.csv",
        lexical_summary_rows,
        [
            "language",
            "model",
            "text_count",
            "token_count_mean",
            "unique_token_count_mean",
            "type_token_ratio_mean",
            "repeated_token_count_mean",
            "bigram_diversity_mean",
            "trigram_diversity_mean",
            "char_count_mean",
        ],
    )

    write_csv(
        OUTPUT_DIR / "boxplot_metrics.csv",
        [row for row in boxplot_rows if row["value"] is not None],
        [
            "comparison_scope",
            "language",
            "model_or_pair",
            "article_id",
            "metric",
            "value",
        ],
    )

    write_csv(
        OUTPUT_DIR / "top_examples.csv",
        example_rows,
        [
            "example_type",
            "comparison_scope",
            "language",
            "model_or_pair",
            "rank",
            "article_id",
            "title",
            "word_count_range",
            "char_count_range",
            "longest_word_label",
            "longest_word_count",
            "shortest_word_label",
            "shortest_word_count",
            "longest_char_label",
            "longest_char_count",
            "shortest_char_label",
            "shortest_char_count",
            "centroid_distance_mean",
            "centroid_distance_min",
            "centroid_distance_max",
            "centroid_cosine_mean",
            "centroid_cosine_min",
            "centroid_cosine_max",
        ],
    )

    report_lines = [
        "# Significance Homogeneity Report",
        "",
        "This report compares significance-text embeddings and text variation across repeated runs.",
        "",
        "## What is measured",
        "",
        "- Intra-model homogeneity: compare the 3 runs within the same model and language.",
        "- Inter-model homogeneity: compare all 9 OpenAI-vs-Sarvam run pairs within the same language.",
        "- Cosine similarity: checks how close the significance embeddings are.",
        "- Word Jaccard similarity: checks token overlap between two significance texts.",
        "- Exact match rate: fraction of pairwise comparisons where the significance text is identical.",
        "- Length variation: range of word counts and character counts for the same article.",
        "",
        "## Files",
        "",
        f"- `summary.csv`: one row per analysis block in `{OUTPUT_DIR}`.",
        "- `pairwise_details.csv`: one row per article-level pairwise comparison.",
        "- `text_variation.csv`: longest/shortest significance diagnostics for each article.",
        "- `centroid_analysis.csv`: per-run distance and cosine against the article centroid.",
        "- `lexical_diversity.csv`: per-text lexical diversity metrics.",
        "- `lexical_summary.csv`: average lexical diversity metrics per model and language.",
        "- `boxplot_metrics.csv`: long-format metrics ready for boxplots.",
        "- `top_examples.csv`: top 20 most repetitive and most variable examples per block.",
        "",
        "## Summary Snapshot",
        "",
    ]

    for row in summary_rows:
        report_lines.append(
            f"- {row['comparison_scope']} | {row['language']} | {row['model_or_pair']}: "
            f"cosine_mean={row['cosine_mean']:.4f}, "
            f"jaccard_mean={row['jaccard_mean']:.4f}, "
            f"exact_match_rate={row['exact_match_rate']:.4f}, "
            f"centroid_cosine_mean={row['mean_centroid_cosine']:.4f}"
        )

    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote outputs to: {OUTPUT_DIR}")
    print("Created files:")
    print(f"  - {OUTPUT_DIR / 'summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'pairwise_details.csv'}")
    print(f"  - {OUTPUT_DIR / 'text_variation.csv'}")
    print(f"  - {OUTPUT_DIR / 'centroid_analysis.csv'}")
    print(f"  - {OUTPUT_DIR / 'lexical_diversity.csv'}")
    print(f"  - {OUTPUT_DIR / 'lexical_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'boxplot_metrics.csv'}")
    print(f"  - {OUTPUT_DIR / 'top_examples.csv'}")
    print(f"  - {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
