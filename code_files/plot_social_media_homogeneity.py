import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path("output/analysis/social_media_homogeneity")
INPUT_CSV = BASE_DIR / "boxplot_metrics.csv"
OUTPUT_DIR = BASE_DIR / "plots"
METRICS = (
    "cosine_similarity",
    "word_jaccard_similarity",
    "word_count_delta",
    "char_count_delta",
)
LANGUAGES = ("english", "hindi")
GROUP_ORDER = {
    "intra_model": ("openai", "sarvam"),
    "inter_model": ("openai_vs_sarvam",),
}
DISPLAY_NAMES = {
    "openai": "OpenAI",
    "sarvam": "Sarvam",
    "openai_vs_sarvam": "OpenAI vs Sarvam",
    "cosine_similarity": "Cosine Similarity",
    "word_jaccard_similarity": "Word Jaccard Similarity",
    "word_count_delta": "Word Count Delta",
    "char_count_delta": "Character Count Delta",
}
COLORS = {
    "openai": "#1f77b4",
    "sarvam": "#ff7f0e",
    "openai_vs_sarvam": "#2ca02c",
}


def load_rows():
    with INPUT_CSV.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_values(rows, language, comparison_scope, metric):
    grouped = defaultdict(list)
    for row in rows:
        if row["language"] != language:
            continue
        if row["comparison_scope"] != comparison_scope:
            continue
        if row["metric"] != metric:
            continue
        grouped[row["model_or_pair"]].append(float(row["value"]))
    return grouped


def save_boxplot(language, comparison_scope, metric, grouped_values):
    labels = [
        group_name for group_name in GROUP_ORDER[comparison_scope]
        if grouped_values.get(group_name)
    ]
    data = [grouped_values[group_name] for group_name in labels]
    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(
        data,
        tick_labels=[DISPLAY_NAMES[label] for label in labels],
        patch_artist=True,
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.55)

    ax.set_title(f"{language.title()} | {comparison_scope.replace('_', ' ').title()} | {DISPLAY_NAMES[metric]}")
    ax.set_ylabel(DISPLAY_NAMES[metric])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"{language}_{comparison_scope}_{metric}_boxplot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_histogram(language, comparison_scope, metric, grouped_values):
    labels = [
        group_name for group_name in GROUP_ORDER[comparison_scope]
        if grouped_values.get(group_name)
    ]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in labels:
        ax.hist(
            grouped_values[label],
            bins=30,
            alpha=0.5,
            label=DISPLAY_NAMES[label],
            color=COLORS[label],
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_title(f"{language.title()} | {comparison_scope.replace('_', ' ').title()} | {DISPLAY_NAMES[metric]}")
    ax.set_xlabel(DISPLAY_NAMES[metric])
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"{language}_{comparison_scope}_{metric}_histogram.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    created_files = []
    for language in LANGUAGES:
        for comparison_scope in GROUP_ORDER:
            for metric in METRICS:
                grouped_values = group_values(rows, language, comparison_scope, metric)
                before = set(OUTPUT_DIR.glob("*.png"))
                save_boxplot(language, comparison_scope, metric, grouped_values)
                save_histogram(language, comparison_scope, metric, grouped_values)
                after = set(OUTPUT_DIR.glob("*.png"))
                created_files.extend(sorted(str(path) for path in (after - before)))

    summary_path = OUTPUT_DIR / "README.md"
    summary_lines = [
        "# Social Media Homogeneity Plots",
        "",
        "This folder contains boxplots and histograms generated from `boxplot_metrics.csv`.",
        "",
        "Generated plot groups:",
        "",
        "- English intra-model: OpenAI vs Sarvam",
        "- English inter-model: OpenAI vs Sarvam",
        "- Hindi intra-model: OpenAI vs Sarvam",
        "- Hindi inter-model: OpenAI vs Sarvam",
        "",
        "Metrics:",
        "",
        "- Cosine similarity",
        "- Word Jaccard similarity",
        "- Word count delta",
        "- Character count delta",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved plots to: {OUTPUT_DIR}")
    print(f"Created {len(list(OUTPUT_DIR.glob('*.png')))} PNG files")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
