import csv
import json
import re
import unicodedata
from itertools import combinations, product
from pathlib import Path
from statistics import mean, median


BASE_DIR = Path("output")
INPUT_ROOTS = {
    "openai": BASE_DIR / "openai" / "nearest",
    "sarvam": BASE_DIR / "sarvam" / "nearest",
}
LANGUAGES = ("english", "hindi")
RUNS = ("1", "2", "3")
OUTPUT_DIR = BASE_DIR / "analysis" / "nearest_text_consistency"
TEXT_FIELD = "nearest"

ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "being", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "them", "they", "this",
    "to", "was", "were", "which", "while", "with", "would", "reminds", "news",
    "article", "most", "strongly", "broader", "historical", "pattern", "past",
    "present", "event", "like", "than", "also", "there", "about", "after", "before",
}
HINDI_STOPWORDS = {
    "और", "एक", "यह", "इस", "उस", "उसके", "उन", "उनके", "कि", "के", "का", "की", "को",
    "में", "से", "पर", "भी", "था", "थे", "थी", "है", "हैं", "हो", "होता", "होती", "होते",
    "या", "तो", "जो", "तक", "लिए", "लिए", "गया", "गई", "गए", "बाद", "पहले", "समाचार",
    "लेख", "याद", "दिलाता", "दिलाती", "मामला", "घटना", "ऐतिहासिक", "पैटर्न", "वर्तमान",
    "पूर्व", "सबसे", "मजबूत", "तरह", "जैसा", "क्योंकि",
}
COUNTRY_WORDS = {
    "india", "indian", "pakistan", "pakistani", "china", "chinese", "america",
    "american", "united", "states", "uk", "britain", "british", "russia", "russian",
    "ukraine", "iran", "iraq", "israel", "palestine", "afghanistan", "nepal", "sri",
    "lanka", "bangladesh", "france", "germany", "japan", "australia", "canada",
    "भारत", "भारतीय", "पाकिस्तान", "चीन", "अमेरिका", "ब्रिटेन", "रूस", "यूक्रेन", "ईरान",
    "इराक", "इज़राइल", "फिलिस्तीन", "अफ़ग़ानिस्तान", "नेपाल", "श्रीलंका", "बांग्लादेश",
    "फ्रांस", "जर्मनी", "जापान", "ऑस्ट्रेलिया", "कनाडा",
}
CATEGORY_KEYWORDS = {
    "war_conflict": {
        "war", "conflict", "battle", "civil war", "invasion", "occupation", "military",
        "coup", "rebellion", "insurgency", "riot", "riots", "genocide", "uprising",
        "युद्ध", "संघर्ष", "लड़ाई", "हमला", "आक्रमण", "सैन्य", "तख्तापलट", "विद्रोह", "दंगा",
    },
    "election_politics": {
        "election", "parliament", "politics", "government", "coalition", "brexit",
        "referendum", "president", "prime minister", "democracy", "scandal", "watergate",
        "चुनाव", "राजनीति", "सरकार", "गठबंधन", "संसद", "प्रधानमंत्री", "राष्ट्रपति", "लोकतंत्र",
    },
    "disaster": {
        "earthquake", "flood", "tsunami", "hurricane", "cyclone", "wildfire", "famine",
        "disaster", "भोपाल", "भूकंप", "बाढ़", "सुनामी", "चक्रवात", "आग", "आपदा",
    },
    "terrorism": {
        "terror", "terrorism", "9/11", "attack", "bombing", "extremism", "assassination",
        "आतंक", "आतंकवाद", "हमला", "बम", "उग्रवाद", "हत्या",
    },
    "economic_crisis": {
        "recession", "depression", "financial crisis", "inflation", "economic crisis",
        "market crash", "economy", "मंदी", "आर्थिक", "वित्तीय", "महामंदी", "मुद्रास्फीति",
    },
    "protest_social_unrest": {
        "protest", "movement", "social unrest", "demonstration", "arab spring", "strike",
        "civil rights", "occupy", "प्रदर्शन", "आंदोलन", "विरोध", "अशांति", "हड़ताल", "विद्रोह",
    },
    "pandemic_health": {
        "pandemic", "epidemic", "covid", "coronavirus", "plague", "sars", "ebola",
        "महामारी", "कोविड", "कोरोना", "स्वास्थ्य", "वायरस", "प्लेग", "सार्स",
    },
    "institutional_breakdown": {
        "constitutional crisis", "institutional breakdown", "collapse", "governance",
        "system failure", "trust crisis", "polarization", "breakdown", "अस्थिरता",
        "संवैधानिक", "व्यवस्था", "विफलता", "ध्रुवीकरण", "संकट",
    },
}
EVENT_KEYWORDS = (
    "war", "crisis", "election", "scandal", "pandemic", "attack", "bombing", "riot",
    "protest", "movement", "revolution", "flood", "earthquake", "tsunami", "disaster",
    "genocide", "emergency", "referendum", "partition", "महामारी", "युद्ध", "संकट",
    "चुनाव", "हमला", "आंदोलन", "क्रांति", "बाढ़", "भूकंप", "दंगा", "आपदा", "आपातकाल",
    "विभाजन",
)


def safe_mean(values):
    return mean(values) if values else 0.0


def safe_median(values):
    return median(values) if values else 0.0


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def tokenize(text):
    text = text or ""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def get_stopwords(language):
    return ENGLISH_STOPWORDS if language == "english" else HINDI_STOPWORDS


def normalize_text(text):
    text = (text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_event_key(text, language):
    stopwords = get_stopwords(language)
    tokens = [tok for tok in tokenize(normalize_text(text)) if tok not in stopwords]
    if not tokens:
        return ""
    return " ".join(tokens[:18])


def build_ngrams(tokens, n):
    return list(zip(*(tokens[i:] for i in range(n)))) if len(tokens) >= n else []


def english_entity_phrases(text):
    phrases = set()
    pattern = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b")
    for match in pattern.finditer(text or ""):
        phrase = match.group(0).strip()
        if len(phrase) > 2:
            phrases.add(phrase)
    lower_text = (text or "").lower()
    for keyword in EVENT_KEYWORDS:
        if keyword in lower_text:
            phrases.add(keyword)
    return phrases


def hindi_entity_phrases(text):
    phrases = set()
    text = text or ""
    for phrase in re.findall(r"'([^']+)'|\"([^\"]+)\"", text):
        value = next((part for part in phrase if part), "").strip()
        if value:
            phrases.add(value)
    devanagari_chunks = re.findall(r"[\u0900-\u097F][\u0900-\u097F\s\-]{3,}", text)
    for chunk in devanagari_chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if any(keyword in chunk for keyword in EVENT_KEYWORDS):
            phrases.add(chunk)
    for keyword in EVENT_KEYWORDS:
        if keyword in text.lower():
            phrases.add(keyword)
    return phrases


def extract_entity_features(text, language):
    normalized = normalize_text(text)
    years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", normalized))
    number_tokens = set(re.findall(r"\b\d+\b", normalized))
    country_hits = {token for token in tokenize(normalized) if token in COUNTRY_WORDS}
    if language == "english":
        entity_phrases = english_entity_phrases(text)
    else:
        entity_phrases = hindi_entity_phrases(text)
    return {
        "entity_phrases": entity_phrases,
        "years": years,
        "numbers": number_tokens,
        "countries": country_hits,
    }


def classify_category(text):
    normalized = normalize_text(text)
    best_category = "other"
    best_score = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in normalized:
                score += 1
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def load_run(model, language, run):
    folder = INPUT_ROOTS[model] / f"{language}_{run}"
    items = {}
    for path in sorted(folder.glob("*.json"), key=lambda p: int(p.stem)):
        data = json.loads(path.read_text())
        text_value = data.get(TEXT_FIELD)
        items[data["id"]] = {
            "file_name": path.name,
            "text_value": text_value,
            "title": data.get("title", ""),
            "summary": data.get("summary", ""),
            "url": data.get("url", ""),
            "event_key": build_event_key(text_value, language),
            "category": classify_category(text_value or ""),
            "entity_features": extract_entity_features(text_value or "", language),
        }
    return items


def lexical_metrics(text):
    tokens = tokenize(text)
    unique_tokens = set(tokens)
    bigrams = build_ngrams(tokens, 2)
    trigrams = build_ngrams(tokens, 3)
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


def set_jaccard(left_set, right_set):
    union = left_set | right_set
    return safe_ratio(len(left_set & right_set), len(union)) if union else 1.0


def compare_records(left, right, language):
    left_text = left["text_value"] or ""
    right_text = right["text_value"] or ""
    left_tokens = tokenize(left_text)
    right_tokens = tokenize(right_text)
    left_token_set = set(left_tokens)
    right_token_set = set(right_tokens)
    left_bigrams = set(build_ngrams(left_tokens, 2))
    right_bigrams = set(build_ngrams(right_tokens, 2))
    left_trigrams = set(build_ngrams(left_tokens, 3))
    right_trigrams = set(build_ngrams(right_tokens, 3))
    left_features = left["entity_features"]
    right_features = right["entity_features"]

    return {
        "exact_text_match": int(left_text == right_text),
        "word_overlap_count": len(left_token_set & right_token_set),
        "word_jaccard_similarity": set_jaccard(left_token_set, right_token_set),
        "event_key_left": left["event_key"],
        "event_key_right": right["event_key"],
        "exact_event_key_match": int(left["event_key"] == right["event_key"] and left["event_key"] != ""),
        "bigram_jaccard_similarity": set_jaccard(left_bigrams, right_bigrams),
        "trigram_jaccard_similarity": set_jaccard(left_trigrams, right_trigrams),
        "entity_phrase_overlap_count": len(left_features["entity_phrases"] & right_features["entity_phrases"]),
        "entity_phrase_jaccard_similarity": set_jaccard(left_features["entity_phrases"], right_features["entity_phrases"]),
        "year_overlap_count": len(left_features["years"] & right_features["years"]),
        "year_jaccard_similarity": set_jaccard(left_features["years"], right_features["years"]),
        "country_overlap_count": len(left_features["countries"] & right_features["countries"]),
        "country_jaccard_similarity": set_jaccard(left_features["countries"], right_features["countries"]),
        "category_left": left["category"],
        "category_right": right["category"],
        "same_category": int(left["category"] == right["category"]),
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
    categories = {}

    for left_label, right_label in combinations(labels, 2):
        metrics = compare_records(records_by_label[left_label], records_by_label[right_label], "english")
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
    for label, record in records_by_label.items():
        categories[label] = record["category"]

    longest_word_label = max(all_word_counts, key=all_word_counts.get)
    shortest_word_label = min(all_word_counts, key=all_word_counts.get)
    longest_char_label = max(all_char_counts, key=all_char_counts.get)
    shortest_char_label = min(all_char_counts, key=all_char_counts.get)

    unique_categories = sorted(set(categories.values()))

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
        "category_count": len(unique_categories),
        "category_labels": " | ".join(unique_categories),
    }


def summarize_pair_rows(pair_rows):
    return {
        "pair_count": len(pair_rows),
        "exact_text_match_rate": safe_mean([row["exact_text_match"] for row in pair_rows]),
        "word_jaccard_mean": safe_mean([row["word_jaccard_similarity"] for row in pair_rows]),
        "word_jaccard_median": safe_median([row["word_jaccard_similarity"] for row in pair_rows]),
        "exact_event_key_match_rate": safe_mean([row["exact_event_key_match"] for row in pair_rows]),
        "bigram_jaccard_mean": safe_mean([row["bigram_jaccard_similarity"] for row in pair_rows]),
        "trigram_jaccard_mean": safe_mean([row["trigram_jaccard_similarity"] for row in pair_rows]),
        "entity_phrase_jaccard_mean": safe_mean([row["entity_phrase_jaccard_similarity"] for row in pair_rows]),
        "year_jaccard_mean": safe_mean([row["year_jaccard_similarity"] for row in pair_rows]),
        "country_jaccard_mean": safe_mean([row["country_jaccard_similarity"] for row in pair_rows]),
        "same_category_rate": safe_mean([row["same_category"] for row in pair_rows]),
        "word_count_delta_mean": safe_mean([row["word_count_delta"] for row in pair_rows]),
        "char_count_delta_mean": safe_mean([row["char_count_delta"] for row in pair_rows]),
    }


def summarize_variation_rows(rows):
    return {
        "article_count": len(rows),
        "mean_word_count_range": safe_mean([row["word_count_range"] for row in rows]),
        "mean_char_count_range": safe_mean([row["char_count_range"] for row in rows]),
        "mean_category_count": safe_mean([row["category_count"] for row in rows]),
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
    lexical_rows = []
    lexical_summary_rows = []
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
                        TEXT_FIELD: record["text_value"] or "",
                        "event_key": record["event_key"],
                        "category": record["category"],
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
            pair_rows = []
            variation_group = []
            common_ids = sorted(set.intersection(*(set(data[model][language][run]) for run in RUNS)))
            for article_id in common_ids:
                records = {
                    f"{model}_{run}": data[model][language][run][article_id]
                    for run in RUNS
                }
                labels = sorted(records)
                local_pair_rows = []
                for left_label, right_label in combinations(labels, 2):
                    row = compare_records(records[left_label], records[right_label], language)
                    row["left_label"] = left_label
                    row["right_label"] = right_label
                    detail_rows.append({
                        "comparison_scope": "intra_model",
                        "language": language,
                        "model_or_pair": model,
                        "article_id": article_id,
                        "title": records[f"{model}_1"]["title"],
                        "left_label": left_label,
                        "right_label": right_label,
                        f"left_{TEXT_FIELD}": records[left_label]["text_value"],
                        f"right_{TEXT_FIELD}": records[right_label]["text_value"],
                        **row,
                    })
                    local_pair_rows.append(row)
                    pair_rows.append(row)
                categories = sorted({record["category"] for record in records.values()})
                variation_entry = {
                    "comparison_scope": "intra_model",
                    "language": language,
                    "model_or_pair": model,
                    "article_id": article_id,
                    "title": records[f"{model}_1"]["title"],
                    "word_count_range": max(len(tokenize(r["text_value"])) for r in records.values()) - min(len(tokenize(r["text_value"])) for r in records.values()),
                    "char_count_range": max(len((r["text_value"] or "")) for r in records.values()) - min(len((r["text_value"] or "")) for r in records.values()),
                    "category_count": len(categories),
                    "category_labels": " | ".join(categories),
                    "exact_text_match_rate": safe_mean([r["exact_text_match"] for r in local_pair_rows]),
                    "exact_event_key_match_rate": safe_mean([r["exact_event_key_match"] for r in local_pair_rows]),
                    "same_category_rate": safe_mean([r["same_category"] for r in local_pair_rows]),
                    "word_jaccard_mean": safe_mean([r["word_jaccard_similarity"] for r in local_pair_rows]),
                    "bigram_jaccard_mean": safe_mean([r["bigram_jaccard_similarity"] for r in local_pair_rows]),
                    "trigram_jaccard_mean": safe_mean([r["trigram_jaccard_similarity"] for r in local_pair_rows]),
                    "entity_phrase_jaccard_mean": safe_mean([r["entity_phrase_jaccard_similarity"] for r in local_pair_rows]),
                    "year_jaccard_mean": safe_mean([r["year_jaccard_similarity"] for r in local_pair_rows]),
                }
                variation_rows.append(variation_entry)
                variation_group.append(variation_entry)

            summary_rows.append({
                "comparison_scope": "intra_model",
                "language": language,
                "model_or_pair": model,
                "article_count": len(common_ids),
                "run_pairs_considered": 3,
                **summarize_pair_rows(pair_rows),
                **summarize_variation_rows(variation_group),
            })

            repetitive_sorted = sorted(
                variation_group,
                key=lambda row: (
                    -row["exact_event_key_match_rate"],
                    -row["same_category_rate"],
                    -row["word_jaccard_mean"],
                    row["word_count_range"],
                ),
            )[:20]
            variable_sorted = sorted(
                variation_group,
                key=lambda row: (
                    row["exact_event_key_match_rate"],
                    row["same_category_rate"],
                    row["word_jaccard_mean"],
                    -row["category_count"],
                    -row["word_count_range"],
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

        model_pair = "openai_vs_sarvam"
        common_ids = sorted(
            set.intersection(
                *(set(data["openai"][language][run]) for run in RUNS),
                *(set(data["sarvam"][language][run]) for run in RUNS),
            )
        )
        inter_pair_rows = []
        inter_variation = []
        for article_id in common_ids:
            records = {
                **{f"openai_{run}": data["openai"][language][run][article_id] for run in RUNS},
                **{f"sarvam_{run}": data["sarvam"][language][run][article_id] for run in RUNS},
            }
            local_pair_rows = []
            for openai_run, sarvam_run in product(RUNS, RUNS):
                left_label = f"openai_{openai_run}"
                right_label = f"sarvam_{sarvam_run}"
                row = compare_records(records[left_label], records[right_label], language)
                row["left_label"] = left_label
                row["right_label"] = right_label
                detail_rows.append({
                    "comparison_scope": "inter_model",
                    "language": language,
                    "model_or_pair": model_pair,
                    "article_id": article_id,
                    "title": records["openai_1"]["title"],
                    "left_label": left_label,
                    "right_label": right_label,
                    f"left_{TEXT_FIELD}": records[left_label]["text_value"],
                    f"right_{TEXT_FIELD}": records[right_label]["text_value"],
                    **row,
                })
                local_pair_rows.append(row)
                inter_pair_rows.append(row)

            categories = sorted({record["category"] for record in records.values()})
            variation_entry = {
                "comparison_scope": "inter_model",
                "language": language,
                "model_or_pair": model_pair,
                "article_id": article_id,
                "title": records["openai_1"]["title"],
                "word_count_range": max(len(tokenize(r["text_value"])) for r in records.values()) - min(len(tokenize(r["text_value"])) for r in records.values()),
                "char_count_range": max(len((r["text_value"] or "")) for r in records.values()) - min(len((r["text_value"] or "")) for r in records.values()),
                "category_count": len(categories),
                "category_labels": " | ".join(categories),
                "exact_text_match_rate": safe_mean([r["exact_text_match"] for r in local_pair_rows]),
                "exact_event_key_match_rate": safe_mean([r["exact_event_key_match"] for r in local_pair_rows]),
                "same_category_rate": safe_mean([r["same_category"] for r in local_pair_rows]),
                "word_jaccard_mean": safe_mean([r["word_jaccard_similarity"] for r in local_pair_rows]),
                "bigram_jaccard_mean": safe_mean([r["bigram_jaccard_similarity"] for r in local_pair_rows]),
                "trigram_jaccard_mean": safe_mean([r["trigram_jaccard_similarity"] for r in local_pair_rows]),
                "entity_phrase_jaccard_mean": safe_mean([r["entity_phrase_jaccard_similarity"] for r in local_pair_rows]),
                "year_jaccard_mean": safe_mean([r["year_jaccard_similarity"] for r in local_pair_rows]),
            }
            variation_rows.append(variation_entry)
            inter_variation.append(variation_entry)

        summary_rows.append({
            "comparison_scope": "inter_model",
            "language": language,
            "model_or_pair": model_pair,
            "article_count": len(common_ids),
            "run_pairs_considered": 9,
            **summarize_pair_rows(inter_pair_rows),
            **summarize_variation_rows(inter_variation),
        })

        repetitive_sorted = sorted(
            inter_variation,
            key=lambda row: (
                -row["exact_event_key_match_rate"],
                -row["same_category_rate"],
                -row["word_jaccard_mean"],
                row["word_count_range"],
            ),
        )[:20]
        variable_sorted = sorted(
            inter_variation,
            key=lambda row: (
                row["exact_event_key_match_rate"],
                row["same_category_rate"],
                row["word_jaccard_mean"],
                -row["category_count"],
                -row["word_count_range"],
            ),
        )[:20]
        for rank, row in enumerate(repetitive_sorted, start=1):
            example_rows.append({
                "example_type": "most_repetitive",
                "comparison_scope": "inter_model",
                "language": language,
                "model_or_pair": model_pair,
                "rank": rank,
                **row,
            })
        for rank, row in enumerate(variable_sorted, start=1):
            example_rows.append({
                "example_type": "most_variable",
                "comparison_scope": "inter_model",
                "language": language,
                "model_or_pair": model_pair,
                "rank": rank,
                **row,
            })

    write_csv(
        OUTPUT_DIR / "summary.csv",
        summary_rows,
        [
            "comparison_scope", "language", "model_or_pair", "article_count",
            "run_pairs_considered", "pair_count", "exact_text_match_rate",
            "word_jaccard_mean", "word_jaccard_median", "exact_event_key_match_rate",
            "bigram_jaccard_mean", "trigram_jaccard_mean", "entity_phrase_jaccard_mean",
            "year_jaccard_mean", "country_jaccard_mean", "same_category_rate",
            "word_count_delta_mean", "char_count_delta_mean",
            "mean_word_count_range", "mean_char_count_range", "mean_category_count",
        ],
    )

    write_csv(
        OUTPUT_DIR / "pairwise_details.csv",
        detail_rows,
        [
            "comparison_scope", "language", "model_or_pair", "article_id", "title",
            "left_label", "right_label", f"left_{TEXT_FIELD}", f"right_{TEXT_FIELD}",
            "exact_text_match", "word_overlap_count", "word_jaccard_similarity",
            "event_key_left", "event_key_right", "exact_event_key_match",
            "bigram_jaccard_similarity", "trigram_jaccard_similarity",
            "entity_phrase_overlap_count", "entity_phrase_jaccard_similarity",
            "year_overlap_count", "year_jaccard_similarity",
            "country_overlap_count", "country_jaccard_similarity",
            "category_left", "category_right", "same_category",
            "left_word_count", "right_word_count", "left_char_count",
            "right_char_count", "word_count_delta", "char_count_delta",
        ],
    )

    write_csv(
        OUTPUT_DIR / "variation_summary.csv",
        variation_rows,
        [
            "comparison_scope", "language", "model_or_pair", "article_id", "title",
            "word_count_range", "char_count_range", "category_count", "category_labels",
            "exact_text_match_rate", "exact_event_key_match_rate", "same_category_rate",
            "word_jaccard_mean", "bigram_jaccard_mean", "trigram_jaccard_mean",
            "entity_phrase_jaccard_mean", "year_jaccard_mean",
        ],
    )

    write_csv(
        OUTPUT_DIR / "lexical_diversity.csv",
        lexical_rows,
        [
            "language", "model", "run", "article_id", "title", TEXT_FIELD, "event_key",
            "category", "token_count", "unique_token_count", "type_token_ratio",
            "repeated_token_count", "bigram_count", "unique_bigram_count",
            "bigram_diversity", "trigram_count", "unique_trigram_count",
            "trigram_diversity", "char_count",
        ],
    )

    write_csv(
        OUTPUT_DIR / "lexical_summary.csv",
        lexical_summary_rows,
        [
            "language", "model", "text_count", "token_count_mean",
            "unique_token_count_mean", "type_token_ratio_mean",
            "repeated_token_count_mean", "bigram_diversity_mean",
            "trigram_diversity_mean", "char_count_mean",
        ],
    )

    write_csv(
        OUTPUT_DIR / "top_examples.csv",
        example_rows,
        [
            "example_type", "comparison_scope", "language", "model_or_pair", "rank",
            "article_id", "title", "word_count_range", "char_count_range",
            "category_count", "category_labels", "exact_text_match_rate",
            "exact_event_key_match_rate", "same_category_rate", "word_jaccard_mean",
            "bigram_jaccard_mean", "trigram_jaccard_mean",
            "entity_phrase_jaccard_mean", "year_jaccard_mean",
        ],
    )

    report_lines = [
        "# Nearest Event Consistency Report",
        "",
        "This report compares raw nearest-event texts without using embeddings.",
        "",
        "## What is measured",
        "",
        "- Exact text match across runs.",
        "- Token overlap via Jaccard similarity.",
        "- Normalized event-key match after lowercasing, punctuation removal, and stopword removal.",
        "- Bigram and trigram overlap.",
        "- Heuristic entity-style overlap: event phrases, years, and country mentions.",
        "- Category consistency: whether outputs stay in the same coarse analogy category.",
        "- Length variation: word-count and character-count range.",
        "",
        "## Summary Snapshot",
        "",
    ]
    for row in summary_rows:
        report_lines.append(
            f"- {row['comparison_scope']} | {row['language']} | {row['model_or_pair']}: "
            f"exact_event_key_match_rate={row['exact_event_key_match_rate']:.4f}, "
            f"word_jaccard_mean={row['word_jaccard_mean']:.4f}, "
            f"same_category_rate={row['same_category_rate']:.4f}"
        )
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote outputs to: {OUTPUT_DIR}")
    print("Created files:")
    print(f"  - {OUTPUT_DIR / 'summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'pairwise_details.csv'}")
    print(f"  - {OUTPUT_DIR / 'variation_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'lexical_diversity.csv'}")
    print(f"  - {OUTPUT_DIR / 'lexical_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'top_examples.csv'}")
    print(f"  - {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
