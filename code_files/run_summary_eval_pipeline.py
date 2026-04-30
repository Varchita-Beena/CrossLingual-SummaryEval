import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from openai import OpenAI


ROOT = Path("output")
GROUPS = [
    {"source_model": "openai", "language": "english", "run": "1"},
    {"source_model": "openai", "language": "hindi", "run": "1"},
    {"source_model": "sarvam", "language": "english", "run": "1"},
    {"source_model": "sarvam", "language": "hindi", "run": "1"},
]
OUTPUT_ROOT = Path("output/evaluation/summary_qa_pipeline")
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_MODE = "all"
DEFAULT_QUESTIONS_PER_SAMPLE = 5
DEFAULT_SAMPLE_LIMIT = 75
LABEL_SCORES = {
    "Correct": 1.0,
    "Partial": 0.5,
    "Incorrect": 0.0,
    "Not mentioned": 0.0,
}
QUESTION_TASKS = [
    {
        "name": "factual_consistency",
        "source_text_field": "generated_summary",
        "answer_text_field": "text",
        "include_importance": False,
        "metric_role": "factual_consistency",
    },
    {
        "name": "summary_quality",
        "source_text_field": "summary",
        "answer_text_field": "generated_summary",
        "include_importance": False,
        "metric_role": "summary_quality",
    },
    {
        "name": "source_coverage",
        "source_text_field": "text",
        "answer_text_field": "generated_summary",
        "include_importance": True,
        "importance_context_field": "summary",
        "metric_role": "source_coverage_weighted",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a custom summary evaluation pipeline with question generation, answering, judging, and analysis."
    )
    parser.add_argument(
        "--mode",
        choices=("run", "analyze", "all"),
        default=DEFAULT_MODE,
        help="Run API generation, analysis, or both.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. Required for mode=run or mode=all unless OPENAI_API_KEY is set.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for saved pipeline outputs.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help="Maximum number of valid samples per group.",
    )
    parser.add_argument(
        "--questions-per-sample",
        type=int,
        default=DEFAULT_QUESTIONS_PER_SAMPLE,
        help="How many questions to request in each question-generation call.",
    )
    parser.add_argument(
        "--question-runs",
        type=int,
        default=1,
        help="Number of question-generation runs per sample and task. Default is 1.",
    )
    parser.add_argument(
        "--answer-runs",
        type=int,
        default=1,
        help="Number of answer-generation runs per sample and task. Default is 1.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1200,
        help="Maximum completion tokens per request.",
    )
    parser.add_argument(
        "--temperature-question",
        type=float,
        default=0.7,
        help="Temperature for question generation.",
    )
    parser.add_argument(
        "--temperature-answer",
        type=float,
        default=0.2,
        help="Temperature for answer generation.",
    )
    parser.add_argument(
        "--temperature-judge",
        type=float,
        default=0.0,
        help="Temperature for judging.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_json_block(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    object_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if object_match:
        return json.loads(object_match.group(1))

    array_match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if array_match:
        return json.loads(array_match.group(1))

    raise ValueError("Could not parse JSON from model response.")


def group_input_folder(source_model, language, run):
    return ROOT / source_model / "summary" / f"{language}_{run}"


def group_output_folder(output_root, source_model, language, run):
    return output_root / source_model / f"{language}_{run}"


def get_generated_summary_value(source_model, language, data):
    generated_summary = (data.get("generated_summary") or "").strip()
    if generated_summary:
        return generated_summary

    # Data fix: for Sarvam English summary files, generated summaries were stored under `nearest`.
    if source_model == "sarvam" and language == "english":
        fallback = (data.get("nearest") or "").strip()
        if fallback:
            return fallback

    return ""


def select_valid_files(folder_path, sample_limit):
    selected = []
    missing_generated = 0
    source_model = folder_path.parts[-3]
    language = folder_path.name.split("_", 1)[0]
    for path in sorted(folder_path.glob("*.json"), key=lambda p: int(p.stem)):
        try:
            data = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        generated_summary = get_generated_summary_value(source_model, language, data)
        if not generated_summary:
            missing_generated += 1
            continue
        selected.append(path)
        if len(selected) >= sample_limit:
            break
    return selected, missing_generated


def ensure_article_record(article_output_path, source_data):
    if article_output_path.exists():
        return load_json(article_output_path)
    source_model = article_output_path.parts[-4]
    language = article_output_path.parts[-3].split("_", 1)[0]
    record = {
        "id": source_data["id"],
        "url": source_data.get("url", ""),
        "title": source_data.get("title", ""),
        "text": source_data.get("text", ""),
        "summary": source_data.get("summary", ""),
        "generated_summary": get_generated_summary_value(source_model, language, source_data),
        "pipeline_results": {},
    }
    save_json(article_output_path, record)
    return record


def build_question_messages(task, source_text, language, questions_per_sample, reference_summary=""):
    language_name = "Hindi" if language == "hindi" else "English"
    if task["include_importance"]:
        extra = (
            "You are generating factual evaluation questions for summary evaluation.\n"
            "Given the SOURCE TEXT below, generate closed-ended questions that can be answered using only the information explicitly stated in the source.\n\n"
            "Rules:\n"
            "1. Generate questions only from facts explicitly present in the source.\n"
            "2. Do not use outside knowledge.\n"
            "3. Do not create questions about facts that are implied but not clearly stated.\n"
            "4. Each question must have a short, verifiable answer.\n"
            "5. Include a mix of question types: Who, What, When, Where, Which, Yes/No, Name one.\n"
            "6. Avoid vague questions such as 'What happened?' or 'Why is this important?'\n"
            "7. Avoid multi-hop questions unless both pieces of evidence are clearly stated.\n"
            "8. Avoid questions that require long explanations.\n"
            "9. For yes/no questions, include both yes-answer and no-answer questions where possible.\n"
            "10. For every question, provide: question_type, question, expected_answer, evidence_sentence, importance.\n"
            "11. Importance must always be a number between 0 and 1.\n"
            "12. If the question is answerable from the REFERENCE SUMMARY as well, assign higher importance.\n"
            f"13. Generate up to {questions_per_sample} questions.\n"
            "Return JSON with this shape:\n"
            '{"questions":[{"question_type":"Who","question":"...","expected_answer":"...","evidence_sentence":"...","importance":0.83}]}\n'
            "Return only the JSON object.\n\n"
            f"Response language is {language_name}.\n\n"
            f"REFERENCE SUMMARY:\n{reference_summary.strip()}\n\n"
            f"SOURCE TEXT:\n{source_text.strip()}"
        )
    else:
        extra = (
            "You are generating factual evaluation questions for summary evaluation.\n"
            "Given the TEXT below, generate closed-ended questions that can be answered using only the information explicitly stated in the text.\n\n"
            "Rules:\n"
            "1. Generate questions only from facts explicitly present in the source.\n"
            "2. Do not use outside knowledge.\n"
            "3. Do not create questions about facts that are implied but not clearly stated.\n"
            "4. Each question must have a short, verifiable answer.\n"
            "5. Include a mix of question types: Who, What, When, Where, Which, Yes/No, Name one.\n"
            "6. Avoid vague questions.\n"
            "7. Avoid multi-hop questions unless both pieces of evidence are clearly stated.\n"
            "8. Avoid questions that require long explanations.\n"
            "9. For every question, provide: question_type, question, expected_answer, evidence_sentence.\n"
            f"10. Generate up to {questions_per_sample} questions.\n"
            "Return JSON with this shape:\n"
            '{"questions":[{"question_type":"Who","question":"...","expected_answer":"...","evidence_sentence":"..."}]}\n'
            "Return only the JSON object.\n\n"
            f"Response language is {language_name}.\n\n"
            f"TEXT:\n{source_text.strip()}"
        )

    return [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": extra},
    ]


def build_answer_messages(text, questions, language):
    language_name = "Hindi" if language == "hindi" else "English"
    question_lines = []
    for idx, item in enumerate(questions, start=1):
        question_lines.append(f"{idx}. {item['question']}")
    user_message = (
        "You are answering factual questions using only the given TEXT.\n\n"
        "Rules:\n"
        "1. Answer only using information explicitly present in the TEXT.\n"
        "2. Do not use outside knowledge.\n"
        '3. If the answer is not present, write "Not mentioned".\n'
        "4. Keep each answer short.\n"
        "Return JSON with this shape:\n"
        '{"answers":[{"question_index":1,"answer":"..."},{"question_index":2,"answer":"..."}]}\n'
        "Return only the JSON object.\n\n"
        f"Response language is {language_name}.\n\n"
        f"TEXT:\n{text.strip()}\n\n"
        "QUESTIONS:\n"
        + "\n".join(question_lines)
    )
    return [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": user_message},
    ]


def build_judge_messages(questions, comparison_answers):
    payload = []
    for idx, question in enumerate(questions, start=1):
        payload.append(
            {
                "question_index": idx,
                "question": question["question"],
                "source_answer": question["expected_answer"],
                "summary_answer": comparison_answers[idx - 1],
            }
        )
    user_message = (
        "You are evaluating whether a summary answer is factually consistent with the source answer.\n\n"
        "Rules:\n"
        "1. Mark Correct if the summary answer has the same meaning as the source answer.\n"
        "2. Mark Correct if it is a valid paraphrase or less specific but still accurate.\n"
        "3. Mark Incorrect if it contradicts the source answer.\n"
        "4. Mark Partial if the summary answer contains some correct information but misses important details.\n"
        '5. If the summary answer is "Not mentioned" and the source answer contains the answer, return "Not mentioned".\n'
        "6. Do not use outside knowledge.\n"
        "Return JSON with this shape:\n"
        '{"results":[{"question_index":1,"label":"Correct"},{"question_index":2,"label":"Partial"}]}\n'
        "Allowed labels: Correct, Incorrect, Partial, Not mentioned.\n"
        "Return only the JSON object.\n\n"
        f"ITEMS:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": user_message},
    ]


def call_json(client, model_name, messages, temperature, max_completion_tokens):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content or ""
    parsed = extract_json_block(raw_text)
    usage = response.usage.model_dump() if response.usage else None
    return raw_text, parsed, usage


def normalize_questions(parsed, include_importance):
    questions = []
    for item in parsed.get("questions", []):
        question = (item.get("question") or "").strip()
        expected_answer = (item.get("expected_answer") or "").strip()
        evidence_sentence = (item.get("evidence_sentence") or "").strip()
        question_type = (item.get("question_type") or "").strip() or "Unknown"
        if not question or not expected_answer:
            continue
        normalized = {
            "question_type": question_type,
            "question": question,
            "expected_answer": expected_answer,
            "evidence_sentence": evidence_sentence,
        }
        if include_importance:
            try:
                importance = float(item.get("importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
            normalized["importance"] = max(0.0, min(1.0, importance))
        questions.append(normalized)
    return questions


def normalize_answers(parsed, expected_count):
    answers_map = {}
    for item in parsed.get("answers", []):
        try:
            idx = int(item.get("question_index"))
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= expected_count:
            answers_map[idx] = (item.get("answer") or "").strip() or "Not mentioned"
    return [answers_map.get(idx, "Not mentioned") for idx in range(1, expected_count + 1)]


def normalize_labels(parsed, expected_count):
    labels_map = {}
    for item in parsed.get("results", []):
        try:
            idx = int(item.get("question_index"))
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= expected_count:
            label = (item.get("label") or "").strip()
            if label not in LABEL_SCORES:
                label = "Incorrect"
            labels_map[idx] = label
    return [labels_map.get(idx, "Incorrect") for idx in range(1, expected_count + 1)]


def run_pipeline(args):
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    print(f"[run] mode=run model={args.model_name} output_root={args.output_root}")
    manifest = {
        "model_name": args.model_name,
        "sample_limit": args.sample_limit,
        "questions_per_sample": args.questions_per_sample,
        "question_runs": args.question_runs,
        "answer_runs": args.answer_runs,
        "groups": [],
    }

    for group in GROUPS:
        source_model = group["source_model"]
        language = group["language"]
        run = group["run"]
        input_folder = group_input_folder(source_model, language, run)
        output_folder = group_output_folder(args.output_root, source_model, language, run)
        articles_folder = output_folder / "articles"
        input_folder_exists = input_folder.exists()
        group_name = f"{source_model}/{language}_{run}"

        print(f"[group] start {group_name}")
        print(f"[group] input={input_folder}")
        print(f"[group] output={output_folder}")

        group_entry = {
            "source_model": source_model,
            "language": language,
            "run": run,
            "input_folder": str(input_folder),
            "output_folder": str(output_folder),
            "input_folder_exists": input_folder_exists,
            "status": "pending",
        }

        if not input_folder_exists:
            print(f"[group] skip {group_name}: missing input folder")
            group_entry["status"] = "missing_input_folder"
            manifest["groups"].append(group_entry)
            continue

        selected_files, missing_generated = select_valid_files(input_folder, args.sample_limit)
        group_entry["selected_file_count"] = len(selected_files)
        group_entry["missing_generated_summary_count_observed"] = missing_generated
        print(
            f"[group] valid_samples={len(selected_files)} "
            f"missing_generated_summary_seen={missing_generated}"
        )

        if not selected_files:
            print(f"[group] skip {group_name}: no valid generated summaries")
            group_entry["status"] = "no_valid_generated_summary_samples"
            manifest["groups"].append(group_entry)
            continue

        group_entry["status"] = "processed"
        processed_articles = 0

        for article_index, file_path in enumerate(selected_files, start=1):
            source_data = load_json(file_path)
            article_output_path = articles_folder / file_path.name
            article_record = ensure_article_record(article_output_path, source_data)
            pipeline_results = article_record.setdefault("pipeline_results", {})
            print(
                f"[article] {group_name} {article_index}/{len(selected_files)} "
                f"file={file_path.name} id={source_data.get('id','')}"
            )
            print(f"[article] store={article_output_path}")

            for task in QUESTION_TASKS:
                task_name = task["name"]
                task_store = pipeline_results.setdefault(
                    task_name,
                    {"question_runs": [], "answer_runs": [], "judge_runs": []},
                )
                source_text = (article_record.get(task["source_text_field"]) or "").strip()
                reference_summary = (article_record.get(task.get("importance_context_field", "")) or "").strip()
                if not source_text:
                    print(f"[task] skip task={task_name}: empty source field={task['source_text_field']}")
                    continue
                print(
                    f"[task] start task={task_name} "
                    f"question_runs_done={len(task_store['question_runs'])} "
                    f"answer_runs_done={len(task_store['answer_runs'])} "
                    f"judge_runs_done={len(task_store['judge_runs'])}"
                )

                existing_question_runs = {item["question_run"] for item in task_store["question_runs"]}
                for question_run in range(1, args.question_runs + 1):
                    if question_run in existing_question_runs:
                        print(f"[task] task={task_name} question_run={question_run} already exists")
                        continue
                    print(f"[task] task={task_name} generating questions run={question_run}")
                    raw_output, parsed, usage = call_json(
                        client=client,
                        model_name=args.model_name,
                        messages=build_question_messages(
                            task=task,
                            source_text=source_text,
                            language=language,
                            questions_per_sample=args.questions_per_sample,
                            reference_summary=reference_summary,
                        ),
                        temperature=args.temperature_question,
                        max_completion_tokens=args.max_completion_tokens,
                    )
                    questions = normalize_questions(parsed, include_importance=task["include_importance"])
                    task_store["question_runs"].append(
                        {
                            "question_run": question_run,
                            "raw_output": raw_output,
                            "questions": questions,
                            "usage": usage,
                        }
                    )
                    save_json(article_output_path, article_record)
                    print(
                        f"[task] task={task_name} question_run={question_run} "
                        f"saved_questions={len(questions)}"
                    )

                question_runs_sorted = sorted(task_store["question_runs"], key=lambda item: item["question_run"])
                for question_run_entry in question_runs_sorted:
                    question_run = question_run_entry["question_run"]
                    questions = question_run_entry["questions"]
                    if not questions:
                        print(f"[task] task={task_name} question_run={question_run} has no questions")
                        continue

                    existing_answer_keys = {
                        (item["question_run"], item["answer_run"])
                        for item in task_store["answer_runs"]
                    }
                    for answer_run in range(1, args.answer_runs + 1):
                        key = (question_run, answer_run)
                        if key in existing_answer_keys:
                            print(
                                f"[task] task={task_name} question_run={question_run} "
                                f"answer_run={answer_run} already exists"
                            )
                            continue
                        answer_text = (article_record.get(task["answer_text_field"]) or "").strip()
                        print(
                            f"[task] task={task_name} question_run={question_run} "
                            f"answer_run={answer_run} answer_text_field={task['answer_text_field']}"
                        )
                        raw_output, parsed, usage = call_json(
                            client=client,
                            model_name=args.model_name,
                            messages=build_answer_messages(
                                text=answer_text,
                                questions=questions,
                                language=language,
                            ),
                            temperature=args.temperature_answer,
                            max_completion_tokens=args.max_completion_tokens,
                        )
                        answers = normalize_answers(parsed, expected_count=len(questions))
                        task_store["answer_runs"].append(
                            {
                                "question_run": question_run,
                                "answer_run": answer_run,
                                "raw_output": raw_output,
                                "answers": answers,
                                "usage": usage,
                            }
                        )
                        save_json(article_output_path, article_record)
                        print(
                            f"[task] task={task_name} question_run={question_run} "
                            f"answer_run={answer_run} saved_answers={len(answers)}"
                        )

                    existing_judge_keys = {
                        (item["question_run"], item["answer_run"])
                        for item in task_store["judge_runs"]
                    }
                    for answer_run in range(1, args.answer_runs + 1):
                        if (question_run, answer_run) in existing_judge_keys:
                            print(
                                f"[task] task={task_name} question_run={question_run} "
                                f"judge_for_answer_run={answer_run} already exists"
                            )
                            continue

                        comparison_answers = None
                        for answer_entry in task_store["answer_runs"]:
                            if answer_entry["question_run"] == question_run and answer_entry["answer_run"] == answer_run:
                                comparison_answers = answer_entry["answers"]

                        if comparison_answers is None:
                            print(
                                f"[task] task={task_name} question_run={question_run} "
                                f"judge_for_answer_run={answer_run} waiting_for_answer"
                            )
                            continue

                        print(
                            f"[task] task={task_name} question_run={question_run} "
                            f"judging answer_run={answer_run}"
                        )
                        raw_output, parsed, usage = call_json(
                            client=client,
                            model_name=args.model_name,
                            messages=build_judge_messages(
                                questions=questions,
                                comparison_answers=comparison_answers,
                            ),
                            temperature=args.temperature_judge,
                            max_completion_tokens=args.max_completion_tokens,
                        )
                        labels = normalize_labels(parsed, expected_count=len(questions))
                        task_store["judge_runs"].append(
                            {
                                "question_run": question_run,
                                "answer_run": answer_run,
                                "raw_output": raw_output,
                                "labels": labels,
                                "usage": usage,
                            }
                        )
                        save_json(article_output_path, article_record)
                        print(
                            f"[task] task={task_name} question_run={question_run} "
                            f"judge_for_answer_run={answer_run} saved_labels={len(labels)}"
                        )

            processed_articles += 1

        group_entry["processed_articles"] = processed_articles
        print(f"[group] done {group_name} processed_articles={processed_articles}")
        manifest["groups"].append(group_entry)

    save_json(args.output_root / "manifest.json", manifest)
    print(f"[run] manifest saved to {args.output_root / 'manifest.json'}")


def analyze_pipeline(output_root):
    manifest_path = output_root / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {"groups": []}
    print(f"[analyze] output_root={output_root}")
    overall_rows = []
    label_rows = []
    question_type_rows = []
    skipped_rows = []

    for group in manifest.get("groups", []):
        source_model = group["source_model"]
        language = group["language"]
        run = group["run"]
        group_folder = group_output_folder(output_root, source_model, language, run)
        articles_folder = group_folder / "articles"

        if group.get("status") != "processed":
            print(
                f"[analyze] skip group {source_model}/{language}_{run}: "
                f"status={group.get('status')}"
            )
            skipped_rows.append(
                {
                    "source_model": source_model,
                    "language": language,
                    "run": run,
                    "status": group.get("status"),
                    "selected_file_count": group.get("selected_file_count", 0),
                }
            )
            continue

        article_paths = sorted(articles_folder.glob("*.json"), key=lambda p: int(p.stem))
        print(
            f"[analyze] group {source_model}/{language}_{run} "
            f"articles={len(article_paths)} folder={articles_folder}"
        )
        per_task_scores = defaultdict(list)
        per_task_weights = defaultdict(list)
        per_task_label_counts = defaultdict(Counter)
        per_task_question_type_counts = defaultdict(Counter)
        per_task_question_type_label_counts = defaultdict(lambda: defaultdict(Counter))

        for article_path in article_paths:
            article = load_json(article_path)
            pipeline_results = article.get("pipeline_results", {})
            for task in QUESTION_TASKS:
                task_name = task["name"]
                task_store = pipeline_results.get(task_name, {})
                question_run_map = {
                    item["question_run"]: item
                    for item in task_store.get("question_runs", [])
                }
                judge_run_map = {
                    (item["question_run"], item["answer_run"]): item
                    for item in task_store.get("judge_runs", [])
                }

                for question_run, question_entry in question_run_map.items():
                    questions = question_entry.get("questions", [])
                    if not questions:
                        continue
                    for answer_run in range(1, 100):
                        judge_entry = judge_run_map.get((question_run, answer_run))
                        if judge_entry is None:
                            continue
                        labels = judge_entry.get("labels", [])
                        if len(labels) != len(questions):
                            continue
                        for idx, (question, label) in enumerate(zip(questions, labels), start=1):
                            weight = 1.0
                            if task["include_importance"]:
                                try:
                                    weight = float(question.get("importance", 0.5))
                                except (TypeError, ValueError):
                                    weight = 0.5
                            score = LABEL_SCORES.get(label, 0.0)
                            per_task_scores[task_name].append(score * weight if task["include_importance"] else score)
                            per_task_weights[task_name].append(weight if task["include_importance"] else 1.0)
                            per_task_label_counts[task_name][label] += 1
                            question_type = question.get("question_type", "Unknown")
                            per_task_question_type_counts[task_name][question_type] += 1
                            per_task_question_type_label_counts[task_name][question_type][label] += 1

        summary = {
            "source_model": source_model,
            "language": language,
            "run": run,
            "article_count": len(article_paths),
        }

        for task in QUESTION_TASKS:
            task_name = task["name"]
            raw_scores = per_task_scores[task_name]
            weights = per_task_weights[task_name]
            weighted_score = (sum(raw_scores) / sum(weights)) if weights and sum(weights) else 0.0
            prefix = task_name
            summary[f"{prefix}_score"] = weighted_score
            summary[f"{prefix}_count"] = len(weights)
            for label in LABEL_SCORES:
                summary[f"{prefix}_{label.lower().replace(' ', '_')}_count"] = per_task_label_counts[task_name][label]

        summary["factual_consistency"] = summary["factual_consistency_score"]
        summary["summary_quality"] = summary["summary_quality_score"]
        summary["source_coverage_weighted"] = summary["source_coverage_score"]
        summary["overall_mean"] = mean(
            [summary["factual_consistency"], summary["summary_quality"], summary["source_coverage_weighted"]]
        )
        overall_rows.append(summary)

        for task in QUESTION_TASKS:
            task_name = task["name"]
            for label, count in per_task_label_counts[task_name].items():
                label_rows.append(
                    {
                        "source_model": source_model,
                        "language": language,
                        "run": run,
                        "task": task_name,
                        "label": label,
                        "count": count,
                    }
                )
            for question_type, count in per_task_question_type_counts[task_name].items():
                counts_by_label = per_task_question_type_label_counts[task_name][question_type]
                question_type_rows.append(
                    {
                        "source_model": source_model,
                        "language": language,
                        "run": run,
                        "task": task_name,
                        "question_type": question_type,
                        "question_count": count,
                        "correct_count": counts_by_label["Correct"],
                        "partial_count": counts_by_label["Partial"],
                        "incorrect_count": counts_by_label["Incorrect"],
                        "not_mentioned_count": counts_by_label["Not mentioned"],
                        "error_count": counts_by_label["Incorrect"] + counts_by_label["Partial"] + counts_by_label["Not mentioned"],
                    }
                )

    analysis_folder = output_root / "analysis"
    write_csv(
        analysis_folder / "group_summary.csv",
        overall_rows,
        sorted(overall_rows[0].keys()) if overall_rows else ["source_model", "language", "run"],
    )
    write_csv(
        analysis_folder / "label_distribution.csv",
        label_rows,
        ["source_model", "language", "run", "task", "label", "count"],
    )
    write_csv(
        analysis_folder / "question_type_distribution.csv",
        question_type_rows,
        [
            "source_model", "language", "run", "task", "question_type", "question_count",
            "correct_count", "partial_count", "incorrect_count", "not_mentioned_count", "error_count",
        ],
    )
    write_csv(
        analysis_folder / "skipped_groups.csv",
        skipped_rows,
        ["source_model", "language", "run", "status", "selected_file_count"],
    )

    report_lines = [
        "# Summary Evaluation Pipeline Report",
        "",
        "## Group Summary",
        "",
    ]
    for row in overall_rows:
        report_lines.append(
            f"- {row['source_model']} | {row['language']} | {row['run']}: "
            f"factual_consistency={row['factual_consistency']:.4f}, "
            f"summary_quality={row['summary_quality']:.4f}, "
            f"source_coverage_weighted={row['source_coverage_weighted']:.4f}, "
            f"overall_mean={row['overall_mean']:.4f}"
        )
    if skipped_rows:
        report_lines.extend(["", "## Skipped Groups", ""])
        for row in skipped_rows:
            report_lines.append(
                f"- {row['source_model']} | {row['language']} | {row['run']}: status={row['status']}"
            )
    (analysis_folder / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[analyze] summary csv: {analysis_folder / 'group_summary.csv'}")
    print(f"[analyze] label distribution csv: {analysis_folder / 'label_distribution.csv'}")
    print(f"[analyze] question type distribution csv: {analysis_folder / 'question_type_distribution.csv'}")
    print(f"[analyze] skipped groups csv: {analysis_folder / 'skipped_groups.csv'}")
    print(f"[analyze] report: {analysis_folder / 'report.md'}")


def main():
    args = parse_args()
    if args.mode in {"run", "all"}:
        run_pipeline(args)
    if args.mode in {"analyze", "all"}:
        analyze_pipeline(args.output_root)


if __name__ == "__main__":
    main()
