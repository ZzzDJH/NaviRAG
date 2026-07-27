from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SUPPORTED_DATASETS = ("narrative", "loogle", "lbv2")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and report malformed records with line numbers."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected a JSON object at {path}:{line_number}, "
                    f"got {type(record).__name__}"
                )

            records.append(record)

    return records


def normalize_text(text: str) -> str:
    """Normalize text using the same rules as the original evaluation code."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def compute_f1(gold: str, prediction: str) -> float:
    gold_tokens = normalize_text(gold).split()
    prediction_tokens = normalize_text(prediction).split()

    if not gold_tokens or not prediction_tokens:
        return 0.0

    common = Counter(gold_tokens) & Counter(prediction_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact(gold: str, prediction: str) -> float:
    return float(normalize_text(gold) == normalize_text(prediction))


def compute_context_recall(context: str, gold: str) -> float:
    """Compute token recall of one reference answer in retrieved context."""
    context_tokens = normalize_text(context).split()
    gold_tokens = normalize_text(gold).split()

    if not context_tokens or not gold_tokens:
        return 0.0

    common = Counter(gold_tokens) & Counter(context_tokens)
    return sum(common.values()) / len(gold_tokens)


def require_string(
    record: dict[str, Any],
    field: str,
    *,
    file_label: str,
    index: int,
    allow_missing: bool = False,
) -> str:
    """Read a string field and provide an actionable validation error."""
    if field not in record:
        if allow_missing:
            return ""
        raise KeyError(f"Missing '{field}' in {file_label} record {index}")

    value = record[field]
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(
            f"'{field}' in {file_label} record {index} must be a string, "
            f"got {type(value).__name__}"
        )
    return value


def validate_alignment(
    prediction_records: list[dict[str, Any]],
    qa_records: list[dict[str, Any]],
) -> None:
    if len(prediction_records) != len(qa_records):
        raise ValueError(
            "Prediction and QA files must contain the same number of records: "
            f"{len(prediction_records)} != {len(qa_records)}"
        )
    if not qa_records:
        raise ValueError("The input files contain no evaluation records")


def evaluate_narrative(
    prediction_records: list[dict[str, Any]],
    qa_records: list[dict[str, Any]],
) -> dict[str, float | int]:
    total_f1 = 0.0
    total_recall = 0.0
    total_exact = 0.0

    for index, (prediction_record, qa_record) in enumerate(
        zip(prediction_records, qa_records),
        start=1,
    ):
        prediction = require_string(
            prediction_record,
            "prediction",
            file_label="prediction",
            index=index,
        )
        context = require_string(
            prediction_record,
            "context",
            file_label="prediction",
            index=index,
        )
        require_string(
            qa_record,
            "query",
            file_label="QA",
            index=index,
        )

        if "answer" not in qa_record:
            raise KeyError(f"Missing 'answer' in QA record {index}")
        answers = qa_record["answer"]
        if not isinstance(answers, list) or not answers:
            raise TypeError(
                f"'answer' in narrative QA record {index} must be a non-empty list[str]"
            )
        if not all(isinstance(answer, str) and answer.strip() for answer in answers):
            raise TypeError(
                f"Every narrative answer in QA record {index} must be a non-empty string"
            )

        # Empty predictions naturally receive F1=0 and EM=0. Retrieval recall is
        # independent of answer generation and is still measured from context.
        total_f1 += max(compute_f1(answer, prediction) for answer in answers)
        total_exact += max(compute_exact(answer, prediction) for answer in answers)
        total_recall += max(
            compute_context_recall(context, answer) for answer in answers
        )

    total = len(qa_records)
    return {
        "total_samples": total,
        "F1": total_f1 / total,
        "Recall": total_recall / total,
        "ExactMatch": total_exact / total,
    }


def build_loogle_judge_prompt(
    question: str,
    reference: str,
    prediction: str,
) -> str:
    return (
        "Given a question, a reference answer, and a predicted answer, decide "
        "whether the predicted answer is semantically equivalent to the reference "
        "answer. Only output True or False.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Predicted answer: {prediction}\n"
    )


def parse_boolean_judgment(output: str) -> bool | None:
    """Parse a strict True/False judgment, allowing surrounding whitespace."""
    normalized = output.strip().lower()
    if normalized.startswith("true"):
        return True
    if normalized.startswith("false"):
        return False
    return None


def create_openai_client():
    """Create a lightweight OpenAI-compatible client for Loogle judging."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY must be set for Loogle LLM-as-Judge evaluation"
        )

    client_kwargs: dict[str, str] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def evaluate_loogle(
    prediction_records: list[dict[str, Any]],
    qa_records: list[dict[str, Any]],
    judge_model: str,
) -> dict[str, float | int]:
    client = create_openai_client()

    correct = 0
    failures = 0
    total = len(qa_records)

    for index, (prediction_record, qa_record) in enumerate(
        zip(prediction_records, qa_records),
        start=1,
    ):
        prediction = require_string(
            prediction_record,
            "prediction",
            file_label="prediction",
            index=index,
        )
        question = require_string(
            qa_record,
            "query",
            file_label="QA",
            index=index,
        )
        reference = require_string(
            qa_record,
            "answer",
            file_label="QA",
            index=index,
        )
        if not reference.strip():
            raise ValueError(f"'answer' in Loogle QA record {index} must not be empty")

        # Empty predictions count as incorrect without spending a judge API call.
        if not prediction.strip():
            continue

        prompt = build_loogle_judge_prompt(question, reference, prediction)
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            output = response.choices[0].message.content
            if output is None:
                raise ValueError("Judge model returned no message content")
            judgment = parse_boolean_judgment(output)
            if judgment is None:
                failures += 1
                print(
                    f"[WARN] Judge returned an invalid label for sample {index}: "
                    f"{output!r}",
                    file=sys.stderr,
                )
                continue
            correct += int(judgment)
        except Exception as exc:
            failures += 1
            print(
                f"[WARN] Judge failed for sample {index}: {exc}",
                file=sys.stderr,
            )

    return {
        "total_samples": total,
        "LLMJudgeAccuracy": correct / total,
        "judge_failures": failures,
    }


def extract_choice(prediction: str) -> str:
    """Extract the first standalone A/B/C/D choice from a model response."""
    if not prediction:
        return ""

    normalized = prediction.strip().lower()
    match = re.search(r"\b(a|b|c|d)\b", normalized)
    if match:
        return match.group(1)

    if normalized and normalized[0] in {"a", "b", "c", "d"}:
        return normalized[0]

    return ""


def evaluate_lbv2(
    prediction_records: list[dict[str, Any]],
    qa_records: list[dict[str, Any]],
) -> dict[str, float | int]:
    correct = 0

    for index, (prediction_record, qa_record) in enumerate(
        zip(prediction_records, qa_records),
        start=1,
    ):
        prediction = require_string(
            prediction_record,
            "prediction",
            file_label="prediction",
            index=index,
        )
        require_string(
            qa_record,
            "query",
            file_label="QA",
            index=index,
        )
        answer = require_string(
            qa_record,
            "answer",
            file_label="QA",
            index=index,
        ).strip().lower()

        if answer not in {"a", "b", "c", "d"}:
            raise ValueError(
                f"'answer' in LBV2 QA record {index} must be A, B, C, or D; "
                f"got {qa_record['answer']!r}"
            )

        # Empty or unparsable predictions yield an empty choice and score zero.
        correct += int(extract_choice(prediction) == answer)

    total = len(qa_records)
    return {
        "total_samples": total,
        "Accuracy": correct / total,
    }


def print_results(dataset: str, results: dict[str, float | int]) -> None:
    print(f"Evaluation dataset: {dataset}")
    print(f"Total samples: {results['total_samples']}")

    if dataset == "narrative":
        print(f"F1: {results['F1']:.4f}")
        print(f"Recall: {results['Recall']:.4f}")
        print(f"ExactMatch: {results['ExactMatch']:.4f}")
    elif dataset == "loogle":
        print(f"LLMJudgeAccuracy: {results['LLMJudgeAccuracy']:.4f}")
        print(f"Judge failures: {results['judge_failures']}")
    else:
        print(f"Accuracy: {results['Accuracy']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Narrative, Loogle, or LongBench v2 predictions."
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        required=True,
        help="JSONL file containing prediction and context fields.",
    )
    parser.add_argument(
        "--qa-file",
        type=Path,
        required=True,
        help="JSONL QA file containing query and answer fields.",
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Evaluation dataset.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="LLM judge model used only for Loogle evaluation (default: gpt-4o).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prediction_records = load_jsonl(args.prediction_file.resolve())
    qa_records = load_jsonl(args.qa_file.resolve())
    validate_alignment(prediction_records, qa_records)

    if args.dataset == "narrative":
        results = evaluate_narrative(prediction_records, qa_records)
    elif args.dataset == "loogle":
        results = evaluate_loogle(
            prediction_records,
            qa_records,
            judge_model=args.judge_model,
        )
    else:
        results = evaluate_lbv2(prediction_records, qa_records)

    print_results(args.dataset, results)


if __name__ == "__main__":
    main()