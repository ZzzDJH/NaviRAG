from __future__ import annotations

import datetime
import functools
import json
import os
import re
import threading
import time
import traceback
from collections import defaultdict
from glob import glob
from typing import List, Optional
from collections import Counter

import requests
import torch
from difflib import SequenceMatcher
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI


_TOKEN_LOG_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    """Create the OpenAI-compatible client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY must be set when the API LLM backend is used"
        )

    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def chatgpt(
    messages,
    gpt_model: str = "gpt-4o",
    token_log_path: Optional[str] = None,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    response = _get_openai_client().chat.completions.create(
        model=gpt_model,
        messages=messages,
        temperature=0.1,
    )
    answer = response.choices[0].message.content
    if answer is None:
        raise ValueError(f"The model returned no message content: {response}")

    if token_log_path and response.usage is not None:
        absolute_log_path = os.path.abspath(token_log_path)
        os.makedirs(os.path.dirname(absolute_log_path), exist_ok=True)
        usage = response.usage
        log_line = (
            f"{datetime.datetime.now().isoformat()} | model: {gpt_model} | "
            f"prompt_tokens: {usage.prompt_tokens}, "
            f"completion_tokens: {usage.completion_tokens}, "
            f"total_tokens: {usage.total_tokens}\n"
        )
        with _TOKEN_LOG_LOCK:
            with open(absolute_log_path, "a", encoding="utf-8") as file:
                file.write(log_line)

    return answer


class LLM:
    """Minimal HTTP client for an externally deployed vLLM service."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
    ):
        if not model_name:
            raise ValueError("model_name must not be empty.")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def asking_vllm(
        self,
        prompt_or_prompts: str | list[str],
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> str | list[str]:
        """Generate one or more responses through the vLLM HTTP endpoint."""
        is_batch = isinstance(prompt_or_prompts, list)
        prompts = prompt_or_prompts if is_batch else [prompt_or_prompts]

        results: list[str] = []

        for prompt in prompts:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                },
            )

            response.raise_for_status()

            response_data = response.json()
            result_text = response_data["choices"][0]["message"]["content"]
            results.append(result_text.strip())

        return results if is_batch else results[0]

    def generate(
        self,
        prompt_or_prompts: str | list[str],
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1280,
    ) -> str | list[str]:
        """Expose a shared generation interface for the main pipeline."""
        return self.asking_vllm(
            prompt_or_prompts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )


def input_file(file_path: str) -> list[dict]:
    """Load a JSONL input file into memory."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def data_prepare(
    data,
    max_words_per_segment: int = 512,
    overlap_rate: float = 0.0,
    is_save: bool = False,
    save_path: str = "segments.jsonl",
    start_index: int = 0,
):
    """Split document content into indexed segments."""
    overlap = int(max_words_per_segment * overlap_rate) if overlap_rate > 0 else 0
    splitter = SentenceSplitter(
        chunk_size=max_words_per_segment,
        chunk_overlap=overlap,
    )
    segments = []
    global_index = start_index

    for item in data:
        chunks = splitter.split_text(item["content"])
        for chunk in chunks:
            segments.append(
                {
                    "index": global_index,
                    "title": item["title"],
                    "text": chunk,
                }
            )
            global_index += 1

    if is_save:
        absolute_save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(absolute_save_path), exist_ok=True)
        with open(absolute_save_path, "w", encoding="utf-8") as file:
            for segment in segments:
                file.write(json.dumps(segment, ensure_ascii=False) + "\n")

    return segments


def batch_process_by_title(segments):
    """Yield one batch for each document title, preserving input order."""
    grouped = defaultdict(list)
    for segment in segments:
        grouped[segment["title"]].append(segment)

    yield from grouped.items()


def sim_matching(title: str, candidate: str) -> float:
    """Return character-level similarity between two topic titles."""
    return SequenceMatcher(None, title.strip(), candidate.strip()).ratio()


def parse_titles_from_response(
    response: str,
    valid_titles: List[str],
    temp_file_path: Optional[str] = None,
    sim_threshold: float = 0.92,
) -> Optional[List[str]]:
    """Parse selected titles from an LLM response and match minor variations."""
    parsed_titles = []
    output_file = None

    if temp_file_path:
        absolute_path = os.path.abspath(temp_file_path)
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
        output_file = open(absolute_path, "a", encoding="utf-8")

    try:
        for line in response.strip().splitlines():
            if output_file is not None:
                output_file.write(f"{line}\n")

            parts = line.strip().split("//")
            if len(parts) != 3:
                continue

            _, title, _ = map(str.strip, parts)
            if title in {"-1", "None", "", "none"}:
                continue

            if title not in valid_titles:
                best_match = None
                best_score = 0.0
                for candidate in valid_titles:
                    score = sim_matching(title, candidate)
                    if score > best_score:
                        best_match = candidate
                        best_score = score

                if best_score < sim_threshold:
                    continue

                print(
                    f"[SIMILARITY MATCH] Replaced '{title}' with "
                    f"'{best_match}' ({best_score:.2f})"
                )
                title = best_match

            parsed_titles.append(title)
    finally:
        if output_file is not None:
            output_file.close()

    return parsed_titles or None


def retry(max_retries: int = 3, sleep_time: float = 5, allow_empty: bool = True):
    """Retry a synchronous function when it raises or returns an empty result."""

    def retry_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    if not allow_empty and result in (None, "", []):
                        raise ValueError("Result is null or empty")
                    return result
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    retries += 1
                    print(
                        f"[RETRY] Attempt {retries}/{max_retries} for "
                        f"{func.__name__} failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    traceback.print_exc()
                    if retries < max_retries:
                        time.sleep(sleep_time)

            raise ValueError(
                f"{func.__name__} failed after {max_retries} attempts"
            )

        return wrapper

    return retry_decorator


def is_invalid_title(response: str) -> bool:
    """Detect explanatory sentences that should not be used as topic titles."""
    stripped_response = response.strip()
    lowercase_response = stripped_response.lower()

    invalid_starts = (
        "这是",
        "这个",
        "我认为",
        "总结为",
        "this is",
        "the title is",
        "i suggest",
        "my suggestion is",
    )
    if any(lowercase_response.startswith(phrase) for phrase in invalid_starts):
        return True

    if re.search(r".+是.+的", stripped_response) or re.search(
        r"\bis a\b", lowercase_response
    ):
        return True

    invalid_phrases = ("为你生成", "is a good title", "the above")
    return any(phrase in lowercase_response for phrase in invalid_phrases)


def merge_similar_keys(data, emb_model, similarity_threshold: float = 0.8):
    """Merge similarly named leaf nodes across a nested dictionary."""
    key_locations = {}

    def are_keys_similar(key1, key2) -> bool:
        embedding_a = emb_model.encode(key1, convert_to_tensor=True).unsqueeze(0)
        embedding_b = emb_model.encode(key2, convert_to_tensor=True).unsqueeze(0)
        similarity = torch.nn.functional.cosine_similarity(
            embedding_a,
            embedding_b,
            dim=-1,
        )
        return bool((similarity >= similarity_threshold).item())

    def find_representative_key(key):
        for representative_key in key_locations:
            if are_keys_similar(key, representative_key):
                print(f"[MERGE] {key} -> {representative_key}")
                return representative_key
        return None

    def traverse(current_dict):
        keys_to_delete = []
        for key in list(current_dict.keys()):
            if key == "others":
                continue

            value = current_dict[key]
            if isinstance(value, dict):
                traverse(value)
                continue

            representative_key = find_representative_key(key)
            if representative_key:
                first_parent, first_key = key_locations[representative_key]
                first_parent[first_key] += value
                keys_to_delete.append(key)
            else:
                key_locations[key] = (current_dict, key)

        for key in keys_to_delete:
            del current_dict[key]

    traverse(data)


def load_all_temp_dicts(temp_dir: str) -> dict:
    """Load and merge all temporary JSON dictionaries in filename order."""
    current_dict = {}
    json_files = sorted(glob(os.path.join(temp_dir, "temp_*.json")))

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for key, value in data.items():
            if key not in current_dict:
                current_dict[key] = value
            elif isinstance(current_dict[key], dict) and isinstance(value, dict):
                current_dict[key].update(value)
            else:
                current_dict[key] = value

    return current_dict


def parse_outline_to_dict(text: str) -> dict:
    """Parse a '//'-separated outline into an empty topic dictionary."""
    stripped_text = text.strip()
    if not stripped_text:
        raise ValueError("The outline response is empty")

    topics = [topic.strip() for topic in stripped_text.split("//") if topic.strip()]
    if len(topics) < 2:
        raise ValueError(
            f"Only {len(topics)} topic(s) were detected in the outline response: "
            f"{stripped_text}"
        )

    return {topic: "" for topic in topics}


@retry(max_retries=3, sleep_time=1)
def outline_second(
    prompt,
    llm,
    is_vllm,
    gpt_model,
    token_log_path: Optional[str] = None,
):
    """Generate and parse the initial topic outline."""
    if is_vllm:
        response = llm.generate(prompt)
    else:
        response = chatgpt(
            [{"role": "user", "content": prompt}],
            gpt_model,
            token_log_path=token_log_path,
        )

    return parse_outline_to_dict(response.strip())


def clean_empty_keys(data):
    """Recursively remove keys whose values are empty."""
    if not isinstance(data, dict):
        return data

    cleaned = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned_value = clean_empty_keys(value)
            if cleaned_value:
                cleaned[key] = cleaned_value
        elif isinstance(value, list):
            cleaned_list = [
                clean_empty_keys(item) if isinstance(item, dict) else item
                for item in value
            ]
            cleaned_list = [item for item in cleaned_list if item or item == 0]
            if cleaned_list:
                cleaned[key] = cleaned_list
        elif value or value == 0:
            cleaned[key] = value

    return cleaned


def check_key_loss_after_clustering(original_dict, clustered_dict):
    """Report original keys that disappeared after one clustering operation."""
    original_keys = set(original_dict.keys())
    preserved_keys = set()

    for key, value in clustered_dict.items():
        preserved_keys.add(key)
        if isinstance(value, dict):
            preserved_keys.update(value.keys())

    missing_keys = original_keys - preserved_keys
    if missing_keys:
        print(f"[WARN] {len(missing_keys)} key(s) were lost after clustering:")
        for key in sorted(missing_keys):
            print(f"  - {key}")


def find_shallowest_str_node(data: dict) -> list[str] | None:
    """Return the path to the shallowest and shortest string leaf."""
    from collections import deque

    queue = deque([(data, [])])
    candidates = []

    while queue:
        current_dict, path = queue.popleft()
        for key, value in current_dict.items():
            new_path = path + [key]
            if isinstance(value, str):
                candidates.append((len(new_path), len(value), new_path))
            elif isinstance(value, dict):
                queue.append((value, new_path))

    if not candidates:
        return None

    candidates.sort()
    return candidates[0][2]


def handle_title_conflict(current_dict: dict, title: str, new_content: str):
    """Append content to an existing title without replacing nested data."""
    existing_value = current_dict[title]

    if isinstance(existing_value, str):
        current_dict[title] = existing_value + new_content
        return

    if isinstance(existing_value, dict):
        target_path = find_shallowest_str_node(existing_value)
        if target_path is None:
            print(
                f"[WARN] No string leaf was found under '{title}'. "
                "The new content was skipped."
            )
            return
        update_nested_value(existing_value, target_path, new_content)
        return

    print(
        f"[WARN] Cannot handle title '{title}' with value type "
        f"{type(existing_value).__name__}."
    )


def update_nested_value(data: dict, path: list[str], append_text: str):
    """Append text to a string leaf at a nested dictionary path."""
    current = data
    for key in path[:-1]:
        current = current[key]

    leaf_key = path[-1]
    if not isinstance(current[leaf_key], str):
        raise ValueError(f"The final node at path {path} is not a string")
    current[leaf_key] += append_text


def remove_redundant_indices(split_result, repeated, index_to_titles):
    """Remove repeated segment indices from the longest assigned child node."""
    if not repeated:
        return

    print("[FIX] Removing repeated indices from the longest child node:")
    for index in repeated:
        titles = index_to_titles[index]
        print(f"  - [{index}] appears in: {titles}")

        def node_length(text):
            return len(text.strip().replace("\n", ""))

        longest_title = max(titles, key=lambda item: node_length(split_result[item]))
        pattern = rf"\[{index}\].*?(?=(?:\[\d+\])|$)"
        split_result[longest_title] = (
            re.sub(pattern, "", split_result[longest_title], flags=re.DOTALL).strip()
            + "\n"
        )

        if not split_result[longest_title].strip():
            print(f"    -> Removed empty child topic '{longest_title}'")
            del split_result[longest_title]


def fix_missing_indices(split_result, missing, indexed_map, index_to_titles):
    """Assign missing segment indices to the shortest child node."""
    if not missing:
        return

    print(
        "[FIX] Assigning missing indices to the shortest child node: "
        f"{sorted(missing)}"
    )

    def node_length(text):
        return len(text.strip().replace("\n", ""))

    shortest_title = min(
        split_result.items(),
        key=lambda item: node_length(item[1]),
    )[0]

    for index in sorted(missing):
        if index in indexed_map:
            split_result[shortest_title] += f"[{index}]{indexed_map[index]}\n"
            index_to_titles[index].append(shortest_title)

def check_duplicate_keys(key_list):
    key_counter = Counter(key_list)
    duplicates = {k: v for k, v in key_counter.items() if v > 1}
    if duplicates:
        print(f"[WARNING] Duplicate title (key) detected:")
        for k, v in duplicates.items():
            print(f"- {k}: appeared {v} times")
    else:
        print("[Check passed] No duplicate titles found.")