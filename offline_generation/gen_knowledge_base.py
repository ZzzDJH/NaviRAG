from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from gen_utils import *
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer


wiki_prompt_json: Dict[str, Any] = {}


@dataclass
class GenerationConfig:
    input_file_path: str
    output_dir: str
    prompt_path: str

    language: str = "English"

    max_words_per_segment: int = 512
    overlap_rate: float = 0.0
    save_segments: bool = False

    is_vllm: bool = False
    model_path: Optional[str] = None
    gpt_model: str = "gpt-4o"
    embedding_path: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cuda:1"

    group_output_by_title: bool = True
    merge_similar_topics: bool = False

    max_sub_batch_size: int = 250
    start_batch_id: int = 0

    enable_document_parallelism: bool = False
    max_concurrency: int = 8
    enable_sub_batch_parallelism: bool = True

    max_leaf_tokens: int = 1536
    max_topics_per_level: int = 12
    selected_topic_count: int = 2
    max_selected_topics: int = 4
    summary_batch_size: int = 10
    similarity_threshold: float = 0.8
    new_topic_parse_retries: int = 3

    @property
    def output_file(self) -> str:
        """Return the stage-one wiki output path."""
        return os.path.join(self.output_dir, "wiki_ori.jsonl")


    @property
    def save_path(self) -> str:
        """Return the segmented document output path."""
        return os.path.join(self.output_dir, "segments.jsonl")

    def validate(self) -> None:
        if not self.input_file_path:
            raise ValueError("input_file_path must not be empty")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        if not self.prompt_path:
            raise ValueError("prompt_path must not be empty")
        if self.max_words_per_segment <= 0:
            raise ValueError("max_words_per_segment must be positive")
        if not 0.0 <= self.overlap_rate < 1.0:
            raise ValueError("overlap_rate must be in the range [0.0, 1.0)")
        if self.max_sub_batch_size <= 0:
            raise ValueError("max_sub_batch_size must be positive")
        if self.start_batch_id < 0:
            raise ValueError("start_batch_id must be non-negative")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if self.max_leaf_tokens <= 0:
            raise ValueError("max_leaf_tokens must be positive")
        if self.max_topics_per_level <= 0:
            raise ValueError("max_topics_per_level must be positive")
        if self.selected_topic_count <= 0:
            raise ValueError("selected_topic_count must be positive")
        if self.max_selected_topics < self.selected_topic_count:
            raise ValueError(
                "max_selected_topics must be greater than or equal to "
                "selected_topic_count"
            )
        if self.summary_batch_size <= 0:
            raise ValueError("summary_batch_size must be positive")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in the range [0.0, 1.0]")
        if self.new_topic_parse_retries <= 0:
            raise ValueError("new_topic_parse_retries must be positive")


async def run_blocking(semaphore, func, *args, **kwargs):
    """Run a blocking function in a worker thread under a shared semaphore."""
    async with semaphore:
        return await asyncio.to_thread(func, *args, **kwargs)


def request_llm(prompt: str, llm, config: GenerationConfig) -> str:
    """Send a prompt through the configured LLM backend."""
    if config.is_vllm:
        if llm is None:
            raise ValueError("A vLLM client is required when is_vllm is enabled")
        response = llm.generate(prompt)
    else:
        messages = [{"role": "user", "content": prompt}]
        response = chatgpt(messages, config.gpt_model)

    return response.strip()



async def process_topic_branch(
    segment,
    title,
    value,
    tokenizer,
    llm,
    semaphore,
    config: GenerationConfig,
    selection_debug_path: str,
):
    """Process one selected topic on an isolated branch and return its new value."""
    if isinstance(value, str) and not value.strip():
        branch_wiki = {title: value}
        await run_blocking(
            semaphore,
            add_topic_content,
            branch_wiki,
            [],
            title,
            segment,
            llm,
            config,
        )
        return title, branch_wiki[title]

    if isinstance(value, str):
        branch_wiki = {title: value}
        await run_blocking(
            semaphore,
            merge_topic_content,
            branch_wiki,
            [],
            title,
            segment,
            tokenizer,
            llm,
            config,
        )
        return title, branch_wiki[title]

    if isinstance(value, dict):
        branch_wiki = {title: copy.deepcopy(value)}
        await insert_segment(
            segment,
            branch_wiki,
            [title],
            tokenizer,
            llm,
            semaphore=semaphore,
            config=config,
            selection_debug_path=selection_debug_path,
        )
        return title, branch_wiki[title]

    raise ValueError(f"Unexpected value type for title '{title}': {type(value)}")


def merge_dicts_without_overwrite(target, source, prefix=""):
    """Merge dictionaries while preserving colliding keys with a suffix."""
    if source is None:
        print(f"[WARN] Empty sub-batch result: {prefix}")
        return

    for key, value in source.items():
        new_key = key
        if new_key in target:
            new_key = f"{key}__{prefix}"
        target[new_key] = value


@retry(max_retries=10, sleep_time=1)
def get_llm_cluster_labels(
    titles: List[str],
    llm,
    config: GenerationConfig,
) -> List[int]:
    """Cluster topic titles with the LLM and return one label per title."""
    prompt_template = wiki_prompt_json[config.language]["LLM_cluster"]
    formatted_titles = "\n".join(
        f"{index + 1}. {title}" for index, title in enumerate(titles)
    )
    prompt = prompt_template.replace("{title_list}", formatted_titles)
    response = request_llm(prompt, llm, config)

    cluster_groups = []
    for line in response.splitlines():
        line = line.strip()
        if not line or "//" not in line:
            continue
        try:
            indexes = [int(item.strip()) - 1 for item in line.split("//")]
        except ValueError:
            continue

        indexes = [index for index in indexes if 0 <= index < len(titles)]
        if len(indexes) >= 2:
            cluster_groups.append(indexes)

    if not cluster_groups:
        raise ValueError("No valid clusters could be parsed from the LLM response")

    labels = [-1] * len(titles)
    for cluster_id, group in enumerate(cluster_groups):
        for index in group:
            labels[index] = cluster_id

    next_cluster_id = len(cluster_groups)
    for index, label in enumerate(labels):
        if label == -1:
            labels[index] = next_cluster_id
            next_cluster_id += 1

    print(
        f"[LLM CLUSTER] {len(titles)} titles -> "
        f"{len(set(labels))} clusters"
    )
    return labels


@retry(max_retries=10, sleep_time=1, allow_empty=False)
def generate_cluster_title(
    cluster_titles: List[str],
    all_titles: List[str],
    existing_parent_titles: List[str],
    llm,
    config: GenerationConfig,
) -> Optional[str]:
    """Generate a parent title for one cluster of topic titles."""
    prompt_template = wiki_prompt_json[config.language]["get_parent_title_new"]
    prompt = prompt_template.format(
        now_titles=", ".join(cluster_titles),
        all_titles=", ".join(all_titles),
        other_parent_titles=", ".join(existing_parent_titles),
    )
    response = request_llm(prompt, llm, config)

    if not response or response.lower() in {"none", "无", "null"}:
        raise ValueError(f"Invalid cluster title: '{response}'")
    if is_invalid_title(response):
        raise ValueError(f"The generated cluster title appears to be a sentence: '{response}'")

    return response


def cluster_topics(
    current_dict,
    llm,
    config: GenerationConfig,
) -> Dict[str, Any]:
    """Group sibling topics into an LLM-generated hierarchy."""
    titles = list(current_dict.keys())
    values = list(current_dict.values())

    if len(titles) <= 1:
        return copy.deepcopy(current_dict)

    check_duplicate_keys(titles)
    labels = get_llm_cluster_labels(titles, llm, config)

    clustered_dict: Dict[str, Any] = {}
    existing_parent_titles: List[str] = []

    for cluster_id in sorted(set(labels)):
        indexes = [index for index, label in enumerate(labels) if label == cluster_id]
        cluster_titles = [titles[index] for index in indexes]
        cluster_values = [values[index] for index in indexes]
        cluster_map = {titles[index]: values[index] for index in indexes}

        if len(cluster_titles) == 1:
            clustered_dict[cluster_titles[0]] = cluster_values[0]
            continue

        cluster_title = generate_cluster_title(
            cluster_titles,
            titles,
            existing_parent_titles,
            llm,
            config,
        )
        if not cluster_title:
            cluster_title = "Untitled Topic"

        if cluster_title in clustered_dict:
            existing = clustered_dict[cluster_title]
            if isinstance(existing, dict):
                existing.update(cluster_map)
                print(
                    f"[WARN] Duplicate cluster title '{cluster_title}'. "
                    f"Merged {len(cluster_map)} topics."
                )
            else:
                print(
                    f"[WARN] Cluster title '{cluster_title}' collided with a leaf. "
                    "Converted it to a nested topic."
                )
                clustered_dict[cluster_title] = {
                    cluster_title: existing,
                    **cluster_map,
                }
        else:
            clustered_dict[cluster_title] = cluster_map

        existing_parent_titles.append(cluster_title)

    check_key_loss_after_clustering(current_dict, clustered_dict)
    return clustered_dict


@retry(max_retries=10, sleep_time=1)
def select_topics(
    segment,
    titles,
    llm,
    config: GenerationConfig,
    selection_debug_path: str,
) -> Optional[List[str]]:
    """Select existing topics that are relevant to a segment."""
    text = segment["text"].replace("\n", "").replace("\r", "")
    outlines = "\n".join(titles)
    prompt_template = wiki_prompt_json[config.language]["select_titles_new"]
    prompt = prompt_template.format(
        outlines=outlines,
        text=text,
        select_num=config.selected_topic_count,
    )
    response = request_llm(prompt, llm, config)

    return parse_titles_from_response(
        response,
        titles,
        temp_file_path=selection_debug_path,
    )


def parse_new_topic(response: str) -> Tuple[str, str]:
    """Parse the expected '// title // summary' topic response."""
    raw_response = response.strip()
    parts = raw_response.split("//", 2)
    if len(parts) != 3:
        raise ValueError("The new-topic response does not contain three fields")

    _, title, summary = map(str.strip, parts)
    if not title or not summary or title.lower() in {"none", "-1"}:
        raise ValueError("The new-topic response contains an invalid title or summary")

    return title, summary


def generate_new_topic(
    prompt: str,
    llm,
    config: GenerationConfig,
) -> Tuple[str, str]:
    """Retry malformed new-topic responses and use a final fallback if necessary."""
    last_response = ""
    for attempt in range(1, config.new_topic_parse_retries + 1):
        last_response = request_llm(prompt, llm, config)
        try:
            return parse_new_topic(last_response)
        except ValueError as error:
            if attempt < config.new_topic_parse_retries:
                print(
                    f"[WARN] New-topic response parsing failed "
                    f"({attempt}/{config.new_topic_parse_retries}): {error}"
                )
            else:
                print(
                    "[WARN] New-topic response parsing failed on the final "
                    "attempt. Falling back to title 'part'."
                )

    return "part", last_response


@retry(max_retries=10, sleep_time=1)
def create_topic(
    current_wiki,
    path,
    segment,
    titles,
    llm,
    config: GenerationConfig,
):
    """Create a new topic for a segment and cluster the level when necessary."""
    current_dict = get_by_path(current_wiki, path)
    text = segment["text"].replace("\n", "").replace("\r", "")
    index = segment["index"]
    outlines = "\n".join(titles)

    if path:
        prompt_template = wiki_prompt_json[config.language]["generate_new_node_parent"]
        prompt = prompt_template.format(
            outlines=outlines,
            text=text,
            parent_title=path[-1],
        )
    else:
        prompt_template = wiki_prompt_json[config.language]["generate_new_node_detail"]
        prompt = prompt_template.format(outlines=outlines, text=text)

    new_title, new_content = generate_new_topic(prompt, llm, config)
    new_content = f"[{index}]{new_content}\n"

    if new_title in titles:
        print(
            f"[WARN] Duplicate generated topic '{new_title}'. "
            "Applying the configured conflict handler."
        )
        handle_title_conflict(current_dict, new_title, new_content)
    else:
        current_dict[new_title] = new_content

    if len(current_dict) > config.max_topics_per_level:
        clustered_dict = cluster_topics(current_dict, llm, config)
        set_by_path(current_wiki, path, clustered_dict)


def get_by_path(data, path):
    """Return the nested value at a key path."""
    current = data
    for key in path:
        current = current[key]
    return current


def set_by_path(data, path, value):
    """Replace the nested value at a key path."""
    if not path:
        if not isinstance(data, dict):
            raise TypeError("Expected a dictionary at the root")
        data.clear()
        data.update(value)
        return

    current = data
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def add_topic_content(
    current_wiki,
    path,
    title,
    segment,
    llm,
    config: GenerationConfig,
):
    """Fill an empty topic with content generated from one segment."""
    current_dict = get_by_path(current_wiki, path)
    current_dict[title] = generate_initial_topic_content(segment, title, llm, config)


@retry(max_retries=10, sleep_time=1)
def generate_initial_topic_content(
    segment,
    title,
    llm,
    config: GenerationConfig,
) -> str:
    """Generate the first indexed content item for an empty topic."""
    text = segment["text"].replace("\n", "").replace("\r", "")
    index = segment["index"]
    prompt_template = wiki_prompt_json[config.language]["add_content_detail"]
    prompt = prompt_template.format(topic=title, supply_content=text)
    response = request_llm(prompt, llm, config)
    new_content = parse_merged_content(response)
    return f"[{index}]{new_content}\n"


def parse_merged_content(response: str) -> str:
    """Extract content wrapped in the expected '<<...>>' markers."""
    match = re.search(r"<<(.*?)>>", response, re.DOTALL)
    if not match:
        print(f"[ERROR] Unparsed response: {response}")
        raise ValueError("The response does not contain content wrapped in << >>")
    return match.group(1).strip()


@retry(max_retries=10, sleep_time=1)
def generate_merged_topic_content(
    old_value,
    segment,
    title,
    llm,
    config: GenerationConfig,
) -> str:
    """Merge a segment into an existing leaf topic."""
    text = segment["text"].replace("\n", "").replace("\r", "")
    index = segment["index"]
    prompt_template = wiki_prompt_json[config.language]["content_merge_detail"]
    prompt = prompt_template.format(
        topic=title,
        exist_content=old_value,
        supply_content=text,
    )
    response = request_llm(prompt, llm, config)
    new_content = parse_merged_content(response)
    return old_value + f"[{index}]{new_content}\n"


@retry(max_retries=10, sleep_time=1)
def split_leaf_content(
    title,
    new_value,
    llm,
    config: GenerationConfig,
):
    """Split an oversized leaf into indexed subtopics."""
    prompt_template = wiki_prompt_json[config.language]["new_layer3"]
    prompt = prompt_template.format(original_title=title, content=new_value)
    response = request_llm(prompt, llm, config)

    split_pattern = r"\[(\d+)\](.*?)(?=(?:\[\d+\])|$)"
    indexed_map = {
        match[0]: match[1].strip()
        for match in re.findall(split_pattern, new_value, re.DOTALL)
    }
    original_indices = set(indexed_map.keys())

    split_result = {}
    all_used_indices = []
    index_to_titles = defaultdict(list)

    for line in response.splitlines():
        if "//" not in line:
            continue
        try:
            sub_title, index_text = line.split("//", 1)
            sub_title = sub_title.strip()
            indices = re.findall(r"\d+", index_text)
            all_used_indices.extend(indices)
            for index in indices:
                index_to_titles[index].append(sub_title)
            indexed_segments = [
                f"[{index}]{indexed_map[index]}"
                for index in indices
                if index in indexed_map
            ]
            split_result[sub_title] = "\n".join(indexed_segments) + "\n"
        except Exception as error:
            print(f"[WARN] Failed to parse split line '{line}': {error}")

    if not split_result:
        raise ValueError("The model did not return a valid split structure")

    used_indices = set(all_used_indices)
    missing_indices = original_indices - used_indices
    repeated_indices = [
        index
        for index, count in Counter(all_used_indices).items()
        if count > 1
    ]

    fix_missing_indices(
        split_result,
        missing_indices,
        indexed_map,
        index_to_titles,
    )
    remove_redundant_indices(
        split_result,
        repeated_indices,
        index_to_titles,
    )
    return split_result


def merge_topic_content(
    current_wiki,
    path,
    title,
    segment,
    tokenizer,
    llm,
    config: GenerationConfig,
):
    """Merge a segment into a leaf and split the leaf when it grows too large."""
    current_dict = get_by_path(current_wiki, path)
    old_value = current_dict[title]
    new_value = generate_merged_topic_content(
        old_value,
        segment,
        title,
        llm,
        config,
    )

    if len(tokenizer.tokenize(new_value)) > config.max_leaf_tokens:
        print(f"[INFO] Splitting oversized topic '{title}'")
        current_dict[title] = split_leaf_content(title, new_value, llm, config)
    else:
        current_dict[title] = new_value


async def insert_segment(
    segment,
    current_wiki,
    path,
    tokenizer,
    llm,
    semaphore,
    config: GenerationConfig,
    selection_debug_path: str,
):
    """Insert one segment into the most relevant location in the wiki tree."""
    if path is None:
        path = []

    current_dict = get_by_path(current_wiki, path)
    titles = list(current_dict.keys())

    selected_titles = await run_blocking(
        semaphore,
        select_topics,
        segment,
        titles,
        llm,
        config,
        selection_debug_path,
    )

    if selected_titles is None:
        await run_blocking(
            semaphore,
            create_topic,
            current_wiki,
            path,
            segment,
            titles,
            llm,
            config,
        )
        return

    selected_titles = list(dict.fromkeys(selected_titles))

    if len(selected_titles) > config.selected_topic_count:
        print(
            f"[WARN] Expected at most {config.selected_topic_count} selected topics, "
            f"but received {len(selected_titles)}."
        )

    if len(selected_titles) > config.max_selected_topics:
        removed_count = len(selected_titles) - config.max_selected_topics
        print(
            f"[WARN] Keeping the first {config.max_selected_topics} topics and "
            f"discarding {removed_count} extra selections."
        )
        selected_titles = selected_titles[: config.max_selected_topics]

    tasks = []
    for title in selected_titles:
        if title not in current_dict:
            raise KeyError(f"Selected topic '{title}' does not exist in the current level")
        tasks.append(
            process_topic_branch(
                segment=segment,
                title=title,
                value=current_dict[title],
                tokenizer=tokenizer,
                llm=llm,
                semaphore=semaphore,
                config=config,
                selection_debug_path=selection_debug_path,
            )
        )

    results = await asyncio.gather(*tasks)

    current_dict = get_by_path(current_wiki, path)
    for title, updated_value in results:
        current_dict[title] = updated_value


def build_initial_outline(
    segments,
    use_document_title,
    llm,
    config: GenerationConfig,
):
    """Summarize a batch and generate its initial topic outline."""
    summaries = []
    for start in range(0, len(segments), config.summary_batch_size):
        batch_text = "\n".join(
            segment["text"]
            for segment in segments[start : start + config.summary_batch_size]
        )
        prompt_template = wiki_prompt_json[config.language]["new_summary"]
        prompt = prompt_template.format(summary_segment=batch_text)
        summaries.append(request_llm(prompt, llm, config))

    combined_summary = "\n".join(summaries)
    if use_document_title:
        title = segments[0]["title"]
        outline_prompt = wiki_prompt_json[config.language]["new_outline_title"].format(
            brief_summary=combined_summary,
            title=title,
        )
    else:
        outline_prompt = wiki_prompt_json[config.language]["new_outline"].format(
            brief_summary=combined_summary,
        )

    return outline_second(
        outline_prompt,
        llm,
        config.is_vllm,
        config.gpt_model,
    )


async def _generate_batch_wiki(
    segments,
    batch_id,
    tokenizer,
    llm,
    semaphore,
    config: GenerationConfig,
    temp_file_path,
    use_document_title,
):
    """Generate one wiki dictionary while processing segments sequentially."""
    print(f"[INFO] Initializing outline for batch {batch_id}")
    initial_outline = await run_blocking(
        semaphore,
        build_initial_outline,
        segments,
        use_document_title,
        llm,
        config,
    )
    current_dict = copy.deepcopy(initial_outline)

    with open(temp_file_path, "w", encoding="utf-8") as file:
        json.dump(current_dict, file, ensure_ascii=False, indent=4)

    print(f"[INFO] Building wiki recursively for batch {batch_id}")
    selection_debug_path = f"{temp_file_path}.selected_titles.log"

    for segment in tqdm(segments, desc=f"SubBatch {batch_id} Processing"):
        await insert_segment(
            segment,
            current_dict,
            path=None,
            tokenizer=tokenizer,
            llm=llm,
            semaphore=semaphore,
            config=config,
            selection_debug_path=selection_debug_path,
        )

        with open(temp_file_path, "w", encoding="utf-8") as file:
            json.dump(current_dict, file, ensure_ascii=False, indent=4)

    return current_dict


async def generate_document_wiki(
    segments,
    title,
    batch_id,
    tokenizer,
    llm,
    semaphore,
    config: GenerationConfig,
    temp_dir,
    sub_temp_dir,
):
    """Generate the wiki for one document batch and persist its temporary output."""
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"temp_{batch_id}.json")
    if os.path.exists(temp_file_path):
        print(f"[CLEAN] Removing existing temporary file: {temp_file_path}")
        os.remove(temp_file_path)

    total_segments = len(segments)
    use_sub_batches = total_segments > config.max_sub_batch_size

    if use_sub_batches:
        document_temp_dir = os.path.join(sub_temp_dir, f"batch_{batch_id}")
        if os.path.exists(document_temp_dir):
            print(f"[CLEAN] Removing existing temporary directory: {document_temp_dir}")
            shutil.rmtree(document_temp_dir)
        os.makedirs(document_temp_dir, exist_ok=True)

        sub_batches = [
            segments[start : start + config.max_sub_batch_size]
            for start in range(0, total_segments, config.max_sub_batch_size)
        ]

        async def run_one_sub_batch(index, sub_segments):
            part_name = f"part_{index + 1}"
            sub_temp_file = os.path.join(document_temp_dir, f"{part_name}.json")
            print(f"\n====== Processing {part_name} ======\n")
            sub_dict = await _generate_batch_wiki(
                segments=sub_segments,
                batch_id=f"{batch_id}_{index}",
                tokenizer=tokenizer,
                llm=llm,
                semaphore=semaphore,
                config=config,
                temp_file_path=sub_temp_file,
                use_document_title=False,
            )
            return index, part_name, sub_dict

        if config.enable_sub_batch_parallelism:
            results = await asyncio.gather(
                *[
                    run_one_sub_batch(index, sub_segments)
                    for index, sub_segments in enumerate(sub_batches)
                ]
            )
            results.sort(key=lambda result: result[0])
        else:
            results = []
            for index, sub_segments in enumerate(sub_batches):
                results.append(await run_one_sub_batch(index, sub_segments))

        merged_dict = {}
        for _, part_name, sub_dict in results:
            merge_dicts_without_overwrite(
                merged_dict,
                sub_dict,
                prefix=part_name,
            )

        final_dict = (
            {title: merged_dict}
            if config.group_output_by_title
            else merged_dict
        )
        with open(temp_file_path, "w", encoding="utf-8") as file:
            json.dump(final_dict, file, ensure_ascii=False, indent=4)
        return

    current_dict = await _generate_batch_wiki(
        segments=segments,
        batch_id=batch_id,
        tokenizer=tokenizer,
        llm=llm,
        semaphore=semaphore,
        config=config,
        temp_file_path=temp_file_path,
        use_document_title=True,
    )

    final_dict = (
        {title: current_dict}
        if config.group_output_by_title
        else current_dict
    )
    with open(temp_file_path, "w", encoding="utf-8") as file:
        json.dump(final_dict, file, ensure_ascii=False, indent=4)


def merge_generated_outputs(
    temp_dir,
    output_file,
    embedder,
    llm,
    config: GenerationConfig,
):
    """Merge all temporary wiki files into the final output."""
    loaded_dict = load_all_temp_dicts(temp_dir)
    current_dict = clean_empty_keys(loaded_dict)

    if not config.group_output_by_title:
        current_dict = cluster_topics(current_dict, llm, config)
        if config.merge_similar_topics:
            if embedder is None:
                raise ValueError(
                    "An embedding model is required when merge_similar_topics is enabled"
                )
            merge_similar_keys(
                current_dict,
                embedder,
                similarity_threshold=config.similarity_threshold,
            )

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(current_dict, file, ensure_ascii=False, indent=4)


def load_prompt_config(prompt_path: str) -> Dict[str, Any]:
    """Load the prompt templates used by the wiki generation pipeline."""
    with open(prompt_path, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_structured_wiki(config: GenerationConfig) -> None:
    """Run structured wiki generation from a validated configuration."""
    config.validate()
    asyncio.run(generate_structured_wiki_async(config))


async def generate_structured_wiki_async(config: GenerationConfig) -> None:
    """Asynchronous entry point for structured wiki generation."""
    global wiki_prompt_json
    wiki_prompt_json = load_prompt_config(config.prompt_path)

    print("[INFO] Initializing tokenizer and LLM client")
    tokenizer = AutoTokenizer.from_pretrained(config.embedding_path)

    llm = None
    if config.is_vllm:
        llm = LLM(model_name=config.model_path)

    embedder = None
    if not config.group_output_by_title and config.merge_similar_topics:
        embedder = SentenceTransformer(
            config.embedding_path,
            device=config.embedding_device,
        )

    start_time = time.time()
    data = input_file(config.input_file_path)


    output_dir = os.path.abspath(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = os.path.join(output_dir, "wiki_ori_temp")
    sub_temp_dir = os.path.join(output_dir, "wiki_ori_sub_temp")


    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(sub_temp_dir, exist_ok=True)

    print("[INFO] Splitting documents into segments")
    segments = data_prepare(
        data,
        max_words_per_segment=config.max_words_per_segment,
        overlap_rate=config.overlap_rate,
        is_save=config.save_segments,
        save_path=config.save_path,
        start_index=0,
    )

    print("[INFO] Preparing document batches")
    pending_batches = []
    for batch_id, (title, batch) in enumerate(batch_process_by_title(segments)):
        if batch_id < config.start_batch_id:
            print(
                f"[SKIP] Batch {batch_id} skipped "
                f"(starting from batch {config.start_batch_id})"
            )
            continue
        pending_batches.append((batch_id, title, batch))

    semaphore = asyncio.Semaphore(config.max_concurrency)

    if config.enable_document_parallelism:
        await asyncio.gather(
            *[
                generate_document_wiki(
                    segments=batch,
                    title=title,
                    batch_id=batch_id,
                    tokenizer=tokenizer,
                    llm=llm,
                    semaphore=semaphore,
                    config=config,
                    temp_dir=temp_dir,
                    sub_temp_dir=sub_temp_dir,
                )
                for batch_id, title, batch in pending_batches
            ]
        )
    else:
        for batch_id, title, batch in pending_batches:
            await generate_document_wiki(
                segments=batch,
                title=title,
                batch_id=batch_id,
                tokenizer=tokenizer,
                llm=llm,
                semaphore=semaphore,
                config=config,
                temp_dir=temp_dir,
                sub_temp_dir=sub_temp_dir,
            )

    print("[INFO] Merging temporary outputs")
    merge_generated_outputs(
        temp_dir=temp_dir,
        output_file=config.output_file,
        embedder=embedder,
        llm=llm,
        config=config,
    )

    elapsed_seconds = time.time() - start_time
    print(f"[INFO] Generation completed in {elapsed_seconds:.4f} seconds")


CONFIG_SECTION_FIELDS = {
    "paths": {
        "input_file_path",
        "output_dir",
        "prompt_path",
    },
    "general": {
        "language",
    },
    "segmentation": {
        "max_words_per_segment",
        "overlap_rate",
        "save_segments",
    },
    "models": {
        "is_vllm",
        "model_path",
        "gpt_model",
        "embedding_path",
        "embedding_device",
    },
    "output": {
        "group_output_by_title",
        "merge_similar_topics",
        "similarity_threshold",
    },
    "batching": {
        "max_sub_batch_size",
        "start_batch_id",
    },
    "parallelism": {
        "enable_document_parallelism",
        "max_concurrency",
        "enable_sub_batch_parallelism",
    },
    "generation": {
        "max_leaf_tokens",
        "max_topics_per_level",
        "selected_topic_count",
        "max_selected_topics",
        "summary_batch_size",
        "new_topic_parse_retries",
    },
}

CONFIG_RELATIVE_PATH_FIELDS = {
    "input_file_path",
    "output_dir",
    "prompt_path",
}


def _flatten_config_sections(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate config sections and flatten them into dataclass arguments."""
    allowed_top_level_keys = set(CONFIG_SECTION_FIELDS) | {"config_version"}
    unknown_sections = set(raw_config) - allowed_top_level_keys
    if unknown_sections:
        raise ValueError(
            "Unknown configuration section(s): "
            + ", ".join(sorted(unknown_sections))
        )

    config_version = raw_config.get("config_version", 1)
    if config_version != 1:
        raise ValueError(
            f"Unsupported config_version: {config_version}. Expected version 1."
        )

    flattened: Dict[str, Any] = {}
    for section_name, allowed_fields in CONFIG_SECTION_FIELDS.items():
        section_data = raw_config.get(section_name, {})
        if section_data is None:
            section_data = {}
        if not isinstance(section_data, dict):
            raise TypeError(
                f"Configuration section '{section_name}' must be a mapping"
            )

        unknown_fields = set(section_data) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown field(s) in section '{section_name}': "
                + ", ".join(sorted(unknown_fields))
            )

        for field_name, value in section_data.items():
            if field_name in flattened:
                raise ValueError(
                    f"Configuration field '{field_name}' was defined more than once"
                )
            flattened[field_name] = value

    return flattened


def _resolve_relative_config_paths(
    config_values: Dict[str, Any],
    config_directory: str,
) -> Dict[str, Any]:
    """Resolve file paths relative to the directory containing the config file."""
    resolved_values = dict(config_values)
    for field_name in CONFIG_RELATIVE_PATH_FIELDS:
        value = resolved_values.get(field_name)
        if not value or os.path.isabs(value):
            continue
        resolved_values[field_name] = os.path.normpath(
            os.path.join(config_directory, value)
        )
    return resolved_values


def load_generation_config(config_path: str) -> GenerationConfig:
    """Load a versioned YAML file into a validated GenerationConfig."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load the generation config. "
            "Install it with: pip install PyYAML"
        ) from exc

    absolute_config_path = os.path.abspath(config_path)
    with open(absolute_config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if raw_config is None:
        raise ValueError("The configuration file is empty")
    if not isinstance(raw_config, dict):
        raise TypeError("The top level of the configuration file must be a mapping")

    config_values = _flatten_config_sections(raw_config)
    config_values = _resolve_relative_config_paths(
        config_values,
        os.path.dirname(absolute_config_path),
    )

    required_fields = {
        field.name
        for field in fields(GenerationConfig)
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing_fields = required_fields - set(config_values)
    if missing_fields:
        raise ValueError(
            "Missing required configuration field(s): "
            + ", ".join(sorted(missing_fields))
        )

    try:
        config = GenerationConfig(**config_values)
    except TypeError as exc:
        raise ValueError(f"Invalid generation configuration: {exc}") from exc

    config.validate()
    return config


def build_argument_parser() -> argparse.ArgumentParser:
    """Build a minimal command-line interface that accepts one config file."""
    parser = argparse.ArgumentParser(
        description="Generate a structured knowledge base for structured RAG."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML generation configuration file.",
    )
    return parser


if __name__ == "__main__":
    command_line_args = build_argument_parser().parse_args()
    generation_config = load_generation_config(command_line_args.config)
    generate_structured_wiki(generation_config)