from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from gen_utils import LLM, chatgpt


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostprocessPaths:
    """File paths produced and consumed by the postprocessing pipeline."""

    output_dir: Path
    wiki_ori: Path
    segments: Path
    summaries: Path
    updated_wiki: Path
    chunks: Path
    vector_chunks: Path
    vector_index: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "PostprocessPaths":
        output_dir = output_dir.resolve()
        return cls(
            output_dir=output_dir,
            wiki_ori=output_dir / "wiki_ori.jsonl",
            segments=output_dir / "segments.jsonl",
            summaries=output_dir / "sum.json",
            updated_wiki=output_dir / "ud_wiki.jsonl",
            chunks=output_dir / "chunks.json",
            vector_chunks=output_dir / "v_chunks.json",
            vector_index=output_dir / "v_index.faiss",
        )


@dataclass(frozen=True)
class PostprocessConfig:
    """Configuration required by the postprocessing pipeline."""

    paths: PostprocessPaths
    prompt_path: Path
    dataset: str

    is_vllm: bool
    gpt_model: str
    embedding_path: str
    embedding_device: str
    token_log_path: Optional[Path]

    group_output_by_title: bool
    summary_max_input_tokens: int = 8192
    leaf_rewrite_retries: int = 1
    vector_batch_size: int = 8
    overwrite_existing: bool = True

    def validate(self) -> None:
        if not self.dataset:
            raise ValueError("postprocessing.dataset must not be empty")
        if not self.embedding_path:
            raise ValueError("models.embedding_path must not be empty")
        if self.summary_max_input_tokens <= 0:
            raise ValueError(
                "postprocessing.summary_max_input_tokens must be positive"
            )
        if self.leaf_rewrite_retries <= 0:
            raise ValueError("postprocessing.leaf_rewrite_retries must be positive")
        if self.vector_batch_size <= 0:
            raise ValueError("postprocessing.vector_batch_size must be positive")
        if not self.group_output_by_title:
            raise ValueError(
                "Postprocessing currently requires output.group_output_by_title=true "
                "because segment-to-wiki mapping compares each chunk's root path "
                "with the source document title."
            )


def _require_mapping(raw: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = raw.get(section_name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise TypeError(f"Configuration section '{section_name}' must be a mapping")
    return section


def _resolve_config_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def load_postprocess_config(config_path: str | Path) -> PostprocessConfig:
    """Load postprocessing settings from the shared generation YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load the configuration file. "
            "Install it with: pip install PyYAML"
        ) from exc

    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if raw is None:
        raise ValueError("The configuration file is empty")
    if not isinstance(raw, dict):
        raise TypeError("The top level of the configuration file must be a mapping")

    paths_section = _require_mapping(raw, "paths")
    models_section = _require_mapping(raw, "models")
    output_section = _require_mapping(raw, "output")
    postprocess_section = _require_mapping(raw, "postprocessing")
    logging_section = _require_mapping(raw, "logging")

    output_dir_value = paths_section.get("output_dir")
    if not output_dir_value:
        raise ValueError("paths.output_dir must be provided")

    prompt_path_value = postprocess_section.get("prompt_path")
    if not prompt_path_value:
        raise ValueError("postprocessing.prompt_path must be provided")

    config_dir = config_path.parent
    output_dir = _resolve_config_path(str(output_dir_value), config_dir)
    prompt_path = _resolve_config_path(str(prompt_path_value), config_dir)

    token_log_value = logging_section.get("token_log_path")
    token_log_path = None
    if token_log_value:
        token_log_path = _resolve_config_path(str(token_log_value), config_dir)

    config = PostprocessConfig(
        paths=PostprocessPaths.from_output_dir(output_dir),
        prompt_path=prompt_path,
        dataset=str(postprocess_section.get("dataset", "")).strip(),
        is_vllm=bool(models_section.get("is_vllm", False)),
        gpt_model=str(models_section.get("gpt_model", "gpt-4o")),
        embedding_path=str(
            models_section.get("embedding_path", "all-MiniLM-L6-v2")
        ),
        embedding_device=str(models_section.get("embedding_device", "cpu")),
        token_log_path=token_log_path,
        group_output_by_title=bool(
            output_section.get("group_output_by_title", True)
        ),
        summary_max_input_tokens=int(
            postprocess_section.get("summary_max_input_tokens", 8192)
        ),
        leaf_rewrite_retries=int(
            postprocess_section.get("leaf_rewrite_retries", 1)
        ),
        vector_batch_size=int(postprocess_section.get("vector_batch_size", 8)),
        overwrite_existing=bool(
            postprocess_section.get("overwrite_existing", True)
        ),
    )
    config.validate()
    return config


def load_json(path: Path) -> Any:
    """Load one JSON value from a file, regardless of its filename extension."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Any, path: Path) -> None:
    """Write JSON through a temporary file to avoid leaving partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def load_postprocessing_prompts(config: PostprocessConfig) -> dict[str, str]:
    """Load and validate the summary and leaf-rewrite prompt templates."""
    if not config.prompt_path.is_file():
        raise FileNotFoundError(
            f"Postprocessing prompt file does not exist: {config.prompt_path}"
        )

    prompt_data = load_json(config.prompt_path)
    if config.dataset not in prompt_data:
        raise KeyError(
            f"Dataset '{config.dataset}' was not found in {config.prompt_path}"
        )

    dataset_prompts = prompt_data[config.dataset]
    if not isinstance(dataset_prompts, dict):
        raise TypeError(
            f"Prompt entry for dataset '{config.dataset}' must be a mapping"
        )

    required_keys = {"summary", "leaf_update"}
    missing_keys = required_keys - set(dataset_prompts)
    if missing_keys:
        raise KeyError(
            "Missing postprocessing prompt key(s): "
            + ", ".join(sorted(missing_keys))
        )

    return {
        "summary": str(dataset_prompts["summary"]),
        "leaf_update": str(dataset_prompts["leaf_update"]),
    }


def request_llm(
    prompt: str,
    llm: Optional[LLM],
    config: PostprocessConfig,
) -> str:
    """Send one prompt through the same backend configuration as stage one."""
    if config.is_vllm:
        if llm is None:
            raise ValueError("A vLLM client is required when models.is_vllm=true")
        response = llm.generate(prompt)
    else:
        response = chatgpt(
            [{"role": "user", "content": prompt}],
            gpt_model=config.gpt_model,
            token_log_path=(
                str(config.token_log_path) if config.token_log_path else None
            ),
        )

    if not isinstance(response, str):
        raise TypeError(f"Expected a string LLM response, received {type(response)}")
    return response.strip()


def count_wiki_nodes(node: Any) -> int:
    """Count dictionary nodes and string leaves in a wiki tree."""
    if isinstance(node, str):
        return 1
    if isinstance(node, dict):
        return 1 + sum(count_wiki_nodes(child) for child in node.values())
    raise TypeError(f"Unexpected wiki node type: {type(node)}")


def split_text_in_half(text: str, tokenizer: Any) -> tuple[str, str]:
    """Split text into two approximately equal token sequences."""
    tokens = tokenizer.tokenize(text)
    midpoint = len(tokens) // 2
    first = tokenizer.convert_tokens_to_string(tokens[:midpoint])
    second = tokenizer.convert_tokens_to_string(tokens[midpoint:])
    return first, second


def summarize_text(
    text: str,
    prompt_template: str,
    tokenizer: Any,
    llm: Optional[LLM],
    config: PostprocessConfig,
) -> str:
    """Summarize text, using the original two-part reduction for long inputs."""
    if not text:
        return ""

    tokens = tokenizer.tokenize(text)
    if len(tokens) <= config.summary_max_input_tokens:
        return request_llm(prompt_template.format(text=text), llm, config)

    first, second = split_text_in_half(text, tokenizer)
    partial_summaries = [
        request_llm(prompt_template.format(text=part), llm, config)
        for part in (first, second)
    ]
    merged_summary = "\n".join(partial_summaries)
    return request_llm(
        prompt_template.format(text=merged_summary),
        llm,
        config,
    )


def generate_wiki_summaries(
    config: PostprocessConfig,
    prompt_template: str,
    tokenizer: Any,
    llm: Optional[LLM],
) -> None:
    """Generate post-order summaries for every node in wiki_ori.jsonl."""
    wiki_data = load_json(config.paths.wiki_ori)
    summaries: dict[str, dict[str, Any]] = {}
    id_counter = 0
    total_nodes = count_wiki_nodes(wiki_data)

    with tqdm(total=total_nodes, desc="Wiki summaries", unit="node") as progress:

        def visit(node: Any, path: list[str]) -> str:
            nonlocal id_counter

            if isinstance(node, str):
                summary = summarize_text(
                    node,
                    prompt_template,
                    tokenizer,
                    llm,
                    config,
                )
            elif isinstance(node, dict):
                child_summaries = [
                    visit(child, path + [title])
                    for title, child in node.items()
                ]
                summary = summarize_text(
                    "\n".join(child_summaries),
                    prompt_template,
                    tokenizer,
                    llm,
                    config,
                )
            else:
                raise TypeError(
                    f"Unexpected wiki node type at path {path}: {type(node)}"
                )

            summaries[str(id_counter)] = {
                "path": path.copy(),
                "summary": summary,
            }
            id_counter += 1
            progress.update(1)
            return summary

        visit(wiki_data, [])

    write_json(summaries, config.paths.summaries)
    LOGGER.info("Saved %d summaries to %s", len(summaries), config.paths.summaries)


def extract_source_citations(text: str) -> set[str]:
    """Extract source indices written at the beginning of wiki lines."""
    citations = set()
    for line in text.splitlines():
        match = re.match(r"^\[(\d+)\]", line.strip())
        if match:
            citations.add(match.group(1))
    return citations


def extract_rewritten_citations(text: str) -> set[str]:
    """Extract angle-bracket citations from rewritten leaf text."""
    return set(re.findall(r"<(\d+)>", text))


def clean_hallucinated_entries(
    text: str,
    original_citations: set[str],
) -> str:
    """Remove generated lines or citation markers unsupported by the source leaf."""
    cleaned_lines = []

    for line in text.splitlines():
        indices = re.findall(r"<(\d+)>", line)
        if not indices:
            continue

        valid_indices = set(indices) & original_citations
        if not valid_indices:
            continue

        if valid_indices != set(indices):

            def replace_invalid(match: re.Match[str]) -> str:
                index = match.group(1)
                return f"<{index}>" if index in valid_indices else ""

            line = re.sub(r"<(\d+)>", replace_invalid, line)
            line = re.sub(r"\s+", " ", line).strip()

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_source_snippet(original_text: str, index: str) -> str:
    """Extract one indexed source snippet, including a final snippet without newline."""
    pattern = re.compile(
        rf"\[{re.escape(index)}\](.*?)(?=(?:\n?\[\d+\])|\Z)",
        flags=re.DOTALL,
    )
    match = pattern.search(original_text)
    return match.group(1).strip() if match else ""


def count_leaf_nodes(node: Any) -> int:
    """Count string leaves in a wiki tree."""
    if isinstance(node, str):
        return 1
    if isinstance(node, dict):
        return sum(count_leaf_nodes(child) for child in node.values())
    raise TypeError(f"Unexpected wiki node type: {type(node)}")


def rewrite_wiki_leaves(
    config: PostprocessConfig,
    prompt_template: str,
    llm: Optional[LLM],
) -> None:
    """Rewrite leaves while preserving source-citation traceability."""
    wiki_data = load_json(config.paths.wiki_ori)
    problematic_nodes = 0
    total_leaves = count_leaf_nodes(wiki_data)

    with tqdm(total=total_leaves, desc="Leaf rewriting", unit="leaf") as progress:

        def visit(node: Any, title_path: list[str]) -> Any:
            nonlocal problematic_nodes

            if isinstance(node, dict):
                return {
                    title: visit(child, title_path + [title])
                    for title, child in node.items()
                }

            if not isinstance(node, str):
                raise TypeError(
                    f"Unexpected wiki node type at path {title_path}: {type(node)}"
                )

            original_citations = extract_source_citations(node)
            if not original_citations:
                progress.update(1)
                return ""

            title = " > ".join(title_path)
            rewritten = ""
            missing = set(original_citations)

            for attempt in range(1, config.leaf_rewrite_retries + 1):
                prompt = prompt_template.format(title=title, text=node)
                rewritten = request_llm(prompt, llm, config)
                rewritten = clean_hallucinated_entries(
                    rewritten,
                    original_citations,
                )
                rewritten_citations = extract_rewritten_citations(rewritten)
                missing = original_citations - rewritten_citations

                if not missing:
                    progress.update(1)
                    return rewritten

                LOGGER.warning(
                    "Missing citations on attempt %d/%d for '%s': %s",
                    attempt,
                    config.leaf_rewrite_retries,
                    title,
                    sorted(missing, key=int),
                )

            fallback_lines = []
            for index in sorted(missing, key=int):
                snippet = extract_source_snippet(node, index)
                fallback_lines.append(f"{snippet} <{index}>")

            problematic_nodes += 1
            progress.update(1)
            fallback_text = "\n".join(fallback_lines)
            if rewritten and fallback_text:
                return f"{rewritten}\n{fallback_text}"
            return rewritten or fallback_text

        updated_wiki = visit(wiki_data, [])

    write_json(updated_wiki, config.paths.updated_wiki)
    LOGGER.info("Saved rewritten wiki to %s", config.paths.updated_wiki)
    LOGGER.info("Leaf nodes requiring citation fallback: %d", problematic_nodes)


def flatten_wiki_leaves(
    node: Any,
    current_path: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Flatten every string leaf into a path/text metadata record."""
    path = current_path or []

    if isinstance(node, str):
        return [{"path": path, "text": node}]
    if not isinstance(node, dict):
        raise TypeError(f"Unexpected wiki node type at path {path}: {type(node)}")

    entries = []
    for title, child in node.items():
        entries.extend(flatten_wiki_leaves(child, path + [title]))
    return entries


def write_leaf_chunks(config: PostprocessConfig) -> None:
    """Extract ud_wiki.jsonl leaves into chunks.json without a FAISS index."""
    wiki_data = load_json(config.paths.updated_wiki)
    entries = flatten_wiki_leaves(wiki_data)
    output = {
        str(index): {
            "path": entry["path"],
            "text": entry["text"],
        }
        for index, entry in enumerate(entries)
    }
    write_json(output, config.paths.chunks)
    LOGGER.info("Saved %d leaf chunks to %s", len(output), config.paths.chunks)


def load_segments(path: Path) -> list[dict[str, Any]]:
    """Load stage-one segments from JSONL."""
    segments = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            segment = json.loads(line)
            if not isinstance(segment, dict):
                raise TypeError(
                    f"Segment at line {line_number} must be a JSON object"
                )
            segments.append(segment)
    return segments


def validate_segment_indices(segments: list[dict[str, Any]]) -> None:
    """Ensure segment IDs align exactly with FAISS row positions."""
    indices = [segment.get("index") for segment in segments]
    expected = list(range(len(segments)))
    if indices != expected:
        raise ValueError(
            "segments.jsonl indices must be contiguous, start at 0, and match "
            "file order so that metadata IDs align with FAISS row IDs."
        )


def build_segment_to_wiki_mapping(
    wiki_chunks: dict[str, dict[str, Any]],
) -> dict[int, list[int]]:
    """Map each source segment index to leaf chunk IDs citing that segment."""
    mapping: dict[int, list[int]] = {}

    for wiki_index_text, chunk in wiki_chunks.items():
        try:
            wiki_index = int(wiki_index_text)
        except ValueError:
            continue

        text = str(chunk.get("text", ""))
        for segment_index_text in re.findall(r"<(\d+)>", text):
            segment_index = int(segment_index_text)
            mapping.setdefault(segment_index, []).append(wiki_index)

    return mapping


def write_faiss_index(index: Any, path: Path) -> None:
    """Write a FAISS index through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    faiss.write_index(index, str(temporary_path))
    os.replace(temporary_path, path)


def build_vector_retrieval_files(
    config: PostprocessConfig,
    embedder: SentenceTransformer,
) -> None:
    """Build v_chunks.json and v_index.faiss from segments and leaf metadata."""
    segments = load_segments(config.paths.segments)
    if not segments:
        raise ValueError(f"No segments were found in {config.paths.segments}")

    validate_segment_indices(segments)
    wiki_chunks = load_json(config.paths.chunks)
    if not isinstance(wiki_chunks, dict):
        raise TypeError("chunks.json must contain a JSON object")

    texts = [str(segment["text"]) for segment in segments]
    embeddings = embedder.encode(
        texts,
        batch_size=config.vector_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if len(embeddings) != len(segments):
        raise ValueError(
            "The number of segment embeddings does not match the segment count"
        )

    vector_index = faiss.IndexFlatIP(embeddings.shape[1])
    vector_index.add(embeddings)

    segment_to_wiki = build_segment_to_wiki_mapping(wiki_chunks)
    vector_chunks: dict[str, dict[str, Any]] = {}

    for segment in segments:
        segment_index = int(segment["index"])
        title = str(segment["title"])
        wiki_indices = sorted(set(segment_to_wiki.get(segment_index, [])))

        filtered_indices = []
        for wiki_index in wiki_indices:
            chunk = wiki_chunks.get(str(wiki_index), {})
            path = chunk.get("path", [])
            if path and path[0] == title:
                filtered_indices.append(wiki_index)

        vector_chunks[str(segment_index)] = {
            "path": [title],
            "text": str(segment["text"]),
            "wiki_indices": filtered_indices,
        }

    write_json(vector_chunks, config.paths.vector_chunks)
    write_faiss_index(vector_index, config.paths.vector_index)
    LOGGER.info("Saved segment metadata to %s", config.paths.vector_chunks)
    LOGGER.info("Saved segment FAISS index to %s", config.paths.vector_index)


def _stage_should_run(output_paths: tuple[Path, ...], overwrite: bool) -> bool:
    if overwrite:
        return True
    return not all(path.exists() for path in output_paths)


def _require_file(path: Path, purpose: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {purpose} file does not exist: {path}")


def run_postprocessing(config: PostprocessConfig) -> None:
    """Run all postprocessing stages in a strict serial order."""
    config.validate()
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = _stage_should_run(
        (config.paths.summaries,),
        config.overwrite_existing,
    )
    run_rewrite = _stage_should_run(
        (config.paths.updated_wiki,),
        config.overwrite_existing,
    )
    run_chunks = run_rewrite or _stage_should_run(
        (config.paths.chunks,),
        config.overwrite_existing,
    )
    run_vectors = run_chunks or _stage_should_run(
        (config.paths.vector_chunks, config.paths.vector_index),
        config.overwrite_existing,
    )

    if run_summary or run_rewrite:
        _require_file(config.paths.wiki_ori, "stage-one wiki")
        prompts = load_postprocessing_prompts(config)
        llm = LLM() if config.is_vllm else None
    else:
        prompts = {}
        llm = None

    if run_summary or run_vectors:
        LOGGER.info(
            "Initializing embedding model/tokenizer from %s on %s",
            config.embedding_path,
            config.embedding_device,
        )
        embedder: Optional[SentenceTransformer] = SentenceTransformer(
            config.embedding_path,
            device=config.embedding_device,
        )
    else:
        embedder = None

    LOGGER.info("[1/4] Generating wiki summaries")
    if run_summary:
        if embedder is None:
            raise RuntimeError("The embedding tokenizer was not initialized")
        generate_wiki_summaries(
            config,
            prompts["summary"],
            embedder.tokenizer,
            llm,
        )
    else:
        LOGGER.info("Skipping summary stage because %s exists", config.paths.summaries)

    LOGGER.info("[2/4] Rewriting wiki leaves")
    if run_rewrite:
        rewrite_wiki_leaves(config, prompts["leaf_update"], llm)
    else:
        LOGGER.info(
            "Skipping leaf rewrite stage because %s exists",
            config.paths.updated_wiki,
        )

    LOGGER.info("[3/4] Extracting leaf chunks")
    if run_chunks:
        _require_file(config.paths.updated_wiki, "rewritten wiki")
        write_leaf_chunks(config)
    else:
        LOGGER.info("Skipping chunk stage because %s exists", config.paths.chunks)

    LOGGER.info("[4/4] Building segment vector retrieval files")
    if run_vectors:
        _require_file(config.paths.segments, "stage-one segments")
        _require_file(config.paths.chunks, "leaf chunks")
        if embedder is None:
            raise RuntimeError("The embedding model was not initialized")
        build_vector_retrieval_files(config, embedder)
    else:
        LOGGER.info(
            "Skipping vector stage because both %s and %s exist",
            config.paths.vector_chunks,
            config.paths.vector_index,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all structured-wiki postprocessing stages."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the shared generation YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    config = load_postprocess_config(args.config)

    start_time = time.time()
    run_postprocessing(config)
    elapsed = time.time() - start_time
    LOGGER.info("Postprocessing completed in %.4f seconds", elapsed)


if __name__ == "__main__":
    main()