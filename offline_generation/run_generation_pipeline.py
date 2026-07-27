from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

STAGE_ONE_SCRIPT = "gen_knowledge_base.py"
POSTPROCESS_SCRIPT = "postprocess.py"

STAGE_ONE_OUTPUTS = (
    "wiki_ori.jsonl",
    "segments.jsonl",
)

POSTPROCESS_OUTPUTS = (
    "sum.json",
    "ud_wiki.jsonl",
    "chunks.json",
    "v_chunks.json",
    "v_index.faiss",
)


def _require_mapping(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config.get(section_name, {})
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


def load_pipeline_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load the shared YAML file and validate cross-stage requirements."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to run the generation pipeline. "
            "Install it with: pip install PyYAML"
        ) from exc

    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {resolved_config_path}"
        )

    with resolved_config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if raw_config is None:
        raise ValueError("The configuration file is empty")
    if not isinstance(raw_config, dict):
        raise TypeError("The top level of the configuration file must be a mapping")

    paths = _require_mapping(raw_config, "paths")
    segmentation = _require_mapping(raw_config, "segmentation")
    output = _require_mapping(raw_config, "output")
    postprocessing = _require_mapping(raw_config, "postprocessing")

    if not paths.get("output_dir"):
        raise ValueError("paths.output_dir must be provided")
    if not postprocessing.get("dataset"):
        raise ValueError("postprocessing.dataset must be provided")
    if not postprocessing.get("prompt_path"):
        raise ValueError("postprocessing.prompt_path must be provided")

    if not bool(segmentation.get("save_segments", False)):
        raise ValueError(
            "segmentation.save_segments must be true for the complete pipeline, "
            "because postprocessing requires segments.jsonl"
        )

    if not bool(output.get("group_output_by_title", True)):
        raise ValueError(
            "output.group_output_by_title must be true because the current "
            "postprocessing mapping matches wiki root titles to segment titles"
        )

    return raw_config, resolved_config_path


def create_stage_one_config(
    raw_config: dict[str, Any],
    original_config_path: Path,
) -> Path:
    """
    Create a temporary stage-one view of the shared config.

    The current stage-one loader performs strict top-level validation. Removing
    postprocessing-only sections lets both stages share one user-facing config
    without changing the tested stage-one generation logic.
    """
    import yaml

    stage_one_config = dict(raw_config)
    stage_one_config.pop("postprocessing", None)
    stage_one_config.pop("pipeline", None)

    temporary_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".stage_one.yaml",
        prefix=".generation_pipeline_",
        dir=original_config_path.parent,
        delete=False,
    )
    try:
        yaml.safe_dump(
            stage_one_config,
            temporary_file,
            allow_unicode=True,
            sort_keys=False,
        )
        return Path(temporary_file.name)
    finally:
        temporary_file.close()


def run_stage(
    script_path: Path,
    config_path: Path,
    stage_name: str,
) -> None:
    """Run one pipeline stage as a blocking child process."""
    if not script_path.is_file():
        raise FileNotFoundError(f"{stage_name} script does not exist: {script_path}")

    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    LOGGER.info("Starting %s", stage_name)
    LOGGER.info("Command: %s", " ".join(command))

    completed = subprocess.run(
        command,
        cwd=script_path.parent,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage_name} failed with exit code {completed.returncode}"
        )

    LOGGER.info("Completed %s", stage_name)


def require_outputs(
    output_dir: Path,
    filenames: tuple[str, ...],
    stage_name: str,
) -> None:
    """Verify that a completed stage produced all expected artifacts."""
    missing = [str(output_dir / name) for name in filenames if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{stage_name} completed but did not produce the expected file(s): "
            + ", ".join(missing)
        )


def run_pipeline(config_path: str | Path) -> None:
    """Run structured-wiki generation followed by all postprocessing stages."""
    raw_config, resolved_config_path = load_pipeline_config(config_path)
    config_dir = resolved_config_path.parent
    paths = _require_mapping(raw_config, "paths")
    output_dir = _resolve_config_path(str(paths["output_dir"]), config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    stage_one_script = script_dir / STAGE_ONE_SCRIPT
    postprocess_script = script_dir / POSTPROCESS_SCRIPT

    stage_one_config_path = create_stage_one_config(
        raw_config,
        resolved_config_path,
    )

    started_at = time.time()
    try:
        run_stage(
            stage_one_script,
            stage_one_config_path,
            "stage one: structured wiki generation",
        )
        require_outputs(output_dir, STAGE_ONE_OUTPUTS, "Stage one")

        run_stage(
            postprocess_script,
            resolved_config_path,
            "stage two: wiki postprocessing",
        )
        require_outputs(output_dir, POSTPROCESS_OUTPUTS, "Stage two")
    finally:
        stage_one_config_path.unlink(missing_ok=True)

    elapsed = time.time() - started_at
    LOGGER.info("Full generation pipeline completed in %.4f seconds", elapsed)
    LOGGER.info("Artifacts are available in %s", output_dir)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run structured-wiki generation and postprocessing with one shared "
            "YAML configuration file."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the shared generation YAML configuration file.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_argument_parser().parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()