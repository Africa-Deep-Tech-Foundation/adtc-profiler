"""Assemble + schema-validate the final profiler report.

Fail-fast on schema validation error: the schema is the contract, and a
broken report at participant time is far cheaper to catch than at audit time.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any

import jsonschema

from . import PROFILER_VERSION, SCHEMA_VERSION


class SchemaValidationError(RuntimeError):
    """Assembled report failed schema validation."""


def load_schema() -> dict:
    """Load the bundled adtc-profiler schema."""
    with resources.files("adtc_profiler.schema").joinpath(
        "adtc-profiler.schema.json"
    ).open("r") as f:
        return json.load(f)


def assemble(
    *,
    submission: dict,
    environment: dict,
    throughput: dict,
    memory: dict,
    accuracy: list[dict],
    cpu_thermal: dict,
    reproducibility: dict,
    model_info: dict | None = None,
) -> dict[str, Any]:
    """Build the report dict in schema order."""
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profiler_version": PROFILER_VERSION,
        "submission": submission,
        "environment": environment,
        "throughput": throughput,
        "memory": memory,
        "accuracy": accuracy,
        "cpu_thermal": cpu_thermal,
        "reproducibility": reproducibility,
    }
    if model_info is not None:
        out["model_info"] = model_info
    return out


def validate_submission_block(submission_block: dict) -> None:
    """Validate just the `submission` section against its subschema.

    Lets the CLI reject broken metadata BEFORE the (minutes-long) benchmark
    run instead of failing at report-write time.
    """
    subschema = load_schema()["properties"]["submission"]
    try:
        jsonschema.validate(submission_block, subschema)
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "(root)"
        raise SchemaValidationError(
            f"submission metadata invalid at {path}: {e.message}"
        ) from e


def validate(report: dict) -> None:
    """Raise SchemaValidationError if `report` does not match the schema."""
    schema = load_schema()
    try:
        jsonschema.validate(report, schema)
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "(root)"
        raise SchemaValidationError(
            f"report failed schema validation at {path}: {e.message}"
        ) from e


def write(report: dict, output_path: str) -> None:
    """Validate, then write JSON to `output_path` (UTF-8, 2-space indent)."""
    validate(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
