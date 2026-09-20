"""Round-trip schema validation against the canonical sample-submission.json
that lives in .jarvis/context/private/julian/adtf/hackathon-2026/.

If this test breaks, the schema bundled in the package has drifted from the
canonical schema in jarvis context, OR the sample is wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from adtc_profiler import report

SAMPLE_PATH = Path(__file__).parent / "sample-submission.json"


def test_sample_submission_validates() -> None:
    sample = json.loads(SAMPLE_PATH.read_text())
    report.validate(sample)


def test_bundled_schema_loads() -> None:
    schema = report.load_schema()
    assert schema["$id"] == "https://adtc.africa/schemas/adtc-profiler.schema.json"
    assert "submission" in schema["required"]


def test_assemble_minimal_valid_report() -> None:
    """Build a minimum-shape report from stub data and assert it validates.

    Catches regressions where a field name is misspelled, the assemble() helper
    drifts from the schema, or required fields go missing.
    """
    submission = {
        "team_id": "test-team",
        "domain": "coding_assistants",
        "language_scope": ["en"],
        "african_alpha_claim": False,
        "budget_laptop_claim": True,
        "submitter": {
            "name": "Efe Mensah",
            "email": "efe@deeptech.africa",
            "github_handle": "efemensah",
        },
        "cross_disciplinary_pairing": {
            "discipline": "test",
            "load_bearing": False,
            "description": "test fixture",
        },
        "test_prompts": [
            {"prompt_id": "tp_001", "prompt": "stub 1"},
            {"prompt_id": "tp_002", "prompt": "stub 2"},
        ],
        "model": {
            "name": "stub",
            "runtime": "llama.cpp",
            "quantization": "Q4_K_M",
            "parameters_estimate": "1B",
            "packaging": "docker_image",
        },
    }
    environment = {
        "measured_on": "participant_laptop",
        "cpu_model": "stub-cpu",
        "ram_gb": 8.0,
        "gpu": "none",
        "os": "Linux 5.15",
    }
    throughput = {
        "tokens_per_second_generation": 15.5,
        "first_token_latency_ms": 500.0,
        "prompt_tokens": 512,
        "generated_tokens": 128,
    }
    memory = {"peak_rss_mb": 2048.0, "steady_state_rss_mb": 1800.0, "peak_vms_mb": 3000.0}
    accuracy: list[dict] = []
    cpu_thermal = {"cpu_percent_p99": 92.0, "core_temp_c_peak": None, "throttled": False}
    reproducibility = {
        "git_commit_sha": "0123456789ab",
        "docker_image_digest": "sha256:abc123",
        "random_seed": 42,
    }
    r = report.assemble(
        submission=submission,
        environment=environment,
        throughput=throughput,
        memory=memory,
        accuracy=accuracy,
        cpu_thermal=cpu_thermal,
        reproducibility=reproducibility,
    )
    report.validate(r)


def _minimal_model_block() -> dict:
    return {
        "name": "stub",
        "runtime": "llama.cpp",
        "quantization": "Q4_K_M",
        "parameters_estimate": "1B",
        "packaging": "docker_image",
    }


def test_base_model_commit_sha_is_accepted_when_present() -> None:
    """Gate 2 asks teams to add base_model_commit_sha to metadata.json for provenance.
    Regression guard for the incident where adding this field to a team's real metadata.json
    tripped additionalProperties:false and aborted the profiler run before any benchmark —
    the field must validate, not reject the run outright."""
    submission_block = {
        "team_id": "test-team",
        "domain": "coding_assistants",
        "language_scope": ["en"],
        "african_alpha_claim": False,
        "budget_laptop_claim": True,
        "submitter": {
            "name": "Efe Mensah",
            "email": "efe@deeptech.africa",
            "github_handle": "efemensah",
        },
        "cross_disciplinary_pairing": {
            "discipline": "test",
            "load_bearing": False,
            "description": "test fixture",
        },
        "test_prompts": [
            {"prompt_id": "tp_001", "prompt": "stub 1"},
            {"prompt_id": "tp_002", "prompt": "stub 2"},
        ],
        "model": {
            **_minimal_model_block(),
            "base_model_commit_sha": "3fb3c9d4b0e6",
        },
    }
    report.validate_submission_block(submission_block)


def test_base_model_commit_sha_stays_optional() -> None:
    """A submission that omits base_model_commit_sha entirely must still validate — it is a
    Gate 2 checklist item, not an automated gate, so its absence must never block a run."""
    submission_block = {
        "team_id": "test-team",
        "domain": "coding_assistants",
        "language_scope": ["en"],
        "african_alpha_claim": False,
        "budget_laptop_claim": True,
        "submitter": {
            "name": "Efe Mensah",
            "email": "efe@deeptech.africa",
            "github_handle": "efemensah",
        },
        "cross_disciplinary_pairing": {
            "discipline": "test",
            "load_bearing": False,
            "description": "test fixture",
        },
        "test_prompts": [
            {"prompt_id": "tp_001", "prompt": "stub 1"},
            {"prompt_id": "tp_002", "prompt": "stub 2"},
        ],
        "model": _minimal_model_block(),
    }
    report.validate_submission_block(submission_block)


def test_missing_required_field_raises() -> None:
    """A report missing a required sub-field must raise SchemaValidationError."""
    bogus = {
        "schema_version": "1.0.0",
        "profiler_version": "test",
        "submission": {"team_id": "x"},  # missing required submission sub-fields
        "environment": {},
        "throughput": {},
        "memory": {},
        "accuracy": [],
        "cpu_thermal": {},
        "reproducibility": {},
    }
    with pytest.raises(report.SchemaValidationError):
        report.validate(bogus)
