"""Tests for the comparator: tolerances, status ladder, structural failures."""
from __future__ import annotations

import copy
import json
from pathlib import Path


from adtc_profiler import comparator


SAMPLE_PATH = Path(__file__).parent / "sample-submission.json"


def _load_sample() -> dict:
    return json.loads(SAMPLE_PATH.read_text())


def _audit_from(submission: dict, **adjustments) -> dict:
    """Build an audit.json by deep-copying submission and applying scaled deltas.

    adjustments are dotted paths -> multiplier (e.g. {"throughput.tokens_per_second_generation": 0.80}).
    """
    audit = copy.deepcopy(submission)
    audit["environment"]["measured_on"] = "audit_cloud_vm"
    for path, multiplier in adjustments.items():
        parts = path.split(".")
        cur = audit
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = cur[parts[-1]] * multiplier
    return audit


class TestCompareReports:
    def test_identical_reports_pass(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub)
        v = comparator.compare_reports(sub, audit)
        assert v.verdict == "pass", v.notes
        assert all(c.status in ("pass", "missing") for c in v.checks)

    def test_throughput_within_25pct_passes(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub, **{"throughput.tokens_per_second_generation": 0.80})
        v = comparator.compare_reports(sub, audit)
        throughput_check = next(
            c for c in v.checks if c.field == "throughput.tokens_per_second_generation"
        )
        assert throughput_check.status == "pass"
        assert v.verdict == "pass"

    def test_throughput_beyond_25pct_flags(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub, **{"throughput.tokens_per_second_generation": 0.65})
        v = comparator.compare_reports(sub, audit)
        throughput_check = next(
            c for c in v.checks if c.field == "throughput.tokens_per_second_generation"
        )
        assert throughput_check.status == "flag"
        assert v.verdict == "flag"

    def test_throughput_beyond_50pct_fails(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub, **{"throughput.tokens_per_second_generation": 0.40})
        v = comparator.compare_reports(sub, audit)
        throughput_check = next(
            c for c in v.checks if c.field == "throughput.tokens_per_second_generation"
        )
        assert throughput_check.status == "fail"
        assert v.verdict == "fail"

    def test_memory_within_15pct_passes(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub, **{"memory.peak_rss_mb": 1.12})
        v = comparator.compare_reports(sub, audit)
        memory_check = next(c for c in v.checks if c.field == "memory.peak_rss_mb")
        assert memory_check.status == "pass"

    def test_memory_beyond_15pct_flags(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub, **{"memory.peak_rss_mb": 1.20})
        v = comparator.compare_reports(sub, audit)
        memory_check = next(c for c in v.checks if c.field == "memory.peak_rss_mb")
        assert memory_check.status == "flag"

    def test_memory_30_to_50pct_band_flags_not_fails(self) -> None:
        """The documented rule: exceeding tolerance flags for judge review;
        only >50% hard-fails. A 2x-tolerance ladder would wrongly auto-fail
        memory deltas in the 30-50% band."""
        sub = _load_sample()
        audit = _audit_from(sub, **{"memory.peak_rss_mb": 1.40})
        v = comparator.compare_reports(sub, audit)
        memory_check = next(c for c in v.checks if c.field == "memory.peak_rss_mb")
        assert memory_check.status == "flag"
        assert v.verdict == "flag"

    def test_memory_beyond_50pct_fails(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub, **{"memory.peak_rss_mb": 1.60})
        v = comparator.compare_reports(sub, audit)
        memory_check = next(c for c in v.checks if c.field == "memory.peak_rss_mb")
        assert memory_check.status == "fail"
        assert v.verdict == "fail"

    def test_team_id_mismatch_demotes_to_fail(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub)
        audit["submission"]["team_id"] = "different-team"
        v = comparator.compare_reports(sub, audit)
        assert v.verdict == "fail"
        assert any("team_id mismatch" in n for n in v.notes)

    def test_schema_invalid_audit_demotes_to_fail(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub)
        del audit["throughput"]
        v = comparator.compare_reports(sub, audit, strict=True)
        assert v.verdict == "fail"
        assert any("schema invalid" in n for n in v.notes)

    def test_accuracy_passed_through(self) -> None:
        sub = _load_sample()
        audit = _audit_from(sub)
        audit["accuracy"] = [
            {
                "benchmark": "hidden-medqa",
                "dataset_version": "hidden-v1",
                "language": "en",
                "samples": 100,
                "score": 0.61,
                "metric": "accuracy",
            }
        ]
        v = comparator.compare_reports(sub, audit)
        assert v.accuracy_audit == audit["accuracy"]

    def test_wrong_audit_environment_is_structural_fail(self) -> None:
        """A participant-produced file supplied as the audit report (identical
        deltas, perfect pass otherwise) must FAIL, not slip through as a note —
        exit-code-driven orchestrators only see the verdict."""
        sub = _load_sample()
        audit = _audit_from(sub)
        audit["environment"]["measured_on"] = "participant_laptop"
        v = comparator.compare_reports(sub, audit)
        assert any("audit environment.measured_on" in n for n in v.notes)
        assert v.verdict == "fail"


def test_zero_submission_value_is_missing_not_div_by_zero() -> None:
    sub = {
        "schema_version": "1.0.0",
        "profiler_version": "test",
        "submission": {"team_id": "x"},
        "environment": {"measured_on": "participant_laptop"},
        "throughput": {"tokens_per_second_generation": 0, "first_token_latency_ms": 100},
        "memory": {"peak_rss_mb": 100, "steady_state_rss_mb": 100},
        "accuracy": [],
        "cpu_thermal": {"cpu_percent_p99": 50, "throttled": False},
        "reproducibility": {"git_commit_sha": "abc1234", "docker_image_digest": "x", "random_seed": 0},
    }
    audit = copy.deepcopy(sub)
    audit["environment"]["measured_on"] = "audit_cloud_vm"
    audit["throughput"]["tokens_per_second_generation"] = 10
    audit["submission"] = sub["submission"]
    v = comparator.compare_reports(sub, audit, strict=False)
    tg_check = next(c for c in v.checks if c.field == "throughput.tokens_per_second_generation")
    # A zero claim cannot be verified and would maximize the efficiency
    # score — it must hard-fail, not cap at flag.
    assert tg_check.status == "fail"
    assert v.verdict == "fail"


def test_non_numeric_value_fails_cleanly_not_crash() -> None:
    sub = {
        "schema_version": "1.0.0",
        "profiler_version": "test",
        "submission": {"team_id": "x"},
        "environment": {"measured_on": "participant_laptop"},
        "throughput": {"tokens_per_second_generation": "fast", "first_token_latency_ms": 100},
        "memory": {"peak_rss_mb": 100, "steady_state_rss_mb": 100},
        "accuracy": [],
        "cpu_thermal": {"cpu_percent_p99": 50, "throttled": False},
        "reproducibility": {"git_commit_sha": "abc1234", "docker_image_digest": "x", "random_seed": 0},
    }
    audit = copy.deepcopy(sub)
    audit["environment"]["measured_on"] = "audit_cloud_vm"
    audit["throughput"]["tokens_per_second_generation"] = 10.0
    v = comparator.compare_reports(sub, audit, strict=False)
    tg_check = next(c for c in v.checks if c.field == "throughput.tokens_per_second_generation")
    assert tg_check.status == "missing"
    assert v.verdict == "fail"
    assert any("not numeric" in n for n in v.notes)
