"""Compare a participant submission.json against an audit.json and emit a verdict.

Tolerances (per spec §6):
  - memory.peak_rss_mb           ±15%
  - memory.steady_state_rss_mb   ±15%
  - throughput.tokens_per_second_generation  ±25%
  - throughput.first_token_latency_ms        ±25% (by analogy; not explicit in spec)

Verdict ladder:
  - pass  — every check inside its tolerance
  - flag  — one or more checks outside tolerance but within the 50% fail bound
            (routed to manual judge review)
  - fail  — at least one |delta| > 50%, or structural issue (missing or
            zero-valued fields, schema-invalid input, team_id mismatch,
            wrong measured_on environment)

Accuracy comparison is NOT a delta-vs-claim check: participant accuracy is on
public benchmarks; audit accuracy is on the hidden 30% subset. The comparator
passes audit accuracy through as-is for judge review rather than diffing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from . import report


# Tolerance percentages per field path (dotted path into the report).
TOLERANCES: dict[str, float] = {
    "memory.peak_rss_mb": 15.0,
    "memory.steady_state_rss_mb": 15.0,
    "throughput.tokens_per_second_generation": 25.0,
    "throughput.first_token_latency_ms": 25.0,
}


@dataclass
class Check:
    field: str
    submission: float | None
    audit: float | None
    delta_pct: float | None
    tolerance_pct: float
    status: str  # "pass" | "flag" | "fail" | "missing"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    team_id: str
    verdict: str  # "pass" | "flag" | "fail"
    checks: list[Check]
    accuracy_audit: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "verdict": self.verdict,
            "checks": [c.to_dict() for c in self.checks],
            "accuracy_audit": self.accuracy_audit,
            "notes": self.notes,
        }


def _dotted_get(d: dict, path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# Documented rule (README + spec): exceeding tolerance flags for manual
# review; only deltas beyond 50% hard-fail. A fixed bound — NOT 2x tolerance,
# which would auto-fail memory deltas in the 30-50% band the rule routes
# to judges.
FAIL_PCT = 50.0


def _classify_delta(delta_pct: float, tolerance_pct: float) -> str:
    """pass / flag / fail based on |delta|."""
    magnitude = abs(delta_pct)
    if magnitude <= tolerance_pct:
        return "pass"
    if magnitude <= FAIL_PCT:
        return "flag"
    return "fail"


def compare_reports(submission: dict, audit: dict, *, strict: bool = True) -> Verdict:
    """Compare two profiler reports and emit a verdict.

    If `strict`, both inputs are schema-validated first. Set to False when
    diagnosing a known-broken audit (e.g. inspecting what the audit produced).
    """
    notes: list[str] = []
    checks: list[Check] = []

    if strict:
        try:
            report.validate(submission)
        except report.SchemaValidationError as e:
            notes.append(f"submission schema invalid: {e}")
        try:
            report.validate(audit)
        except report.SchemaValidationError as e:
            notes.append(f"audit schema invalid: {e}")

    sub_team = submission.get("submission", {}).get("team_id", "<missing>")
    aud_team = audit.get("submission", {}).get("team_id", "<missing>")
    if sub_team != aud_team:
        notes.append(
            f"team_id mismatch: submission='{sub_team}' audit='{aud_team}'"
        )

    sub_env = submission.get("environment", {}).get("measured_on")
    aud_env = audit.get("environment", {}).get("measured_on")
    if sub_env != "participant_laptop":
        notes.append(f"submission environment.measured_on='{sub_env}' (expected participant_laptop)")
    if aud_env != "audit_cloud_vm":
        notes.append(f"audit environment.measured_on='{aud_env}' (expected audit_cloud_vm)")

    for field, tolerance_pct in TOLERANCES.items():
        raw_sub = _dotted_get(submission, field)
        raw_aud = _dotted_get(audit, field)
        sub_val = raw_sub if isinstance(raw_sub, (int, float)) and not isinstance(raw_sub, bool) else None
        aud_val = raw_aud if isinstance(raw_aud, (int, float)) and not isinstance(raw_aud, bool) else None
        if sub_val is None or aud_val is None:
            # Absent or non-numeric: the comparison contract is broken — a
            # doctored or corrupt report must not do better than a bad delta.
            if raw_sub is not None and sub_val is None:
                notes.append(f"{field}: submission value is not numeric ({raw_sub!r})")
            if raw_aud is not None and aud_val is None:
                notes.append(f"{field}: audit value is not numeric ({raw_aud!r})")
            checks.append(Check(
                field=field,
                submission=sub_val,
                audit=aud_val,
                delta_pct=None,
                tolerance_pct=tolerance_pct,
                status="missing",
            ))
            continue
        if sub_val == 0:
            # An unverifiable zero claim maximizes the efficiency score —
            # treating it leniently would reward exactly the wrong behavior.
            checks.append(Check(
                field=field,
                submission=float(sub_val),
                audit=float(aud_val),
                delta_pct=None,
                tolerance_pct=tolerance_pct,
                status="fail",
            ))
            notes.append(f"{field}: submission value is 0, cannot verify claim")
            continue
        delta_pct = (float(aud_val) - float(sub_val)) / float(sub_val) * 100.0
        checks.append(Check(
            field=field,
            submission=float(sub_val),
            audit=float(aud_val),
            delta_pct=round(delta_pct, 2),
            tolerance_pct=tolerance_pct,
            status=_classify_delta(delta_pct, tolerance_pct),
        ))

    # Overall verdict = worst-of any check, with structural issues
    # (schema-invalid, team_id mismatch, wrong environment) demoting to fail.
    structural_fail = any(
        "schema invalid" in n
        or "team_id mismatch" in n
        or "environment.measured_on" in n
        for n in notes
    )
    statuses = {c.status for c in checks}
    if structural_fail or "fail" in statuses or "missing" in statuses:
        overall = "fail"
    elif "flag" in statuses:
        overall = "flag"
    else:
        overall = "pass"

    return Verdict(
        team_id=aud_team,
        verdict=overall,
        checks=checks,
        accuracy_audit=audit.get("accuracy", []),
        notes=notes,
    )


def compare_files(submission_path: Path, audit_path: Path, *, strict: bool = True) -> Verdict:
    submission = json.loads(submission_path.read_text())
    audit = json.loads(audit_path.read_text())
    return compare_reports(submission, audit, strict=strict)
