"""adtc-profiler CLI entry point.

Usage:
    adtc-profiler run \\
        --submission path/to/submission-dir \\
        --mode {participant,audit} \\
        --output audit.json
        [--skip-accuracy]
        [--seed 42]

The submission directory must contain:
  - metadata.json   submission claims (team_id, domain, language_scope, ...)
  - model file referenced by metadata.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from . import environment as env
from . import accuracy, comparator, gguf, memory, report, reproducibility, thermal, throughput

console = Console()


@click.group()
def main() -> None:
    """ADTC 2026 reference profiler."""


@main.command()
@click.option(
    "--submission",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the submission directory (must contain metadata.json).",
)
@click.option(
    "--mode",
    type=click.Choice(["participant", "audit"]),
    required=True,
    help="participant: emits submission.json (Gate 1); audit: emits audit.json (Gate 2).",
)
@click.option(
    "--output", "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Where to write the JSON report.",
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--skip-accuracy",
    is_flag=True,
    help="Skip lm_eval benchmark; emit empty accuracy: []. Useful for fast smoke tests.",
)
@click.option(
    "--accuracy-task",
    default="arc_easy",
    show_default=True,
    help="lm_eval task name. Real audits use the hidden validation subset.",
)
@click.option(
    "--accuracy-limit",
    type=int,
    default=50,
    show_default=True,
    help="Max samples for the accuracy benchmark.",
)
def run(
    submission: Path,
    mode: str,
    output_path: Path,
    seed: int,
    skip_accuracy: bool,
    accuracy_task: str,
    accuracy_limit: int,
) -> None:
    """Run the full profiler pipeline and emit a schema-valid JSON report."""
    metadata_path = submission / "metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]submission missing metadata.json at {metadata_path}[/red]")
        sys.exit(2)

    submission_meta = json.loads(metadata_path.read_text())
    runtime_meta = submission_meta.get("_runtime", {})
    model_path = submission / runtime_meta.get("model_path", "model.gguf")
    if not model_path.exists():
        console.print(
            f"[red]model file not found at {model_path}.[/red] "
            f"Run the submission's download_model.sh first."
        )
        sys.exit(2)

    # Schema is strict (additionalProperties: false). Strip the operational
    # `_runtime` block before assembling the submission section of the report.
    submission_block = {k: v for k, v in submission_meta.items() if not k.startswith("_")}

    measured_on = "audit_cloud_vm" if mode == "audit" else "participant_laptop"
    console.print(f"[bold]adtc-profiler[/bold] mode={mode} model={model_path.name}")

    # Memory + thermal samplers wrap the throughput run so they capture the
    # peak/steady-state under load. Accuracy runs separately afterward.
    with memory.sample_during() as mem_sampler, thermal.sample_thermal() as therm_sampler:
        console.print("→ running llama-bench (throughput)…")
        throughput_block = throughput.measure(model_path, seed=seed)

    memory_block = mem_sampler.report()
    thermal_block = therm_sampler.report()

    if skip_accuracy:
        console.print("→ skipping accuracy (--skip-accuracy)")
        accuracy_block: list[dict] = []
    elif not accuracy.is_available():
        console.print(
            "[yellow]lm_eval not installed — emitting empty accuracy. "
            "Reinstall the profiler to get the accuracy stack.[/yellow]"
        )
        accuracy_block = []
    else:
        console.print(f"→ running lm_eval task={accuracy_task} limit={accuracy_limit}…")
        accuracy_block = [
            accuracy.run_benchmark(
                model_path, task=accuracy_task, limit=accuracy_limit, seed=seed
            )
        ]

    environment_block = env.capture(measured_on)
    reproducibility_block = reproducibility.capture(
        seed=seed,
        repo_path=submission,
        docker_image=runtime_meta.get("docker_image"),
    )

    # Extract GGUF header metadata for fraud detection and run display.
    gguf_meta = gguf.extract_metadata(model_path)
    claimed_estimate = submission_meta.get("model", {}).get("parameters_estimate", "")
    model_info_block = {
        "params_count": gguf_meta.get("params_count"),
        "context_length": gguf_meta.get("context_length"),
        "architecture": gguf_meta.get("architecture"),
        "claimed_params_estimate": claimed_estimate,
        "params_match": gguf.fraud_check(claimed_estimate, gguf_meta.get("params_count")),
    }

    output = report.assemble(
        submission=submission_block,
        environment=environment_block,
        throughput=throughput_block,
        memory=memory_block,
        accuracy=accuracy_block,
        cpu_thermal=thermal_block,
        reproducibility=reproducibility_block,
        model_info=model_info_block,
    )

    try:
        report.write(output, str(output_path))
    except report.SchemaValidationError as e:
        console.print(f"[red]schema validation failed:[/red] {e}")
        sys.exit(3)

    console.print(f"[green]✓[/green] wrote {output_path}")


@main.command(name="compare")
@click.argument(
    "submission_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "audit_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output", "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the full verdict JSON here. Stdout shows a summary table.",
)
@click.option(
    "--no-strict",
    is_flag=True,
    help="Skip schema validation of the inputs (useful when diagnosing a broken audit).",
)
def compare(submission_path: Path, audit_path: Path, output_path: Path | None, no_strict: bool) -> None:
    """Diff a participant submission.json against an audit.json.

    Applies the spec §6 tolerances (±15% memory, ±25% throughput) and emits a
    pass/flag/fail verdict.
    """
    verdict = comparator.compare_files(submission_path, audit_path, strict=not no_strict)

    color = {"pass": "green", "flag": "yellow", "fail": "red"}.get(verdict.verdict, "white")
    console.print(f"[bold]team:[/bold] {verdict.team_id}")
    console.print(f"[bold]verdict:[/bold] [{color}]{verdict.verdict.upper()}[/{color}]\n")

    console.print("[bold]checks:[/bold]")
    for check in verdict.checks:
        status_color = {
            "pass": "green",
            "flag": "yellow",
            "fail": "red",
            "missing": "dim",
        }.get(check.status, "white")
        delta_str = f"{check.delta_pct:+.1f}%" if check.delta_pct is not None else "    -"
        console.print(
            f"  [{status_color}]{check.status:>7}[/{status_color}] "
            f"{check.field:<48} "
            f"sub={check.submission}  aud={check.audit}  Δ={delta_str}  tol=±{check.tolerance_pct}%"
        )

    if verdict.notes:
        console.print("\n[bold]notes:[/bold]")
        for n in verdict.notes:
            console.print(f"  - {n}")

    if output_path:
        output_path.write_text(json.dumps(verdict.to_dict(), indent=2, ensure_ascii=False) + "\n")
        console.print(f"\n[green]✓[/green] wrote {output_path}")

    if verdict.verdict == "fail":
        sys.exit(1)
    if verdict.verdict == "flag":
        sys.exit(2)


if __name__ == "__main__":
    main()
