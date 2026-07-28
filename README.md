# ADTC Profiler

Reference profiler CLI for the **Africa Deep Tech Challenge (ADTC) 2026** Laptop LLM track.

The tool measures local GGUF models running through `llama.cpp` and emits schema-valid JSON benchmark reports capturing throughput, memory footprint, CPU utilization, thermals, and optional accuracy metrics.

---

## 📥 Installation

Install the profiler directly from GitHub using `pip` or `uv`:

```bash
python3 -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
```

> Use `python3 -m pip` (or `pip3`) — macOS and most Linux distros do not ship a bare `pip` command.

To also run the accuracy benchmarks (lm-eval + torch + transformers — a large download), install the `accuracy` extra:

```bash
python3 -m pip install "adtc-profiler[accuracy] @ git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
```

### System Prerequisites
To profile model executions correctly, the tool relies on native binaries:
1. **`llama-bench`**: Must be installed and available on your system `PATH`. This is part of the `llama.cpp` toolset.
2. **Python >= 3.11**

---

## 🧪 Usage

The profiler runs in two primary modes:

### 1. Participant Mode (Local Self-Check)
Used by participants to smoke-test their submission locally. It runs the throughput bench and resource sampling, but skips accuracy evaluations.

```bash
adtc-profiler run \
  --submission /path/to/your-submission-repo \
  --mode participant \
  --output submission.json \
  --skip-accuracy
```

### 2. Audit Mode (Evaluation Sandbox)
Used by the ADTC evaluation orchestrator inside secure cloud VMs. 

```bash
adtc-profiler run \
  --submission /path/to/your-submission-repo \
  --mode audit \
  --output audit.json
```

---

## ⚖️ Comparing Reports

After running both the local self-check and the audit evaluations, you can compare the output JSON files to check if they conform to the competition's variance tolerances:

```bash
adtc-profiler compare submission.json audit.json --output verdict.json
```

### Tolerance Guidelines

To ensure fairness across differing environments, the comparison engine tolerates minor variances:

| Metric | Tolerance | Status |
| :--- | :--- | :--- |
| `memory.peak_rss_mb` | ±15% | Flags if exceeded; fails if >50% |
| `memory.steady_state_rss_mb` | ±15% | Flags if exceeded; fails if >50% |
| `throughput.tokens_per_second_generation` | ±25% | Flags if exceeded; fails if >50% |
| `throughput.first_token_latency_ms` | ±25% | Flags if exceeded; fails if >50% |

The comparison command yields one of three verdicts:
- **`pass`**: Model telemetry matches within tolerance limits.
- **`flag`**: Noticeable variance observed; marked for manual judge review.
- **`fail`**: Out-of-bounds telemetry, mismatched team IDs, or schema violations.

---

## 📊 Leaderboard Scoring

Submissions are scored using a weighted formula combining accuracy, throughput, and memory efficiency, with penalties for thermal throttling:

$$S_{\text{total}} = 0.50 \cdot S_{\text{acc}} + 0.30 \cdot S_{\text{perf}} + 0.20 \cdot S_{\text{eff}} - P_{\text{thermal}}$$

| Component | Formula / Rule | Details |
| :--- | :--- | :--- |
| **$S_{\text{acc}}$ (Accuracy)** | Qualifying score | Based on model responses to participant-submitted prompts, domain prompts, and hidden prompts supplied by judges. |
| **$S_{\text{perf}}$ (Throughput)** | `min(TPS / TPS_REFERENCE, 1.0) * 100` | Normalised against `TPS_REFERENCE = 15.0`. |
| **$S_{\text{eff}}$ (Efficiency)** | `max(0, (RAM_LIMIT_GB - peak_rss_gb) / RAM_LIMIT_GB) * 100` | Normalised against `RAM_LIMIT_GB = 7.0` (8 GB target profile). |
| **$P_{\text{thermal}}$ (Penalty)** | `10` points deduction | Applied if the CPU throttles or core temp exceeds 85°C. |

---

## 🐳 Docker Execution

To run the profiler in a fully isolated container (recommended for reproducible auditing):

### 1. Build the Image
```bash
docker build -t adtc-profiler:latest .
```

### 2. Run Audit
```bash
docker run --rm --memory=7.5g \
  -v "/path/to/submission:/submission:ro" \
  -v "/path/to/artifacts:/artifacts" \
  adtc-profiler:latest run \
  --submission /submission \
  --mode audit \
  --output /artifacts/audit.json \
  --skip-accuracy
```

---

## 🛠️ Local Development

Clone the repository and set up your environment using `uv`:

```bash
# Sync dependencies and build virtual env
uv sync --all-extras

# Run unit tests
uv run pytest

# Run linter and formatting checks
uv run ruff check src tests
```

---

## 🧹 Uninstallation & Clean Up

To uninstall the profiler package and clean up all generated reports or caches:

```bash
# 1. Uninstall the package
pip uninstall -y adtc-profiler

# 2. Delete generated JSON reports
rm -f submission.json audit.json verdict.json

# 3. Clean local development caches (if cloned)
rm -rf .venv/ .uv-cache/ .pytest_cache/ .ruff_cache/
```

---

## 📄 License

This project is licensed under the terms of the [GNU GPL v3 License](LICENSE).

