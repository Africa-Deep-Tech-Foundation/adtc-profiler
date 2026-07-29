"""ADTC 2026 reference profiler.

Composes llama-bench, psutil, lm-evaluation-harness, and lm-sensors into a
schema-valid JSON benchmark report. Same tool runs in two modes:
- participant: on contestant's own laptop (Gate 1) -> submission.json
- audit:       in cloud VM matched to Standard Laptop profile (Gate 2) -> audit.json
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.1.0"
PROFILER_VERSION = f"adtc-profiler {__version__}"
