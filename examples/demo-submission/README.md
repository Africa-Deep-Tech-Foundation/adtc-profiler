# Demo submission

A complete, working example of the submission directory the profiler expects:
`metadata.json` plus the model file it points to. Use it to see the profiler
run end to end before building your own submission.

```bash
# 1. Fetch the demo model (~100 MB, SmolLM2-135M Q4_K_M)
./download_model.sh

# 2. Run the profiler against this directory
adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json
```

Add `--skip-accuracy` while iterating for a much faster loop; ship the output
of a full run.

## Adapting it for your team

Copy this directory and edit `metadata.json`:

- **All top-level fields are required** and no extra fields are allowed
  (`additionalProperties: false`) — the profiler validates before benchmarking
  and tells you exactly what's wrong.
- Replace the `submitter` placeholders with your real details.
- `test_prompts` must contain **exactly two** prompts — judges use them
  alongside domain and hidden prompts.
- `model.parameters_estimate` is checked against the actual parameter count
  read from your GGUF's tensor table (±15%) — state it honestly.
- `_runtime.model_path` tells the profiler where your model file lives,
  relative to this directory. Underscore-prefixed keys never appear in the
  report.

The canonical field-by-field contract is
[`src/adtc_profiler/schema/adtc-profiler.schema.json`](../../src/adtc_profiler/schema/adtc-profiler.schema.json).
