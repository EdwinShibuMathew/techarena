---
name: reproduce-v6
description: >-
  Use this skill when the user asks to reproduce, replay, or verify the V6 baseline model inference, check submission CSV SHA256 checksum, or test prediction pipeline.
---

# Reproduce V6 Baseline Inference

This skill guides the agent through running the portable V6 benchmark inference pipeline and validating byte-for-byte reproducibility against the submitted contest benchmark.

## Prerequisites Check

1. Verify environment and CUDA device availability:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```
2. Verify asset bundle integrity before starting inference:
   ```bash
   python tools/consolidate.py verify
   ```

## Execution Steps

1. **Run Inference**:
   Execute the portable predictor using the GPU:
   ```bash
   python tools/predict_v6.py
   ```
   *(If CUDA is not available or running in a CPU-only environment, add `--device cpu`)*:
   ```bash
   python tools/predict_v6.py --device cpu
   ```

2. **Custom Output Destination** (Optional):
   To avoid modifying existing default artifacts, supply a custom target:
   ```bash
   python tools/predict_v6.py --output artifacts/teammate_reproduced_v6.csv
   ```

## Validation & Success Criteria

1. Check that the script exits with code `0`.
2. Verify the output CSV matches the benchmark SHA256 digest:
   `c386d545b3746e05dd515020c6185afda06cefdbcdf7d48df022454762fb378c`
3. The script internally asserts max absolute difference against reference member predictions `<= 1e-15`.
