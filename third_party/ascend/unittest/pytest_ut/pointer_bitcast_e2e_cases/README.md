# Pointer Bitcast Temporary E2E Suite

This directory temporarily keeps the complete 46-case acceptance suite for the
different-width pointer bitcast change. Remove the directory after remote NPU
validation has finished and the final repository tests have been selected.

Run from anywhere inside the built test container:

```bash
bash third_party/ascend/unittest/pytest_ut/pointer_bitcast_e2e_cases/run.sh
```

The runner defaults to container device 0. Override it only when the container
directly exposes a different logical device:

```bash
NPU_DEVICE=1 bash third_party/ascend/unittest/pytest_ut/pointer_bitcast_e2e_cases/run.sh
```

The runner automatically prefers the current repository's `build/lib.*`
Python package and matching `triton-opt`, uses an isolated Triton cache, and
writes the complete result to `TEST_RESULT.md`.

Return only the result through Git:

```bash
git add third_party/ascend/unittest/pytest_ut/pointer_bitcast_e2e_cases/TEST_RESULT.md
git commit -m "[ascend](test) record pointer bitcast e2e result"
git push
```

Acceptance requires exactly `46 passed` and `0 skipped`. The suite contains 18
real-world/type-matrix regressions, 17 runtime-alignment cases, and 11 direct
TritonToLinalg MLIR contracts.
