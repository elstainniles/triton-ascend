# Pointer Bitcast Result File Index

After `../run.sh` finishes, return `../TEST_RESULT.md` first.
It identifies the failing diagnostic group. Do not return every file in this
directory unless it is explicitly requested.

| File | When it is needed |
| --- | --- |
| `00_environment.txt` | Import, version, wrong-build, device visibility, or collection failures |
| `00_setup_error.log` | The runner reports `SETUP ERROR` before pytest starts |
| `10_e2e.log` | A real-world pointer bitcast case fails; contains the full traceback |
| `10_e2e.xml` | Exact failing node IDs and machine-readable outcomes for that group |
| `20_matrix.log` | A source/destination type matrix case fails |
| `20_matrix.xml` | Exact failing type parameters and outcomes for that group |
| `30_runtime_alignment.log` | Dynamic even/odd or tensor-lane runtime assertion behavior fails |
| `30_runtime_alignment.xml` | Exact failing runtime-alignment node IDs and outcomes |
| `40_mlir_contract.log` | Static proof, runtime assert insertion, or compile-time rejection fails |
| `40_mlir_contract.xml` | Exact failing MLIR-contract node IDs and outcomes |
| `90_manifest.tsv` | A group was not run, timed out, or returned an unexpected exit status |
| `99_summary_builder.log` | `TEST_RESULT.md` is missing or cannot be generated |

When a text-log slice is requested, print it with stable line numbers:

```bash
nl -ba third_party/ascend/unittest/pytest_ut/pointer_bitcast_e2e_cases/test_results/30_runtime_alignment.log \
  | sed -n 'START,ENDp'
```

Replace the file and `START,END` with the values requested after the
compact summary has been reviewed. XML files should normally be returned whole
because each diagnostic group contains at most 17 tests.
