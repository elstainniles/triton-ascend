# Automatic block batching, version 1

This opt-in policy extends the fixed-factor implementation on
`feat/block-batching-ta-only`. It changes the program batching factor through
the existing TTIR rewrite, without changing kernel source, mathematical
operations, input shapes, or the installed BishengIR compiler.

```python
compiled = kernel[grid](*args, block_batching_auto=True, **kernel_options)
print(compiled.block_batching)  # planning and compilation decisions
print(compiled.last_launch)     # decision for this thread's latest call
```

The default is disabled. An explicit `block_batching_factor > 1` conflicts
with automatic selection and raises `ValueError`. An explicit
`block_batching_axis` can constrain the selected axis. Normal cache identities
include the option, policy version, frontend/backend revisions and specialized
source; changing automatic selection does not reuse a fixed-factor bundle.

## Planning and dispatch

1. Freeze the frontend source and compile the unchanged factor-1 baseline.
2. Normalize TTIR. Require one unambiguous used program-id axis, or the explicit
   axis. Version 1 conservatively declines control flow and unknown local sizes.
3. Analyze independent rewrites for factors
   `2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64`. There is no kernel-name allowlist.
4. Use the largest static non-pointer tensor as a ranking heuristic. The target
   is one eighth of raw UB. Keep the smallest legal factor, the largest under
   the target, and a smaller backup near half that preferred factor. Compile
   at most three selected factors with at most **four extra compilation calls**
   shared across factor and layout retries. The baseline is outside that budget.
5. Keep only complete artifacts admitted by the existing backend and batching
   validation. In descending factor order, require at least two tasks per
   vector core, at least 7/8 tail utilization, and the existing runtime memory
   guard. Otherwise dispatch the original baseline.

The compilation counter counts callback attempts, including successful
compiler-cache lookups. This implementation does not benchmark candidates,
search all factors, tune `num_warps`, or promise the fastest factor. Device
kernel time and Python dispatch overhead are separate metrics.

## Resource refusals and fallback

The static tensor and floating live-byte estimates are **not UB allocation
bounds**. Lowering may introduce temporary buffers, alignment and workspaces.
The standard backend allocator must successfully compile every executable
candidate. Its recognized `ub overflow, requires ... bits while ... bits
available` refusal never produces a launchable candidate.

The existing layout retry can compile `semantic_only` from the same frozen
source. It consumes the same four-call budget. Other selected factors can then
be attempted, and factor 1 remains available. Known capability/resource
refusals are cached under the exact compilation identity; an interprocess lock
protects planning and negative-cache writes. Unknown compiler errors and
internal failures propagate with their diagnostics instead of becoming a
successful baseline result.

Dispatch switches the entire artifact: function, launch metadata, workspace
contract and guard. The launcher receives the original grid, preserving the
hidden original-extent ABI. Graph capture uses the warmed baseline. Invalid
original grids raise; empty grids do not launch. **There is no exception-driven
replay after device submission**, since outputs might already have changed.

Compilation success is the backend's resource admission check, not a general
proof of numerical correctness or of freedom from backend defects. Candidate
correctness must be validated on the workload before performance is accepted.

## Scope and tests

This version targets the existing validated Ascend950/Ascend910_95 AIV path,
Torch NPU, AST source and the standard filesystem cache. Unsupported cases
retain the existing conservative baseline behavior. It inherits the fixed
batching rewrite and memory-guard capability limits.

`third_party/ascend/unittest/pytest_ut/test_block_batching_auto.py` covers option
validation, native resource summaries, dynamic grids, candidate diversity,
negative caching, shared compilation budget, complete artifact dispatch,
capture fallback, and no replay after submission errors. Run it with the
installed candidate wheel; do not use a source overlay for validation.
