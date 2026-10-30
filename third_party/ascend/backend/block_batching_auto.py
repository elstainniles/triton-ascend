# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Bounded automatic block batching. Estimates rank; backend checks admit.

No device benchmarking or recompilation takes place during dispatch. Each
variant owns its complete binary, ABI, metadata, workspace and memory guard.
"""
import hashlib
import json
import math
import threading
from collections import namedtuple
from pathlib import Path

from .block_batching import (BatchingRejected, LayoutTargetRejected, BatchRequest, IMPLEMENTATION_VERSION,
                             _capture_state, _freeze_source, _internal_backend_error, _layout_resource_refusal,
                             _runtime_positions)

AUTO_VERSION = 1
FACTORS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
MAX_COMPILE_ATTEMPTS = 4
MAX_VARIANTS = 3


def _cause(error):
    from triton.compiler.errors import MLIRCompilationError
    while isinstance(error, MLIRCompilationError) and error.__cause__ is not None:
        error = error.__cause__
    return error


def runtime_eligibility(grid, factor, axis, cores):
    """Pure, bounded integer policy; original-domain guard runs separately."""
    if type(cores) is not int or cores <= 0:
        return False, "CORE_COUNT_UNKNOWN"
    if axis not in (0, 1, 2) or not 1 < factor <= 64:
        return False, "INVALID_AUTO_PLAN"
    q = (grid[axis] + factor - 1) // factor
    tasks = q * math.prod(grid[j] for j in range(3) if j != axis)
    if tasks >= 2**63:
        return False, "AUTO_GRID_OVERFLOW"
    if tasks < 2 * cores:
        return False, "INSUFFICIENT_PARALLELISM"
    if 8 * grid[axis] < 7 * factor * q:
        return False, "LOW_TAIL_EFFICIENCY"
    return True, "AUTO_GRID_ACCEPTED"


def rank_candidates(rows, raw_ub):
    """Keep a small fallback and explore the resource shoulder, not every F.

    One eighth of raw UB is a target *largest tensor* size, not a memory
    safety bound. Full backend allocation validation remains mandatory.
    """
    valid = [r for r in rows if r['status'] == 'ANALYZED']
    if not valid or raw_ub <= 0:
        return []
    target = raw_ub // 8
    shoulder = [r for r in valid if r['resources']['largest_tensor_bytes'] <= target]
    preferred = max(shoulder, key=lambda r: r['factor']) if shoulder else min(valid, key=lambda r: r['factor'])
    small = min(valid, key=lambda r: r['factor'])
    lower = [r for r in valid if r['factor'] < preferred['factor'] and r['factor'] != small['factor']]
    backup = min(lower, key=lambda r: abs(r['factor'] - preferred['factor'] / 2)) if lower else None
    chosen = []
    for row in (small, preferred, backup):
        if row is not None and row not in chosen:
            chosen.append(row)
    return chosen


def analyze_candidates(backend, frozen, options):
    from triton._C.libtriton import ascend, ir
    from triton.runtime.cache import get_cache_manager
    from .compiler import block_batching_frontend_passes
    base_options = backend.parse_options(dict(options, block_batching_auto=False, block_batching_factor=1))
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = frozen.make_ir(backend.target, base_options, backend.get_codegen_implementation(base_options),
                            backend.get_module_map(), context)
    block_batching_frontend_passes(module).run(module, 'block_batching_auto_normalize')
    summary = dict(ascend.passes.ttir.block_batching_resource_summary(module))
    axes = list(summary['used_axes'])
    axis = options.get('block_batching_axis', -1)
    if summary['has_control_flow']:
        raise BatchingRejected('AUTO_CONTROL_FLOW_UNSUPPORTED')
    if axis == -1:
        if len(axes) != 1:
            raise BatchingRejected('AUTO_AXIS_AMBIGUOUS')
        axis = axes[0]
    elif axis not in axes:
        raise BatchingRejected('AUTO_AXIS_UNUSED')
    text = str(module)
    key = hashlib.sha256(text.encode()).hexdigest()
    path = get_cache_manager(key).put(text, 'auto.normalized.ttir', binary=False)
    rows = []
    for factor in FACTORS:
        candidate = ir.parse_mlir_module(path, context)
        candidate.context = context
        native = dict(ascend.passes.ttir.rewrite_block_batching(candidate, factor, axis,
                                                               options.get('block_batching_layout', 'auto')))
        row = dict(factor=factor, axis=axis, status='REJECTED', reason=native['reason'])
        if native['compile_status'] == 'REWRITTEN':
            resource = dict(ascend.passes.ttir.block_batching_resource_summary(candidate))
            row.update(status='ANALYZED', resources=resource,
                       layout_rewritten=any(a.get('rewritten') for a in json.loads(native['layout_json'])['accesses']))
            if resource['unknown_tensor_size']:
                row.update(status='REJECTED', reason='AUTO_DYNAMIC_LOCAL_SIZE')
        rows.append(row)
    return rows


class AttemptCompiler:
    """Share one budget across factor and layout retries; cache known refusals."""

    def __init__(self, backend, compile_variant, baseline, cache, attempts):
        self.backend, self.compile_variant, self.baseline = backend, compile_variant, baseline
        self.cache, self.attempts = cache, attempts
        self.calls = 0

    def __call__(self, options, frozen):
        import subprocess
        if options.get('block_batching_factor', 1) == 1:
            return self.baseline
        factor, axis = options['block_batching_factor'], options['block_batching_axis']
        layout = options.get('block_batching_layout', 'auto')
        key = hashlib.sha256((self.backend.parse_options(options).hash() + frozen.hash()).encode()).hexdigest()
        filename = f'auto-refusal-{key}.json'
        refusal = self.cache.get_file(filename)
        if refusal:
            record = json.loads(Path(refusal).read_text())
            self.attempts.append(dict(record, cached=True))
            if record['kind'] == 'layout_resource':
                raise LayoutTargetRejected(factor, axis, record['diagnostic'])
            raise BatchingRejected(record['reason'])
        if self.calls >= MAX_COMPILE_ATTEMPTS:
            raise BatchingRejected('AUTO_COMPILE_BUDGET_EXHAUSTED')
        self.calls += 1
        record = dict(factor=factor, axis=axis, layout=layout, cached=False)
        try:
            result = self.compile_variant(options, frozen)
        except Exception as error:
            cause = _cause(error)
            if isinstance(cause, BatchingRejected):
                record.update(status='REJECTED', kind='capability', reason=str(cause))
                if isinstance(cause, LayoutTargetRejected):
                    record.update(kind='layout_resource', diagnostic=cause.diagnostic)
                self.attempts.append(record)
                self.cache.put(json.dumps(record, sort_keys=True), filename)
                raise
            if isinstance(cause, subprocess.CalledProcessError):
                detail = cause.stderr.decode(errors='replace') if isinstance(cause.stderr, bytes) else str(cause.stderr or '')
                if not _internal_backend_error(cause.returncode, detail) and _layout_resource_refusal(cause.returncode, detail):
                    record.update(status='REJECTED', kind='resource', reason='AUTO_BACKEND_RESOURCE_REJECTED',
                                  diagnostic=detail[-16000:])
                    self.attempts.append(record)
                    self.cache.put(json.dumps(record, sort_keys=True), filename)
                    raise BatchingRejected(record['reason']) from error
            # Do not let the legacy wrapper's broad subprocess fallback hide
            # an unknown compiler failure or publish a negative cache entry.
            self.attempts.append(dict(record, status='ERROR', reason=str(cause)[-2000:]))
            raise RuntimeError('automatic batching candidate compilation failed unexpectedly') from error
        self.attempts.append(dict(record, status='COMPILED', binary_hash=result.hash,
                                  required_ub_bits=getattr(result.metadata, 'required_ub_bits', None)))
        return result


def compile_auto_bundle(backend, src, options, compile_variant):
    import fcntl
    from triton.runtime.cache import FileCacheManager, get_cache_manager
    from .block_batching import compile_bundle
    from .utils import backend_policy, ub_size_in_kbytes_for_arch
    normalized = backend.parse_options(options)
    base_options = dict(options, block_batching_auto=False, block_batching_factor=1, block_batching_axis=-1)
    freeze_error = None
    try:
        frozen = _freeze_source(backend, src, base_options, normalized.block_batching_backend_revision)
    except BatchingRejected as error:
        frozen, freeze_error = src, str(error)
    baseline = compile_variant(base_options, frozen)
    key = hashlib.sha256((baseline.hash + normalized.hash() + f'auto:{AUTO_VERSION}').encode()).hexdigest()
    report = dict(mode='auto', auto_version=AUTO_VERSION, implementation_version=IMPLEMENTATION_VERSION,
                  compile_status='BASELINE_ONLY', reason='NO_AUTO_CANDIDATE', candidates=[], attempts=[],
                  planning=[], compile_budget=MAX_COMPILE_ATTEMPTS, resource_validation='backend_compile_success')
    variants, positions = [], []
    cache = get_cache_manager(key)
    try:
        if freeze_error:
            raise BatchingRejected(freeze_error)
        if type(cache) is not FileCacheManager:
            raise BatchingRejected('CUSTOM_CACHE_NOT_SUPPORTED')
        if not str(backend.target.arch).startswith(('Ascend950', 'Ascend910_95')):
            raise BatchingRejected('TARGET_NOT_VALIDATED')
        if normalized.is_pure_simt or normalized.compile_mode == 'simt_only':
            raise BatchingRejected('SIMT_NOT_SUPPORTED')
        if backend_policy not in (None, 'torch_npu'):
            raise BatchingRejected('FRAMEWORK_NOT_VALIDATED')
        positions = _runtime_positions(src)
        # Serialize the plan and refusal writes across independent JIT processes.
        lock_path = Path(cache.cache_dir) / 'auto-plan.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            plan_file = cache.get_file('auto-planning.json')
            if plan_file:
                planning = json.loads(Path(plan_file).read_text())
            else:
                planning = analyze_candidates(backend, frozen, options)
                cache.put(json.dumps(planning, sort_keys=True), 'auto-planning.json')
            report['planning'] = planning
            chosen = rank_candidates(planning, ub_size_in_kbytes_for_arch(backend.target.arch) * 1024)
            compiler = AttemptCompiler(backend, compile_variant, baseline, cache, report['attempts'])
            for row in chosen[:MAX_VARIANTS]:
                variant_options = dict(options, block_batching_auto=False, block_batching_factor=row['factor'],
                                       block_batching_axis=row['axis'])
                bundle = compile_bundle(backend, frozen, variant_options, compiler)
                report['candidates'].append(dict(factor=row['factor'], report=bundle.block_batching))
                if bundle.candidate is not None:
                    variants.append(bundle)
            report['compile_attempts'] = compiler.calls
        if variants:
            report.update(compile_status='COMPILED_GUARDED', reason='AUTO_CANDIDATES_COMPILED')
    except BatchingRejected as error:
        report['reason'] = str(error)
    return AutoBatchKernel(baseline, variants, report, positions, key)


class AutoBatchKernel:
    def __init__(self, baseline, variants, report, positions, key):
        from triton.runtime.cache import get_cache_manager
        self.baseline, self.variants = baseline, tuple(sorted(variants, key=lambda b: b.request.factor, reverse=True))
        self.block_batching, self.positions, self.hash = report, positions, key
        self._lock, self._local, self._ready = threading.RLock(), threading.local(), False
        fields = dict(baseline.metadata._asdict(), block_batching=report)
        self.metadata = namedtuple('AutoBatchKernelMetadata', sorted(fields))(**fields)
        manifest = dict(version=AUTO_VERSION, baseline_hash=baseline.hash,
                        candidate_hashes=[b.candidate.hash for b in self.variants], report=report)
        get_cache_manager(key).put(json.dumps(manifest, sort_keys=True), 'block_batching.auto.bundle.json')

    def __getattr__(self, name):
        return getattr(self.baseline, name)

    @property
    def last_launch(self):
        return getattr(self._local, 'launch', dict(runtime_status='NOT_LAUNCHED'))

    def _init_handles(self):
        with self._lock:
            if self._ready:
                return
            if _capture_state() and getattr(self.baseline, '_run', None) is None:
                raise RuntimeError('warm up the block batching baseline before graph capture')
            self.baseline._init_handles()
            for bundle in self.variants:
                bundle._init_handles()
            self._ready = True

    @property
    def run(self):
        self._init_handles()
        return self._dispatch

    def launch_metadata(self, grid, stream, *arguments):
        return None

    def _dispatch(self, *launch):
        from triton import knobs
        from .block_batching_guard import I32_MAX, integer
        grid = tuple(integer(x) for x in launch[:3])
        if any(x < 0 or x > I32_MAX for x in grid):
            raise ValueError('original grid must contain nonnegative i32 extents')
        if any(x == 0 for x in grid):
            self._local.launch = dict(runtime_status='EMPTY_GRID', applied_factor=1, original_grid=grid)
            return
        selected, factor, axis = self.baseline, 1, -1
        reason, checks, cores = self.block_batching['reason'], [], None
        arguments = launch[9:]
        capture = _capture_state()
        if capture is not False:
            reason = 'CAPTURE_BASELINE' if capture else 'CAPTURE_QUERY_UNAVAILABLE'
        elif self.variants:
            from .driver import NPUUtils
            cores = NPUUtils().get_aivector_core_num()
            runtime_args = [arguments[i] for i in self.positions]
            for bundle in self.variants:
                if bundle.candidate is None:
                    checks.append(dict(factor=bundle.request.factor, reason='CANDIDATE_LOAD_REJECTED'))
                    continue
                candidate_axis = bundle.block_batching['selected_axis']
                accepted, reason = runtime_eligibility(grid, bundle.request.factor, candidate_axis, cores)
                if accepted:
                    accepted, reason = bundle._guard(runtime_args, grid)
                checks.append(dict(factor=bundle.request.factor, reason=reason, accepted=accepted))
                if accepted:
                    selected, factor, axis = bundle.candidate, bundle.request.factor, candidate_axis
                    break
        forwarded = list(launch)
        forwarded[4], forwarded[5] = selected.function, selected.packed_metadata
        forwarded[6] = selected.launch_metadata(grid, launch[3], *arguments)
        # Deliberately no catch/replay after submission: outputs may be modified.
        result = selected.run(*forwarded)
        batched = list(grid)
        if factor > 1:
            batched[axis] = (grid[axis] + factor - 1) // factor
        self._local.launch = dict(runtime_status='APPLIED' if factor > 1 else 'BASELINE_SELECTED', mode='auto',
                                  applied_factor=factor, selected_axis=axis, original_grid=grid, batch_grid=tuple(batched),
                                  physical_cores=cores, reason=reason, checks=checks, selected_hash=selected.hash,
                                  plan_id=self.hash)
        if knobs.ascend.compile_only:
            self._local.launch.update(runtime_status='NOT_LAUNCHED', reason='COMPILE_ONLY')
        return result

    def __getitem__(self, grid):
        def runner(*args, stream=None):
            from triton import knobs
            from triton.runtime import driver
            self._init_handles()
            if stream is None:
                stream = driver.active.get_current_stream(driver.active.get_current_device())
            return self.run(*grid, stream, self.function, self.packed_metadata, None,
                            knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)
        return runner
