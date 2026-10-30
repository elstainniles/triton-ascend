# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host tests for resource rejection, exact artifact dispatch and bounded search."""
from collections import namedtuple
import subprocess
from types import SimpleNamespace

import pytest

from triton.backends.ascend import block_batching_auto as auto
from triton.backends.ascend.block_batching import BatchingRejected, BatchRequest
from triton.backends.ascend.compiler import NPUOptions

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


@pytest.mark.parametrize('value', [1, 0, 'auto', None])
def test_auto_boolean_option(value):
    with pytest.raises(ValueError, match='block_batching_auto'):
        NPUOptions(arch='Ascend950', block_batching_auto=value)


def test_auto_preserves_layout_and_has_separate_cache_identity():
    base = NPUOptions(arch='Ascend950')
    chosen = NPUOptions(arch='Ascend950', block_batching_auto=True)
    assert chosen.block_batching_layout == 'auto'
    assert chosen.hash() != base.hash()
    assert chosen.block_batching_backend_revision
    with pytest.raises(ValueError, match='conflicts'):
        NPUOptions(arch='Ascend950', block_batching_auto=True, block_batching_factor=4)


@pytest.mark.parametrize('grid,factor,cores,accepted', [
    ((2048, 1, 1), 4, 56, True), ((256, 1, 1), 2, 56, True),
    ((256, 1, 1), 4, 56, False), ((9, 100, 1), 4, 56, False),
    ((1024, 1, 1), 4, 0, False), ((2**31-1, 2**31-1, 5), 2, 56, False)])
def test_dynamic_grid_policy(grid, factor, cores, accepted):
    assert auto.runtime_eligibility(grid, factor, 0, cores)[0] is accepted


def test_planning_is_not_capped_at_four():
    rows = [dict(factor=f, axis=0, status='ANALYZED', resources=dict(largest_tensor_bytes=2048*f))
            for f in auto.FACTORS]
    assert [r['factor'] for r in auto.rank_candidates(rows, 256*1024)] == [2, 16, 8]
    assert auto.rank_candidates(rows, 0) == []


def test_native_resource_summary_is_only_a_planning_estimate(tmp_path):
    import json
    from triton._C.libtriton import ascend
    from test_block_batching_rewrite import parse, pointwise
    module = parse(tmp_path, pointwise(0))
    before = dict(ascend.passes.ttir.block_batching_resource_summary(module))
    assert before['used_axes'] == [0]
    assert before['largest_tensor_bytes'] == 16
    assert not before['unknown_tensor_size'] and not before['is_allocation_bound']
    report = dict(ascend.passes.ttir.rewrite_block_batching(module, 8, 0))
    assert report['compile_status'] == 'REWRITTEN'
    after = dict(ascend.passes.ttir.block_batching_resource_summary(module))
    # The layout rewrite widens its 32-element address vector to i64.
    assert after['largest_tensor_bytes'] == 256
    json.dumps(after)  # report and planning cache must be serializable


class Cache:
    def __init__(self, root):
        self.root = root

    def get_file(self, name):
        p = self.root / name
        return str(p) if p.exists() else None

    def put(self, text, name):
        p = self.root / name
        p.write_text(text)
        return str(p)


def attempt(tmp_path, fn):
    backend = SimpleNamespace(parse_options=lambda options: SimpleNamespace(hash=lambda: str(sorted(options.items()))))
    return auto.AttemptCompiler(backend, fn, 'baseline', Cache(tmp_path), [])


def test_known_resource_refusal_is_cached_without_recompilation(tmp_path):
    calls = []

    def rejected(options, source):
        calls.append(options)
        raise subprocess.CalledProcessError(1, 'compiler', stderr=b'ub overflow, requires 300 bits while 200 bits available')

    compiler = attempt(tmp_path, rejected)
    source = SimpleNamespace(hash=lambda: 'frozen-source')
    options = dict(block_batching_factor=16, block_batching_axis=0)
    for _ in range(2):
        with pytest.raises(BatchingRejected, match='AUTO_BACKEND_RESOURCE_REJECTED'):
            compiler(options, source)
    assert len(calls) == 1
    assert compiler.attempts[-1]['cached']


@pytest.mark.parametrize('detail', [b'unknown backend error', b'Assertion: ub overflow, requires 300 bits while 200 bits available'])
def test_unknown_and_internal_failure_are_not_hidden_or_cached(tmp_path, detail):
    def rejected(options, source):
        raise subprocess.CalledProcessError(1, 'compiler', stderr=detail)

    compiler = attempt(tmp_path, rejected)
    with pytest.raises(RuntimeError, match='unexpectedly'):
        compiler(dict(block_batching_factor=8, block_batching_axis=0), SimpleNamespace(hash=lambda: 'src'))
    assert not list(tmp_path.iterdir())


def test_factor_and_layout_calls_share_budget(tmp_path):
    calls = []

    def success(options, source):
        calls.append(options)
        return SimpleNamespace(hash='artifact', metadata=SimpleNamespace())

    compiler = attempt(tmp_path, success)
    source = SimpleNamespace(hash=lambda: 'src')
    for f, layout in ((2, 'auto'), (16, 'auto'), (16, 'semantic_only'), (8, 'auto')):
        compiler(dict(block_batching_factor=f, block_batching_axis=0, block_batching_layout=layout), source)
    with pytest.raises(BatchingRejected, match='BUDGET'):
        compiler(dict(block_batching_factor=4, block_batching_axis=0), source)
    assert len(calls) == auto.MAX_COMPILE_ATTEMPTS
    assert compiler(dict(block_batching_factor=1), source) == 'baseline'


class Artifact:
    def __init__(self, name):
        self.hash = name
        self.function = name + '-function'
        self.packed_metadata = name + '-metadata'
        self.metadata = namedtuple('Metadata', 'mix_mode')('aiv')
        self.calls = []

    def launch_metadata(self, *args):
        return self.hash + '-hooks'

    def run(self, *args):
        self.calls.append(args)


def dispatch_bundle(monkeypatch, tmp_path, accepted=True):
    from triton.runtime import cache
    from triton.backends.ascend import driver
    monkeypatch.setattr(cache, 'get_cache_manager', lambda key: Cache(tmp_path))
    monkeypatch.setattr(auto, '_capture_state', lambda: False)
    monkeypatch.setattr(driver, 'NPUUtils', lambda: SimpleNamespace(get_aivector_core_num=lambda: 56))
    base, candidate = Artifact('base'), Artifact('candidate')
    variant = SimpleNamespace(candidate=candidate, request=BatchRequest(4, 0),
                              block_batching={'selected_axis': 0}, _guard=lambda args, grid: (accepted, 'proof'))
    bundle = auto.AutoBatchKernel(base, [variant], dict(reason='ready'), [], 'plan')
    return bundle, base, candidate


@pytest.mark.parametrize('accepted', [True, False])
def test_dispatch_switches_entire_artifact_before_launch(monkeypatch, tmp_path, accepted):
    bundle, base, candidate = dispatch_bundle(monkeypatch, tmp_path, accepted)
    bundle._dispatch(1024, 1, 1, 'stream', 'stale-function', 'stale-metadata', None, None, None, 'arg')
    selected, unused = (candidate, base) if accepted else (base, candidate)
    assert not unused.calls and len(selected.calls) == 1
    call = selected.calls[0]
    assert call[:3] == (1024, 1, 1)  # original hidden-grid contract
    assert call[4:7] == (selected.function, selected.packed_metadata, selected.hash + '-hooks')
    assert bundle.last_launch['applied_factor'] == (4 if accepted else 1)


def test_no_replay_after_submission_error(monkeypatch, tmp_path):
    bundle, base, candidate = dispatch_bundle(monkeypatch, tmp_path)

    def partial_write(*args):
        raise RuntimeError('device execution failed after submission')

    candidate.run = partial_write
    with pytest.raises(RuntimeError, match='after submission'):
        bundle._dispatch(1024, 1, 1, 'stream', None, None, None, None, None)
    assert base.calls == []


def test_capture_uses_baseline(monkeypatch, tmp_path):
    bundle, base, candidate = dispatch_bundle(monkeypatch, tmp_path)
    monkeypatch.setattr(auto, '_capture_state', lambda: True)
    bundle._dispatch(1024, 1, 1, 'stream', None, None, None, None, None)
    assert len(base.calls) == 1 and not candidate.calls
    assert bundle.last_launch['reason'] == 'CAPTURE_BASELINE'
