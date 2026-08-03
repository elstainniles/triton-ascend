# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import inspect
from types import SimpleNamespace

import pytest

from triton.backends.ascend import testing


class _FakeEvent:

    def __init__(self, device, enable_timing):
        assert enable_timing is True
        self.device = device
        self.timestamp = None

    def record(self):
        self.timestamp = self.device.clock

    def elapsed_time(self, end_event):
        return end_event.timestamp - self.timestamp


class _FakeDevice:

    def __init__(self):
        self.clock = 0.0
        self.synchronize_count = 0

    def Event(self, enable_timing):
        return _FakeEvent(self, enable_timing)

    def synchronize(self):
        self.synchronize_count += 1


class _FakeDriver:

    def __init__(self):
        self.device = _FakeDevice()
        self.clear_cache_count = 0
        self.cache_buffer = object()

    def get_device_interface(self):
        return self.device

    def get_empty_cache_for_benchmark(self):
        return self.cache_buffer

    def clear_cache(self, cache_buffer):
        assert cache_buffer is self.cache_buffer
        self.clear_cache_count += 1
        self.device.clock += 10.0


def _install_fake_driver(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(testing.runtime, "driver", SimpleNamespace(active=driver))
    return driver


def _timed_fn(driver, duration, calls):

    def fn():
        calls.append(duration)
        driver.device.clock += duration

    return fn


def test_do_bench_npu_public_signature_is_unchanged():
    signature = inspect.signature(testing.do_bench_npu)

    assert list(signature.parameters) == [
        "funcs",
        "warmup",
        "active",
        "clear_l2_cache",
        "prof_dir",
        "keep_res",
        "target_kernel_name",
    ]
    assert signature.parameters["warmup"].default == 5
    assert signature.parameters["active"].default == 30
    assert signature.parameters["clear_l2_cache"].default is False
    assert signature.parameters["prof_dir"].default is None
    assert signature.parameters["keep_res"].default is False
    assert signature.parameters["target_kernel_name"].default is None


def test_do_bench_npu_uses_events_and_returns_scalar(monkeypatch):
    driver = _install_fake_driver(monkeypatch)
    calls = []
    fn = _timed_fn(driver, duration=2.5, calls=calls)

    result = testing.do_bench_npu(fn, warmup=2, active=3)

    assert result == 2.5
    assert len(calls) == 1 + 2 + 3
    assert driver.device.synchronize_count == 3


def test_do_bench_npu_returns_one_result_per_callable(monkeypatch):
    driver = _install_fake_driver(monkeypatch)
    first_calls = []
    second_calls = []
    funcs = [
        _timed_fn(driver, duration=1.0, calls=first_calls),
        _timed_fn(driver, duration=4.0, calls=second_calls),
    ]

    result = testing.do_bench_npu(funcs, warmup=1, active=2)

    assert result == [1.0, 4.0]
    assert len(first_calls) == 4
    assert len(second_calls) == 4


def test_do_bench_npu_excludes_l2_clear_from_event_interval(monkeypatch):
    driver = _install_fake_driver(monkeypatch)
    calls = []
    fn = _timed_fn(driver, duration=3.0, calls=calls)

    result = testing.do_bench_npu(fn, warmup=1, active=2, clear_l2_cache=True)

    assert result == 3.0
    assert driver.clear_cache_count == 3


def test_do_bench_npu_bounds_event_allocation_in_batches(monkeypatch):
    driver = _install_fake_driver(monkeypatch)
    calls = []
    fn = _timed_fn(driver, duration=1.0, calls=calls)
    monkeypatch.setattr(testing, "_EVENT_BATCH_SIZE", 2)

    result = testing.do_bench_npu(fn, warmup=0, active=5)

    assert result == 1.0
    # Compile sync + warmup sync + three measurement batch syncs.
    assert driver.device.synchronize_count == 5


def test_do_bench_npu_legacy_profiler_success_is_preserved(monkeypatch):
    monkeypatch.setattr(testing, "_do_bench_npu_profiler", lambda *args, **kwargs: 7.0)

    result = testing.do_bench_npu(lambda: None, keep_res=True)

    assert result == 7.0


def test_do_bench_npu_profiler_mismatch_falls_back_to_events(monkeypatch):
    driver = _install_fake_driver(monkeypatch)
    calls = []
    fn = _timed_fn(driver, duration=6.0, calls=calls)

    def fail_profiler(*args, **kwargs):
        raise testing.ProfilerResultMismatchError("copy_kernel", 3, 0)

    monkeypatch.setattr(testing, "_do_bench_npu_profiler", fail_profiler)

    with pytest.warns(RuntimeWarning, match="MemcpyAsync"):
        result = testing.do_bench_npu(fn, warmup=0, active=3, target_kernel_name="copy_kernel")

    assert result == 6.0


@pytest.mark.parametrize("warmup,active", [(-1, 1), (0, 0), (0, -1)])
def test_do_bench_npu_rejects_invalid_iteration_counts(monkeypatch, warmup, active):
    _install_fake_driver(monkeypatch)

    with pytest.raises(ValueError):
        testing.do_bench_npu(lambda: None, warmup=warmup, active=active)
