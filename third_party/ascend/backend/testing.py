# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import builtins
import multiprocessing
import os
from datetime import datetime, timezone
from typing import Optional

import triton.runtime as runtime
from triton.knobs import cache


class ProfilerResultMismatchError(RuntimeError):

    def __init__(self, target_kernel_name: str, expected_rows: int, actual_rows: int):
        self.target_kernel_name = target_kernel_name
        self.expected_rows = expected_rows
        self.actual_rows = actual_rows
        super().__init__(
            "Profiler rows filtered by target kernel name do not match the expected count. "
            f"target_kernel_name={target_kernel_name!r}, expected_rows={expected_rows}, actual_rows={actual_rows}")


_EVENT_BATCH_SIZE = 1024


def do_bench_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
):
    """Benchmark NPU callables while preserving the public API contract.

    Device events are used by default so compute kernels and stream-ordered
    tasks such as MemcpyAsync share the same lightweight measurement path.
    Supplying a profiler-specific option keeps the legacy profiler behaviour;
    incomplete profiler data falls back to event timing for the whole batch.
    """
    use_legacy_profiler = prof_dir is not None or keep_res or target_kernel_name is not None
    if use_legacy_profiler:
        try:
            result = _do_bench_npu_profiler(
                funcs,
                warmup=warmup,
                active=active,
                clear_l2_cache=clear_l2_cache,
                prof_dir=prof_dir,
                keep_res=keep_res,
                target_kernel_name=target_kernel_name,
            )
            values = result if isinstance(result, list) else [result]
            if all(value != float("inf") for value in values):
                return result
        except ProfilerResultMismatchError as exc:
            import warnings

            warnings.warn(
                "NPU profiler did not produce a complete set of target kernel records; "
                "the operation may be a non-compute task such as MemcpyAsync. "
                f"Falling back to device-event timing. Details: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return _do_bench_npu_event(funcs, warmup=warmup, active=active, clear_l2_cache=clear_l2_cache)


def _do_bench_npu_event(funcs, warmup=5, active=30, clear_l2_cache=False):
    if not isinstance(funcs, list):
        funcs = [funcs]
    if not funcs:
        raise ValueError("funcs must contain at least one callable")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if active <= 0:
        raise ValueError("active must be positive")

    driver = runtime.driver.active
    device_interface = driver.get_device_interface()
    cache_buffer = driver.get_empty_cache_for_benchmark() if clear_l2_cache else None
    results = []

    def run_once(fn):
        if cache_buffer is not None:
            driver.clear_cache(cache_buffer)
        fn()

    for fn in funcs:
        # Compile and initialize runtime state before applying the requested
        # warmup count, matching the historical do_bench_npu behaviour.
        fn()
        device_interface.synchronize()

        for _ in builtins.range(warmup):
            run_once(fn)
        device_interface.synchronize()

        elapsed_ms = 0.0
        measured = 0
        while measured < active:
            batch_size = min(_EVENT_BATCH_SIZE, active - measured)
            start_events = [device_interface.Event(enable_timing=True) for _ in builtins.range(batch_size)]
            end_events = [device_interface.Event(enable_timing=True) for _ in builtins.range(batch_size)]

            for start_event, end_event in zip(start_events, end_events):
                if cache_buffer is not None:
                    driver.clear_cache(cache_buffer)
                start_event.record()
                fn()
                end_event.record()

            device_interface.synchronize()
            elapsed_ms += sum(start.elapsed_time(end) for start, end in zip(start_events, end_events))
            measured += batch_size

        results.append(elapsed_ms / active)

    if len(results) == 1:
        return results[0]
    return results


def _do_bench_npu_profiler(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
):
    """Internal profiler benchmark used by autotune and profiling flows."""
    import torch
    import torch_npu

    if not isinstance(funcs, list):
        funcs = [funcs]

    # warmup kernel
    for fn in funcs:
        fn()
        torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = cache.get_triton_dir("profile_results")
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")

    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()  # to avoid type cast
        buffer.sum()
        torch.npu.synchronize()  # shake out of any npu error

    total = warmup + active
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
    ) as prof:
        for fn in funcs:
            for _ in builtins.range(total):
                if clear_l2_cache:
                    buffer.sum()  # use buffer read to clear l2 cache
                    torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
    if clear_l2_cache:
        del buffer

    try:
        return _collect_prof_result(
            torch_path,
            funcs,
            warmup,
            active,
            target_kernel_name=target_kernel_name,
            clear_l2_cache=clear_l2_cache,
        )
    finally:
        _rm_dic(keep_res, torch_path)


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)


def _collect_prof_result(
    base_dir: str,
    funcs,
    num_warmup: int,
    num_active: int,
    target_kernel_name: Optional[str] = None,
    clear_l2_cache: bool = False,
):
    """
    Collect kernel performance from kernel_details.csv, returned in millisecond.
    The first `num_warmup` rows of each function are warmup data and will be ignored, the next `num_active` rows will be averaged.

    :param base_dir: the profiler path
    :type base_dir: str
    :param funcs: a list of Callable being profiled
    :type funcs: List[Callable]
    :param num_warmup: warmup count in kernel_details.csv of each fn
    :type num_warmup: int
    :param num_active: active count in kernel_details.csv of each fn
    :type num_active: int
    :param target_kernel_name: target triton kernel name reported by profiler
    :type target_kernel_name: Optional[str]
    """

    import numpy as np
    import pandas as pd

    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(funcs)
    if kernel_details_file is None:
        if num_funcs == 1:
            return float("inf")
        else:
            return [float("inf")] * num_funcs

    df = pd.read_csv(kernel_details_file)
    # filter out l2 cache clearing operation
    filter_cond = (not clear_l2_cache) | ~df["Type"].str.contains(r"^ReduceSum$", case=False, na=False)
    filter_df = df[filter_cond]
    if target_kernel_name is not None:
        filter_df = filter_df[filter_df["Name"] == target_kernel_name]

    expected_rows = num_funcs * (num_warmup + num_active)
    actual_rows = len(filter_df)
    if target_kernel_name is not None and actual_rows != expected_rows:
        raise ProfilerResultMismatchError(target_kernel_name, expected_rows, actual_rows)

    time_cost = [0] * num_funcs
    for func_idx in np.arange(0, num_funcs):
        for active_index in np.arange(0, num_active):
            row_index = func_idx * (num_warmup + num_active) + num_warmup + active_index
            time_cost[func_idx] += filter_df.iloc[row_index]["Duration(us)"]
    time_cost = [x / num_active / 1e3 for x in time_cost]

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost
