# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""TTIR selection in a CPU worker; only F1 and the winner reach the backend."""
import json
from pathlib import Path
import tempfile


class _NotApplicable(ValueError):
    """A modeled compiler decline, not an unexpected frontend error."""


def _native(action, contract, **inputs):
    from triton._C.libtriton import ascend
    contract = dict(contract, argument_order=contract.get("argument_order", list(contract.get("arguments", {}))))
    return json.loads(
        ascend.select_block_candidates(json.dumps(dict(action=action, contract=contract, **inputs), allow_nan=False)))


def candidate_plan(contract, *, probe, axis=-1):
    requested = _native("plan", contract, axis=axis)["probes"]
    reports = []
    for item in requested:
        try:
            reports.append(dict(item, report=probe(item["axis"], item["factor"])))
        except _NotApplicable:
            continue
    return _native("plan", contract, axis=axis, reports=reports)


def _select_ttir(backend, frozen, options, env_vars, contract, scalar_arguments, output):
    """Child-only frontend work. Retain the winner's exact TTIR for compilation."""
    from triton import knobs
    from triton._C.libtriton import ascend, ir
    from .block_batching_report import _has_dot
    from .block_batching_worker import (BatchDecline, _variant_key, _candidate_cache, _cached_decline, _pack_ttir)
    from .compiler import make_ttir

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    frontends, declines = {}, {}

    def frontend(axis, factor):
        identity = (axis, factor)
        if identity in declines:
            raise _NotApplicable(declines[identity]["reason"])
        if identity not in frontends:
            values = dict(options, block_batching_factor=factor, block_batching_axis=axis)
            if factor > 1:
                values["block_batching_grid"] = tuple(contract["grid"])
            opts = backend.parse_options(values)
            variant = _variant_key(frozen, backend, opts, env_vars)
            cache = _candidate_cache(variant) if factor > 1 else None
            refusal = _cached_decline(cache,
                                      variant) if cache is not None and not knobs.compilation.always_compile else None
            if refusal is None:
                module = frozen.make_ir(backend.target, opts, backend.get_codegen_implementation(opts),
                                        backend.get_module_map(), context)
                cv = _has_dot(module)
                metadata = dict(vars(opts), hash=variant)
                result = make_ttir(module, metadata, opts)
                if isinstance(result, BatchDecline):
                    refusal = result.payload()
            if refusal is not None:
                declines[identity] = refusal
                raise _NotApplicable(refusal["reason"])
            frontends[identity] = module, metadata, cv
        return frontends[identity][1].get("block_batching", {})

    frontend(-1, 1)
    cv = frontends[-1, 1][2]
    arch, _, limit, _ = backend.parse_options(options).block_compile_context
    if limit is None:
        raise ValueError("automatic TTIR selection requires explicit AIC/AIV core counts")
    a3 = arch.startswith("Ascend910_93")
    contract = dict(contract, a3_resource_policy=a3)
    if a3:
        contract['cv_empirical_short_batch'] = False
    axis = options.get("block_batching_axis", -1)
    if cv:
        plan = candidate_plan(contract, axis=axis, probe=frontend)
    else:
        extents = contract["grid"] if axis == -1 else [contract["grid"][axis]]
        plan = dict(candidates=[dict(axis=axis, factor=f) for f in (2, 4) if max(extents) >= f],
                    reason="VV_F2_F4_AUTOMATIC_AXIS")
    cores = limit[1 if cv else 2]
    rows, prepared = [], {}
    for spec in [dict(axis=-1, factor=1), *plan["candidates"]]:
        requested_axis, factor = spec["axis"], spec["factor"]
        try:
            report = frontend(requested_axis, factor)
        except _NotApplicable as error:
            rows.append(dict(factor=factor, axis=requested_axis, status="COMPILE_REJECTED", error=str(error)))
            continue
        module, metadata, _ = frontends[requested_axis, factor]
        selected_axis = report.get("selected_axis", -1)
        merge_kind = spec.get("merge_kind", "unmerged" if factor == 1 else "vv")
        config = dict(contract=contract, scalar_arguments=scalar_arguments, factor=factor, axis=selected_axis, cv=cv,
                      arch=arch, cores=cores, merge_kind=merge_kind,
                      preserve_grid=report.get("vv_lowering") == "sequential_iterations_v1")
        row = json.loads(ascend.score_block_ttir(module, json.dumps(config, allow_nan=False)))
        row.update(cv=cv, merge_kind=merge_kind, requested_axis=requested_axis)
        rows.append(row)
        prepared["unmerged" if factor == 1 else f"a{selected_axis}_f{factor}"] = module, metadata
    selection = _native("rank_ttir", contract, candidates=rows)
    if a3:
        selection.update(model="ttir-a3-grouping-v1",
                         reason="A3_TTIR_GROUPING" if selection['factor'] > 1 else "A3_KEEP_BASELINE")
    chosen = selection["selected_key"]
    snapshot = _pack_ttir(*prepared[chosen], output) if chosen != "unmerged" else None
    plan["declines"] = [dict(axis=a, factor=f, **decline) for (a, f), decline in declines.items()]
    return dict(selection=selection, plan=plan, prepared=snapshot)


def compile_auto_bundle(backend, src, options, compile_variant, *, env_vars):
    """Score all variants at TTIR; keep a complete, capture-ready F1 fallback."""
    from . import block_batching_worker as compilation
    from .block_batching_worker import BatchDecline, _variant_key

    common = {k: v for k, v in options.items() if k not in ("block_auto_contract", "block_batching_grid")}
    baseline_options = dict(common, block_batching_factor=1, block_batching_axis=-1)
    if not options.get("block_auto_contract"):
        return compile_variant(baseline_options)
    contract = json.loads(options["block_auto_contract"])
    normalized = backend.parse_options(options)
    refusal = compilation._freeze_refusal(normalized) or compilation._candidate_refusal(backend, src, normalized)
    if refusal:
        return compile_variant(baseline_options)
    positions = compilation._runtime_positions(src)
    if isinstance(positions, BatchDecline):
        return compile_variant(baseline_options)
    scalar_arguments = [contract["arguments"].get(src.fn.arg_names[index], {}).get("value") for index in positions]
    frozen = compilation._freeze_source(backend, src, baseline_options, normalized.block_batching_backend_revision)
    if isinstance(frozen, BatchDecline):
        return compile_variant(baseline_options)
    # The hook supplies the same immutable environment to F1 and each child.
    env_vars = dict(env_vars)
    selection_options = dict(common, block_batching_factor=1)
    variant = _variant_key(frozen, backend, backend.parse_options(selection_options), env_vars)
    body = compilation._worker_request(backend, frozen, selection_options, env_vars, variant)
    body.update(contract=contract, scalar_arguments=scalar_arguments)
    with tempfile.TemporaryDirectory(prefix="triton-selected-ttir-") as directory:
        output = Path(directory) / "selected.ttir"
        body["output"] = str(output)
        try:
            report = compilation._attempt("select", body, timeout=30).payload["report"]
        except compilation.BatchCompileError as error:
            if error.kind != "Timeout":
                raise
            report = dict(
                selection=dict(factor=1, selected_axis=-1, selected_key="unmerged", reason="TTIR_SELECTION_TIMEOUT",
                               failure_kind=error.kind, candidates={}, model_complete=False, device_timing_used=False),
                plan=dict(candidates=[]), prepared=None)
        return _compile_selected(backend, frozen, common, baseline_options, compile_variant, env_vars, positions,
                                 contract, report, output)


def _compile_selected(backend, frozen, common, baseline_options, compile_variant, env_vars, positions, contract, report,
                      output):
    from . import block_batching as batching
    from .block_batching_worker import (BatchCompileError, CandidateCompiler, _variant_key, _candidate_cache,
                                        _remember_decline)
    selection, plan, prepared = report["selection"], report["plan"], report["prepared"]
    for decline in plan.get("declines", ()):
        values = dict(common, block_batching_factor=decline["factor"], block_batching_axis=decline["axis"],
                      block_batching_grid=tuple(contract["grid"]))
        identity = _variant_key(frozen, backend, backend.parse_options(values), env_vars)
        _remember_decline(_candidate_cache(identity), identity,
                          {k: v
                           for k, v in decline.items()
                           if k not in ("axis", "factor")})
    factor, axis = selection["factor"], selection["selected_axis"]
    if type(factor) is not int or factor not in (1, 2, 4) or (factor > 1 and axis not in (0, 1, 2)):
        raise BatchCompileError("invalid TTIR selection", kind="ProtocolError", stage="select")
    baseline = compile_variant(baseline_options, frozen)
    if baseline.hash != _variant_key(frozen, backend, backend.parse_options(baseline_options), env_vars):
        raise BatchCompileError("baseline cache identity changed", kind="InvalidArtifact", stage="bind")
    result, declines = None, []
    if factor > 1:
        key = f"a{axis}_f{factor}"
        row = selection["candidates"][key]
        requested_axis = row["requested_axis"]
        if (not isinstance(prepared, dict) or prepared.get("report", {}).get("selected_axis") != axis
                or selection["selected_key"] != key or row["factor"] != factor or row["axis"] != axis
                or not any(p["axis"] == requested_axis and p["factor"] == factor for p in plan["candidates"])):
            raise BatchCompileError("selection differs from candidate plan", kind="ProtocolError", stage="select")
        values = dict(common, block_batching_factor=factor, block_batching_axis=requested_axis,
                      block_batching_grid=tuple(contract["grid"]))
        request = batching.BatchRequest(factor, requested_axis)
        candidate = CandidateCompiler(backend, frozen, env_vars,
                                      prepared=dict(prepared,
                                                    path=str(output))).compile(baseline, values, request, positions,
                                                                               allow_layout_retry=False)
        if candidate.candidate is None:
            declines.append(dict(key=key, reason=candidate.block_batching["reason"]))
            selection["reason"] = "BASELINE_AFTER_CANDIDATE_REJECTION"
            factor = 1
        else:
            result = candidate
    if result is None:
        selection.update(selected_axis=-1, selected_key="unmerged")
        report = dict(compile_status="DISABLED", owner="none", requested_factor=1, applied_factor=1,
                      reason=selection["reason"])
        result = batching.BatchKernel(baseline, None, batching.BatchRequest(), report, [])
    selection.pop("factor")
    result.block_auto_selection = dict(selection, selected_factor=factor, candidate_plan=plan,
                                       compile_rejections=declines)
    return result
