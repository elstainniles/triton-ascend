#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
result_file="$script_dir/TEST_RESULT.md"
details_dir="$script_dir/test_results"
manifest_file="$details_dir/90_manifest.tsv"
device_id="${NPU_DEVICE:-0}"
python_bin="${PYTHON:-python}"
timeout_seconds="${POINTER_BITCAST_TEST_TIMEOUT:-3600}"
cache_dir="${TRITON_CACHE_DIR:-$repo_root/.cache/pointer-bitcast-e2e}"

write_setup_failure() {
  local message="$1"
  local status="${2:-2}"
  mkdir -p "$details_dir"
  {
    echo "# Pointer Bitcast E2E Result"
    echo
    echo "## Overall"
    echo
    echo "**SETUP ERROR**"
    echo
    echo "- Diagnostic: $message"
    echo "- Repository: $repo_root"
    echo "- Device: $device_id"
    echo
    echo "Send this file first. Additional diagnostic files are described in"
    echo "test_results/README.md."
  } > "$result_file"
  printf '%s\n' "$message" > "$details_dir/00_setup_error.log"
  echo
  cat "$result_file"
  echo
  echo "Summary: $result_file"
  echo "Setup log: $details_dir/00_setup_error.log"
  exit "$status"
}

if ! command -v "$python_bin" >/dev/null 2>&1; then
  write_setup_failure "Python executable was not found: $python_bin"
fi

if ! command -v timeout >/dev/null 2>&1; then
  write_setup_failure "The GNU timeout command is required but was not found."
fi

shopt -s nullglob
python_builds=("$repo_root"/build/lib.*)
triton_opt_builds=(
  "$repo_root"/build/cmake.*/bin/triton-opt
  "$repo_root"/build/lib.*/triton/_C/triton-opt
)
shopt -u nullglob

for build_dir in "${python_builds[@]}"; do
  if [[ -d "$build_dir/triton" ]]; then
    export PYTHONPATH="$build_dir${PYTHONPATH:+:$PYTHONPATH}"
    break
  fi
done

if [[ -z "${TRITON_OPT:-}" ]]; then
  if command -v triton-opt >/dev/null 2>&1; then
    TRITON_OPT="$(command -v triton-opt)"
  else
    for candidate in "${triton_opt_builds[@]}"; do
      if [[ -x "$candidate" ]]; then
        TRITON_OPT="$candidate"
        break
      fi
    done
  fi
fi

if [[ -z "${TRITON_OPT:-}" || ! -x "$TRITON_OPT" ]]; then
  write_setup_failure "triton-opt was not found; build the repository first or set TRITON_OPT."
fi

export TRITON_OPT
export TRITON_CACHE_DIR="$cache_dir"
export ASCEND_RT_VISIBLE_DEVICES="$device_id"
export ASCEND_VISIBLE_DEVICES="$device_id"
export NPU_VISIBLE_DEVICES="$device_id"
mkdir -p "$TRITON_CACHE_DIR" "$details_dir"
rm -f "$details_dir/00_setup_error.log"

started_at="$(date -Iseconds)"
started_epoch="$(date +%s)"
commit="$(git -C "$repo_root" rev-parse HEAD)"
environment_file="$details_dir/00_environment.txt"

{
  echo "commit=$commit"
  echo "started=$started_at"
  echo "device=$device_id"
  echo "python=$python_bin"
  echo "PYTHONPATH=${PYTHONPATH:-}"
  echo "TRITON_OPT=$TRITON_OPT"
  echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "ASCEND_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES"
  echo "NPU_VISIBLE_DEVICES=$NPU_VISIBLE_DEVICES"
  echo
  uname -a || true
  "$python_bin" --version || true
  "$python_bin" -c '
import importlib

for name in ("torch", "torch_npu", "triton"):
    try:
        module = importlib.import_module(name)
        print(f"{name}={getattr(module, '\''__version__'\'', '\''unknown'\'')}")
        print(f"{name}_module={getattr(module, '\''__file__'\'', '\''unknown'\'')}")
    except Exception as exc:
        print(f"{name}_import_error={type(exc).__name__}: {exc}")

try:
    from triton._C import libtriton
    print(f"libtriton={libtriton.__file__}")
except Exception as exc:
    print(f"libtriton_import_error={type(exc).__name__}: {exc}")
' || true
  echo
  git -C "$repo_root" status --short --branch || true
} > "$environment_file" 2>&1

printf 'name\tdisplay_name\texpected\texit_status\txml_file\tlog_file\n' > "$manifest_file"

groups=(
  'e2e|Real-world pointer bitcast|3|test_pointer_bitcast_e2e.py|10_e2e'
  'matrix|Type conversion matrix|15|test_pointer_bitcast_e2e_matrix.py|20_matrix'
  'runtime_alignment|Runtime alignment checks|17|test_pointer_bitcast_e2e_runtime_alignment.py|30_runtime_alignment'
  'mlir_contract|TritonToLinalg MLIR contracts|11|test_pointer_bitcast_mlir_contract.py|40_mlir_contract'
)

echo "Running 46 pointer bitcast tests in 4 diagnostic groups."
echo "Detailed output is written under: $details_dir"

group_index=0
for group_spec in "${groups[@]}"; do
  IFS='|' read -r group_name display_name expected test_file result_stem <<< "$group_spec"
  group_index=$((group_index + 1))
  log_file="$details_dir/${result_stem}.log"
  xml_file="$details_dir/${result_stem}.xml"
  rm -f "$xml_file"
  elapsed_seconds=$(( $(date +%s) - started_epoch ))
  remaining_seconds=$(( timeout_seconds - elapsed_seconds ))

  echo "[$group_index/${#groups[@]}] $display_name"
  group_status=0
  if (( remaining_seconds <= 0 )); then
    {
      echo "group=$group_name"
      echo "expected_tests=$expected"
      echo "test_file=$script_dir/$test_file"
      echo "ERROR: the overall $timeout_seconds second timeout was exhausted before this group started."
    } > "$log_file"
    group_status=124
  else
    {
      echo "group=$group_name"
      echo "expected_tests=$expected"
      echo "test_file=$script_dir/$test_file"
      echo "remaining_timeout_seconds=$remaining_seconds"
      echo
      timeout "$remaining_seconds" "$python_bin" -m pytest \
        -q -ra --tb=long --junitxml="$xml_file" "$script_dir/$test_file"
    } > "$log_file" 2>&1 || group_status=$?
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$group_name" "$display_name" "$expected" "$group_status" \
    "$(basename "$xml_file")" "$(basename "$log_file")" \
    >> "$manifest_file"
done

finished_at="$(date -Iseconds)"
summary_builder_log="$details_dir/99_summary_builder.log"
: > "$result_file"

set +e
"$python_bin" "$script_dir/summarize_results.py" \
  --manifest "$manifest_file" \
  --details-dir "$details_dir" \
  --output "$result_file" \
  --commit "$commit" \
  --started "$started_at" \
  --finished "$finished_at" \
  --device "$device_id" \
  --timeout "$timeout_seconds" \
  > "$summary_builder_log" 2>&1
test_status=$?
set -e

if [[ ! -s "$result_file" ]]; then
  write_setup_failure \
    "Result summarization failed; inspect test_results/99_summary_builder.log." \
    "$test_status"
fi

echo
cat "$result_file"
echo
echo "Detailed result index: $details_dir/README.md"
echo "No Git push is required; provide TEST_RESULT.md first."
exit "$test_status"
