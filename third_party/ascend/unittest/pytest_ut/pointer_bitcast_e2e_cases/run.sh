#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
result_file="$script_dir/TEST_RESULT.md"
device_id="${NPU_DEVICE:-0}"
python_bin="${PYTHON:-python}"
timeout_seconds="${POINTER_BITCAST_TEST_TIMEOUT:-3600}"
cache_dir="${TRITON_CACHE_DIR:-$repo_root/.cache/pointer-bitcast-e2e}"

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
  echo "ERROR: triton-opt was not found; build the repository first or set TRITON_OPT." >&2
  exit 2
fi

export TRITON_OPT
export TRITON_CACHE_DIR="$cache_dir"
export ASCEND_RT_VISIBLE_DEVICES="$device_id"
export ASCEND_VISIBLE_DEVICES="$device_id"
export NPU_VISIBLE_DEVICES="$device_id"
mkdir -p "$TRITON_CACHE_DIR"

tmp_output="$(mktemp)"
trap 'rm -f "$tmp_output"' EXIT
started_at="$(date -Iseconds)"
commit="$(git -C "$repo_root" rev-parse HEAD)"

{
  echo "commit=$commit"
  echo "device=$device_id"
  echo "python=$python_bin"
  echo "PYTHONPATH=${PYTHONPATH:-}"
  echo "TRITON_OPT=$TRITON_OPT"
  echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
  "$python_bin" -c 'import triton; from triton._C import libtriton; print(f"triton={triton.__version__}"); print(f"triton_module={triton.__file__}"); print(f"libtriton={libtriton.__file__}")'
  echo
} | tee "$tmp_output"

set +e
timeout "$timeout_seconds" "$python_bin" -m pytest -q -ra "$script_dir" 2>&1 | tee -a "$tmp_output"
test_status=${PIPESTATUS[0]}
set -e
finished_at="$(date -Iseconds)"

{
  echo "# Pointer Bitcast E2E Result"
  echo
  echo "- Commit: \`$commit\`"
  echo "- Started: \`$started_at\`"
  echo "- Finished: \`$finished_at\`"
  echo "- Device: \`$device_id\`"
  echo "- Exit status: \`$test_status\`"
  echo "- Timeout: \`${timeout_seconds}s\`"
  echo
  echo '```text'
  cat "$tmp_output"
  echo '```'
} > "$result_file"

echo
echo "Result written to $result_file"
exit "$test_status"
