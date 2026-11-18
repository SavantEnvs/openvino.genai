#!/usr/bin/env bash
#
# mayhem/test.sh — RUNS upstream's own gtest suite (tests/cpp, built with normal flags by mayhem/build.sh).
# Asserts behavior: every test is a gtest assertion; the pass/fail counts are parsed from the runner's own
# summary, and a run that prints no summary (e.g. an early exit(0)) fails. Does not build.
set -uo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
: "${MAYHEM_JOBS:=$(nproc)}"   # build parallelism; env-overridable, falls back to nproc (use -j"$MAYHEM_JOBS")
: "${SRC:=/mayhem}"
cd "$SRC"

# emit_ctrf <tool> <passed> <failed> [skipped] [pending] [other]
# Writes a CTRF report (file + stdout `CTRF {...}` marker) and returns non-zero iff failed>0.
emit_ctrf() {
  local tool="$1" passed="$2" failed="$3" skipped="${4:-0}" pending="${5:-0}" other="${6:-0}"
  local tests=$(( passed + failed + skipped + pending + other ))
  cat > "${CTRF_REPORT:-$SRC/ctrf-report.json}" <<JSON
{
  "results": {
    "tool": { "name": "$tool" },
    "summary": {
      "tests": $tests,
      "passed": $passed,
      "failed": $failed,
      "pending": $pending,
      "skipped": $skipped,
      "other": $other
    }
  }
}
JSON
  printf 'CTRF {"results":{"tool":{"name":"%s"},"summary":{"tests":%d,"passed":%d,"failed":%d,"pending":%d,"skipped":%d,"other":%d}}}\n' \
    "$tool" "$tests" "$passed" "$failed" "$pending" "$skipped" "$other"
  [ "$failed" -eq 0 ]
}


SUITE="$SRC/build-tests/tests/cpp/tests_continuous_batching"
[ -x "$SUITE" ] || { echo "missing $SUITE: mayhem/build.sh did not build it" >&2; emit_ctrf gtest 0 1 0; exit 1; }

# Model-free subset: tokenizers are not built (BUILD_TOKENIZERS=OFF; the submodule is empty), so the tests that
# need a Tokenizer, and the placeholders for the CACHE_TYPES_CSV real-model suites, are excluded.
FILTER='-AddSecondInputTest.*:*AddSecondInputStatefulTruncationTest.*:GGUFTokenizer.*:GoogleTestVerification.*'

OUT=$(mktemp)
timeout 1500 "$SUITE" --gtest_filter="$FILTER" >"$OUT" 2>&1
rc=$?
tail -n 5 "$OUT"

passed=$(sed -n 's/^\[  PASSED  \] \([0-9]*\) tests\?\..*/\1/p' "$OUT" | tail -n1)
failed=$(sed -n 's/^\[  FAILED  \] \([0-9]*\) tests\?, listed below.*/\1/p' "$OUT" | tail -n1)
skipped=$(sed -n 's/^\[  SKIPPED \] \([0-9]*\) tests\?, listed below.*/\1/p' "$OUT" | tail -n1)
ran=$(sed -n 's/^\[==========\] \([0-9]*\) tests\? from .* ran\..*/\1/p' "$OUT" | tail -n1)
rm -f "$OUT"
passed=${passed:-0}; failed=${failed:-0}; skipped=${skipped:-0}
if [ -z "$ran" ] || [ "$passed" -lt 100 ]; then            # no summary / far too few tests: the runner did not really run
  emit_ctrf gtest "$passed" $(( failed + 1 )) "$skipped"; exit 1
fi
[ "$rc" -eq 0 ] || failed=$(( failed > 0 ? failed : 1 ))
emit_ctrf gtest "$passed" "$failed" "$skipped"
