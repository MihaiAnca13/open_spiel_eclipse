#!/usr/bin/env bash
# Record the checksums of the auto-research IMMUTABLE files into immutables.sha.
# The agent must never modify these; bench.sh voids any run whose manifest entry
# does not match, so there is never credit for tampering.
#
# IMMUTABLE SET: bench.sh (the runner/throughput scorer), gate_audit.sh (the
# fresh-context honesty gate), loop.sh (the fresh-session driver that
# orchestrates bench+gate). frozen_eval.py no longer exists (pure-throughput
# scoring). Run this ONCE per codebase state, and treat the resulting
# immutables.sha as sacred.  Do not regenerate it to "approve" a change.
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AR="$ROOT/autoresearch"
OUT="$AR/immutables.sha"
: > "$OUT"
for f in bench.sh gate_audit.sh loop.sh; do
  h=$(sha256sum "$AR/$f" | awk '{print $1}')
  printf '%s  %s\n' "$h" "$AR/$f" >> "$OUT"
done
echo "wrote $OUT:"
cat "$OUT"
