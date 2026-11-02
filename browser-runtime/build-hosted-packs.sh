#!/bin/bash
# Build, assemble and validate one hosted model's lens, units and core-statistics packs
# from its native fits (README.md, "Lens, units and core packs").  Rerunnable: the costly
# part, the units contrast statistics, is cached by drowse in $DROWSE_HOME.
#
# Usage:  build-hosted-packs.sh MODEL_ID
# Env:    DROWSE_HOME   the home holding the model's native fits (required)
#         PARITY_ROOT   ~/.drowse-parity: exports/<id> holds the q4 models' browser weights
#         PACKS         $PARITY_ROOT/packs: output goes to $PACKS/<id>/<runtime sha256[:16]>
#         RUNTIME_LOCK  browser-runtime/runtime-lock.json: the identity the packs bind
#         PY            python with torch and safetensors (drowse is the submodule's)
#         DEVICE        cpu (default) | mps | cuda;  THREADS  torch threads (default 8)
#         KINDS         jlens,slens,units,core
# Exit:   0 when every pack built and validated.
set -euo pipefail
ID="${1:?usage: build-hosted-packs.sh MODEL_ID}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
: "${DROWSE_HOME:?set DROWSE_HOME to the home holding the native fits}"
export DROWSE_HOME
ROOT="${PARITY_ROOT:-$HOME/.drowse-parity}"
PACKS="${PACKS:-$ROOT/packs}"
LOCK="${RUNTIME_LOCK:-$REPO/browser-runtime/runtime-lock.json}"
PY="${PY:-python3}"
KINDS="${KINDS:-jlens,slens,units,core}"
SCRIPTS="$REPO/webui/scripts"
mkdir -p "$PACKS/$ID"
IDENTITY="$(mktemp "$PACKS/$ID/runtime-identity.XXXXXX")"
trap 'rm -f "$IDENTITY"' EXIT
node "$SCRIPTS/hosted-runtime-identity.mjs" "$ID" --runtime-lock "$LOCK" --output "$IDENTITY"
SHA="$(node -e 'process.stdout.write(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).runtimeIdentitySha256)' "$IDENTITY")"
QUANT="$(node -e 'process.stdout.write(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).runtimeIdentity.quantization)' "$IDENTITY")"
if [ "$QUANT" = q0f32 ]; then
  MODEL="$(node -e 'process.stdout.write(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).runtimeIdentity.sourceModel)' "$IDENTITY")"
else
  MODEL="$ROOT/exports/$ID"
  [ -f "$MODEL/DROWSE-PARITY-EXPORT.json" ] || { echo "no verified export at $MODEL" >&2; exit 1; }
fi
OUT="$PACKS/$ID/${SHA:0:16}"
if [ -e "$OUT" ]; then echo "$OUT exists; remove it to rebuild" >&2; exit 1; fi
echo "=== [$ID] packs for runtime $SHA from $MODEL ($KINDS)"
PYTHONPATH="$REPO/drowse${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 "$PY" "$HERE/build_hosted_packs.py" "$MODEL" \
  --runtime-identity "$IDENTITY" --output "$OUT" --kinds "$KINDS" \
  --device "${DEVICE:-cpu}" --threads "${THREADS:-8}" > /dev/null
cp "$IDENTITY" "$OUT/runtime-identity.json"
for KIND in jlens slens units; do
  [ -d "$OUT/$KIND" ] && node "$SCRIPTS/assemble-hosted-pack.mjs" "$OUT/$KIND" > /dev/null
done
for KIND in jlens slens units; do
  [ -d "$OUT/$KIND" ] || continue
  WITH=()
  # A units pack reads through the R and S packs it binds.
  [ "$KIND" = units ] && WITH=(--with "$OUT/jlens" --with "$OUT/slens")
  node "$SCRIPTS/validate-hosted-instrument-pack.mjs" "$ID" "$KIND" "$OUT/$KIND" ${WITH[@]+"${WITH[@]}"} \
    --runtime-lock "$LOCK" > "$OUT/$KIND.validation.json"
done
node "$SCRIPTS/hosted-pack-sizes.mjs" "$OUT" "$ID" --runtime-lock "$LOCK" > "$OUT/sizes.json"
cat "$OUT/sizes.json"
echo "=== [$ID] done: $OUT"
