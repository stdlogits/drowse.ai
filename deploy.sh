#!/usr/bin/env bash
# Deploy drowse.ai to Cloudflare Pages (project drowse-web): build the release from a clean, pushed
# HEAD, upload it with Wrangler, then verify that the live site serves that revision.
#
#   ./deploy.sh            release build and its output checks, upload, verify
#   ./deploy.sh --full     also run the runtime-lock and distribution gates; needed after changes to
#                          the browser runtime, its locks or the drowse pin. Takes about two hours:
#                          the distribution preflight downloads and hashes every model artifact (~12 GB)
#   ./deploy.sh --dry-run  build and check only
#
# Wrangler is webui/contact-worker's pinned copy. It needs access to the Cloudflare account that
# hosts drowse.ai: `wrangler login` once, or CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID.

set -euo pipefail
cd "$(dirname "$0")"

full=0
dry_run=0
for arg in "$@"; do
  case "$arg" in
    --full) full=1 ;;
    --dry-run) dry_run=1 ;;
    *) echo "deploy: unknown argument $arg" >&2; exit 2 ;;
  esac
done

# The release records HEAD as its source and links that tree on GitHub, so it must be committed
# and pushed.
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'deploy: commit or stash your changes first; the release records HEAD as its source' >&2
  exit 1
fi
git fetch --quiet origin main
if ! git merge-base --is-ancestor HEAD origin/main; then
  echo 'deploy: push HEAD to origin/main first' >&2
  exit 1
fi
rev=$(git rev-parse HEAD)

# Neuron Lens data is gitignored and built separately (AGENTS.md); an upload without it would take
# the explorer's data off the site.
for model in qwen3-8b gemma-4-12b; do
  if [[ ! -f "webui/public-hosted/neurons-data/$model/index.json" ]]; then
    echo "deploy: webui/public-hosted/neurons-data/$model is missing; build it first (AGENTS.md, Neuron Lens data)" >&2
    exit 1
  fi
done

wrangler="$PWD/webui/contact-worker/node_modules/.bin/wrangler"
[[ -x "$wrangler" ]] || npm ci --prefix webui/contact-worker

cd webui
export DROWSE_PUBLIC_ORIGIN=https://drowse.ai
if [[ "$full" == 1 ]]; then
  DROWSE_RELEASE_REVISION="$rev" npm run build:hosted:release
else
  # build:hosted:release without its runtime-lock and distribution gates. The Vite config keys
  # release mode off the npm script's name, and npx would overwrite it, so call Vite directly.
  npm_lifecycle_event=build:hosted:release ../node_modules/.bin/vite build --config vite.hosted.config.ts
  node ./scripts/prepare-hosted-release.mjs
  node ./scripts/check-build-isolation.mjs ./dist-hosted
  node ./scripts/check-hosted-build.mjs --release
fi

[[ "$dry_run" == 1 ]] && exit 0

# main is the project's production branch.
"$wrangler" pages deploy dist-hosted --project-name drowse-web --branch main --commit-hash "$rev" \
  --commit-message "$(git log -1 --format=%s)" --commit-dirty=false
# A public release check must run without Access credentials.
unset CF_ACCESS_CLIENT_ID CF_ACCESS_CLIENT_SECRET
npm run verify:hosted:deployment -- "$DROWSE_PUBLIC_ORIGIN" --channel release --revision "$rev"
