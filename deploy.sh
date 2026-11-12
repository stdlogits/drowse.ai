#!/usr/bin/env bash
# Deploy drowse.ai to Cloudflare Pages (project drowse-web): build the release from a clean, pushed
# HEAD, upload it with Wrangler, then verify that the live site serves that revision.
#
#   ./deploy.sh            the release build (the locks, the signed catalog at the distribution
#                          lock's URL, the build and its output checks), upload, verify
#   ./deploy.sh --dry-run  build and check only
#
# Wrangler is webui/contact-worker's pinned copy. It needs access to the Cloudflare account that
# hosts drowse.ai: `wrangler login` once, or CLOUDFLARE_API_TOKEN. CLOUDFLARE_ACCOUNT_ID defaults
# to that account.

set -euo pipefail
cd "$(dirname "$0")"

dry_run=0
for arg in "$@"; do
  case "$arg" in
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


wrangler="$PWD/webui/contact-worker/node_modules/.bin/wrangler"
[[ -x "$wrangler" ]] || npm ci --prefix webui/contact-worker

# A login that reaches several accounts cannot pick one non-interactively, so name the one that
# hosts drowse-web, and check access now rather than after the build.
export CLOUDFLARE_ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-18903e07043611bf77d67976b4bbc530}"
if [[ "$dry_run" == 0 ]] &&
  ! "$wrangler" pages deployment list --project-name drowse-web >/dev/null 2>&1; then
  echo 'deploy: cannot reach the drowse-web Pages project; check the Cloudflare login and account' >&2
  exit 1
fi

cd webui
export DROWSE_PUBLIC_ORIGIN=https://drowse.ai
# The build reads the catalog at the distribution lock's URL (a runtime generation's own path, such
# as v2/catalog.json), so build:hosted:release checks the signed catalog there before building.
npm run build:hosted:release

[[ "$dry_run" == 1 ]] && exit 0

# main is the project's production branch.
"$wrangler" pages deploy dist-hosted --project-name drowse-web --branch main --commit-hash "$rev" \
  --commit-message "$(git log -1 --format=%s)" --commit-dirty=false
# A public release check must run without Access credentials.
unset CF_ACCESS_CLIENT_ID CF_ACCESS_CLIENT_SECRET
# The custom domain can serve the previous deployment for a little while after the upload.
for attempt in 1 2 3 4 5 6; do
  npm run --silent verify:hosted:deployment -- "$DROWSE_PUBLIC_ORIGIN" --channel release --revision "$rev" && exit 0
  (( attempt < 6 )) && { echo "Live site not on $rev yet; checking again in 20 s"; sleep 20; }
done
exit 1
