#!/usr/bin/env bash
#
# Build and run the Scalismo SSM project (nako.ribs.RibRegistration).
#
# If the repo path contains ':' (e.g. a Nextcloud mount with :8082),
# sbt fails because it treats colons as classpath separators.  In that
# case the script copies the source to a colon-free cache directory
# first; otherwise it runs sbt directly from the repo.
#
# Usage:
#   ./sbt_build.sh compile
#   ./sbt_build.sh "runMain nako.ribs.RibRegistration --input ... --output ..."
#   ./sbt_build.sh                   # interactive sbt shell

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "$SCRIPT_DIR" == *":"* ]]; then
    BUILD_DIR="$HOME/.cache/nako_ribs_scalismo"
    mkdir -p "$BUILD_DIR"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete \
            --exclude 'target/' \
            --exclude 'project/target/' \
            "$SCRIPT_DIR/" "$BUILD_DIR/"
    else
        find "$BUILD_DIR" -maxdepth 1 \
            -not -name target -not -name project -not -name '.' \
            -exec rm -rf {} + 2>/dev/null || true
        cp -R "$SCRIPT_DIR/." "$BUILD_DIR/"
    fi
    echo "[sbt_build] Colon in path detected – building from $BUILD_DIR"
    cd "$BUILD_DIR"
else
    cd "$SCRIPT_DIR"
fi

echo "[sbt_build] Running: sbt $*"
exec sbt "$@"
