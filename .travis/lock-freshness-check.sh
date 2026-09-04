#!/bin/sh
# A `branch = "..."` dependency names a moving target, but Cargo.lock records the
# single revision resolved when the lock was last written. Nothing else in CI
# compares the two, so the lock can keep pinning a months-old commit of a branch
# that has since advanced -- every build silently uses the stale revision while
# the manifest reads as if it tracks the branch.
#
# Fails with the exact `cargo update -p <crate>` needed to resolve the drift.
set -eu

drift=$(mktemp)
trap 'rm -f "$drift"' EXIT

# Each `source = "git+<url>?branch=<branch>#<rev>"` line in the lock, paired with
# the package name that precedes it.
awk '
    /^name = / { name = $3; gsub(/"/, "", name) }
    /^source = "git\+.*\?branch=/ {
        src = $3; gsub(/"/, "", src)
        split(src, parts, "#")
        rev = parts[2]
        url = substr(parts[1], 5)          # drop the "git+" scheme prefix
        split(url, u, "\\?branch=")
        print name, u[1], u[2], rev
    }
' Cargo.lock | sort -u > "$drift.deps"

if [ ! -s "$drift.deps" ]; then
    echo "error: no git branch dependency found in Cargo.lock."
    echo "  This check exists to guard those; if they are genuinely gone, remove it."
    exit 1
fi

while read -r name url branch locked; do
    head=$(git ls-remote "$url" "refs/heads/$branch" | cut -f1)

    if [ -z "$head" ]; then
        echo "warning: $url has no branch '$branch'; skipping $name" >&2
        continue
    fi

    if [ "$head" != "$locked" ]; then
        echo "error: Cargo.lock pins $name at a stale revision of '$branch'."
        echo "         locked: $locked"
        echo "  branch head is: $head"
        echo "  Fix: cargo update -p $name && git add Cargo.lock"
        echo "$name" >> "$drift"
    else
        echo "ok: $name pins the head of '$branch' ($head)"
    fi
done < "$drift.deps"

rm -f "$drift.deps"
[ ! -s "$drift" ]
