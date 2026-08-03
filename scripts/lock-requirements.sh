#!/usr/bin/env bash
# Regenerate backend/requirements.lock from backend/requirements.txt.
#
# Resolves against the *container's* platform (CPython 3.11, linux x86_64)
# rather than whatever the developer happens to be running, so the lock is
# valid for the image regardless of who regenerates it. Runs anywhere pip
# runs — no Docker required.
#
#   ./scripts/lock-requirements.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3.11}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

command -v "$PY" >/dev/null 2>&1 || { echo "error: $PY not found (set PYTHON=...)" >&2; exit 1; }

echo "==> Resolving backend/requirements.txt for CPython 3.11 / linux x86_64 …"
"$PY" -m pip install \
  --quiet --disable-pip-version-check --dry-run --ignore-installed --no-input \
  --report "$WORK/report.json" \
  --python-version 3.11 \
  --platform manylinux2014_x86_64 \
  --platform manylinux_2_28_x86_64 \
  --platform manylinux_2_34_x86_64 \
  --only-binary=:all: \
  --target "$WORK/unused" \
  -r "$REPO_ROOT/backend/requirements.txt"

echo "==> Writing backend/requirements.lock …"
"$PY" - "$WORK/report.json" "$REPO_ROOT/backend/requirements.lock" <<'PY'
import io, json, sys

report_path, lock_path = sys.argv[1], sys.argv[2]
with io.open(report_path, encoding="utf-8") as fh:
    report = json.load(fh)

rows = []
for item in report["install"]:
    meta = item["metadata"]
    hashes = item.get("download_info", {}).get("archive_info", {}).get("hashes", {})
    sha = hashes.get("sha256")
    if not sha:
        raise SystemExit(f"error: no sha256 for {meta['name']} (sdist in the resolution?)")
    rows.append((meta["name"].lower(), meta["version"], sha))
rows.sort()

HEADER = """\
# ─────────────────────────────────────────────────────────────────────────────
# backend/requirements.lock  —  GENERATED, DO NOT EDIT BY HAND
#
# Fully-pinned, hash-verified dependency set for the backend container image.
# Resolved for: CPython 3.11 / linux x86_64 (manylinux2014, _2_28, _2_34)
# which is what python:3.11-slim-bookworm provides.
#
# Regenerate after editing requirements.txt:
#   scripts/lock-requirements.sh
#
# Because the hashes are wheel-specific, this file is only installable on
# linux/amd64. Local non-Linux development installs requirements.txt instead.
# ─────────────────────────────────────────────────────────────────────────────
"""

body = "\n".join(f"{n}=={v} \\\n    --hash=sha256:{h}" for n, v, h in rows)
with io.open(lock_path, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(HEADER + body + "\n")
print(f"    {len(rows)} packages pinned")
PY

echo "==> Done. Review the diff, then rebuild: docker compose build backend"
