#!/usr/bin/env bash
# Regenerate backend/requirements.lock from backend/requirements.txt.
#
# Resolves against the *container's* platform (CPython 3.11, linux x86_64)
# rather than whatever the developer happens to be running, so the same lock
# comes out on Windows, macOS and CI. Runs anywhere Python runs — no Docker.
#
#   ./scripts/lock-requirements.sh
#
# ── Why uv and not pip ───────────────────────────────────────────────────────
#
# This used to shell out to `pip install --dry-run --report` with a list of
# --platform manylinux tags. That is not enough: pip's --platform selects which
# *wheels* are eligible, but it still evaluates environment markers against the
# machine running pip. Dependencies guarded by markers therefore came out wrong
# off-Linux — relocking on Windows dropped uvloop (marker: sys_platform !=
# 'win32', wanted by uvicorn[standard]) and added colorama (marker:
# platform_system == 'Windows', useless in the image). The resulting lock built
# an image whose uvicorn silently fell back to the asyncio event loop, and CI's
# freshness check failed for every developer not on Linux.
#
# uv evaluates markers against --python-platform, so the resolution is a
# property of the target, not the host. The version is pinned for the same
# reason ruff is pinned in CI: an upstream release should not silently change
# the lock and turn an unrelated commit red.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3.11}"
UV_VERSION="0.12.1"

command -v "$PY" >/dev/null 2>&1 || { echo "error: $PY not found (set PYTHON=...)" >&2; exit 1; }

# Bootstrap uv into a throwaway location so the script has no prerequisite
# beyond Python itself, and so a uv already on PATH at a different version
# cannot change the output.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> Fetching uv ${UV_VERSION} …"
"$PY" -m pip install --quiet --disable-pip-version-check \
  --target "$WORK/uv" "uv==${UV_VERSION}"

UV="$("$PY" - "$WORK/uv" <<'PY'
import os, sys
root = sys.argv[1]
for name in ("uv.exe", "uv"):
    for sub in ("bin", "Scripts", ""):
        path = os.path.join(root, sub, name)
        if os.path.isfile(path):
            print(path)
            raise SystemExit(0)
raise SystemExit("error: uv binary not found in the install target")
PY
)"

echo "==> Resolving backend/requirements.txt for CPython 3.11 / linux x86_64 …"
# glibc 2.34, not manylinux2014's 2.17: python:3.11-slim-bookworm is Debian 12
# (glibc 2.36), and lightgbm no longer publishes manylinux2014 wheels at all —
# targeting 2.17 makes the resolution unsatisfiable. A higher floor is more
# permissive, not less: wheels tagged for older glibc still match.
"$UV" pip compile \
  "$REPO_ROOT/backend/requirements.txt" \
  --python-version 3.11 \
  --python-platform x86_64-manylinux_2_34 \
  --only-binary :all: \
  --generate-hashes \
  --no-header \
  --no-annotate \
  --quiet \
  --output-file "$WORK/body.txt"

echo "==> Writing backend/requirements.lock …"
"$PY" - "$WORK/body.txt" "$REPO_ROOT/backend/requirements.lock" <<'PY'
import io, sys

body_path, lock_path = sys.argv[1], sys.argv[2]
with io.open(body_path, encoding="utf-8") as fh:
    body = fh.read().strip("\n")

HEADER = """\
# ─────────────────────────────────────────────────────────────────────────────
# backend/requirements.lock  —  GENERATED, DO NOT EDIT BY HAND
#
# Fully-pinned, hash-verified dependency set for the backend container image.
# Resolved for: CPython 3.11 / linux x86_64 (manylinux2014), which is what
# python:3.11-slim-bookworm provides.
#
# Regenerate after editing requirements.txt:
#   scripts/lock-requirements.sh
#
# Because the hashes are wheel-specific, this file is only installable on
# linux/amd64. Local non-Linux development installs requirements.txt instead.
# ─────────────────────────────────────────────────────────────────────────────
"""

# newline="\n" so a Windows checkout does not produce a CRLF lock that CI, which
# regenerates it on Linux, would then report as stale.
with io.open(lock_path, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(HEADER + body + "\n")

count = sum(1 for line in body.splitlines() if "==" in line and not line.startswith(" "))
print(f"    {count} packages pinned")
PY

echo "==> Done. Review the diff, then rebuild: docker compose build backend"
