"""Container health probe for the PriceRef ML API.

A script rather than a `curl` one-liner because python:*-slim ships neither curl
nor wget, and adding either just for a probe grows the image and the CVE surface.

Exits 0 only when the API reports ok AND a model is actually loaded — a process
that is listening but failed to load its ensemble is not healthy, and letting it
pass the probe would put it into the load-balancer rotation serving errors.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 8


def main() -> int:
    port = os.environ.get("PORT", "8000")
    url = f"http://127.0.0.1:{port}/health"

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                print(f"unhealthy: HTTP {response.status}", file=sys.stderr)
                return 1
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"unhealthy: {exc.reason}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"unhealthy: {exc}", file=sys.stderr)
        return 1

    if payload.get("status") != "ok":
        print(f"unhealthy: status={payload.get('status')!r}", file=sys.stderr)
        return 1

    if not payload.get("model_loaded"):
        print("unhealthy: model_loaded is false", file=sys.stderr)
        return 1

    expected = os.environ.get("ACTIVE_VARIANT_ID")
    actual = payload.get("active_variant")
    if expected and actual != expected:
        # Serving a different model than the one this revision was deployed with
        # means silently wrong prices, so treat it as a failure, not a warning.
        print(f"unhealthy: serving {actual!r}, expected {expected!r}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
