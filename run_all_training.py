"""
run_all_training.py
====================
Clears the model registry and runs all 6 training scripts (train-1.py through train-6.py)
sequentially so every script gets a clean, correctly numbered variant slot.

Usage:
    python run_all_training.py

Each training run will:
  - Reserve the next variant_N slot in registry.json
  - Train CatBoost + LightGBM + XGBoost on its dataset
  - Save artifacts to model_registry/variant_N/
  - Auto-promote the best variant to default

Final result: variant_1 (train-1.py) ... variant_6 (train-6.py) all visible in the UI.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT         = Path(__file__).resolve().parent
REGISTRY_DIR = ROOT / "model_registry"
ML_DIR       = ROOT / "ml_training"

SCRIPTS = [
    ML_DIR / "train-1.py",
    ML_DIR / "train-2.py",
    ML_DIR / "train-3.py",
    ML_DIR / "train-4.py",
    ML_DIR / "train-5.py",
    ML_DIR / "train-6.py",
]

DIV = "=" * 80


def clean_registry() -> None:
    """Delete all existing variant folders and registry.json to start fresh."""
    print(DIV)
    print("CLEANING MODEL REGISTRY")
    print(DIV)

    if REGISTRY_DIR.exists():
        for item in REGISTRY_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  Deleted: {item.name}/")
            elif item.name == "registry.json":
                item.unlink()
                print(f"  Deleted: registry.json")
    else:
        REGISTRY_DIR.mkdir(parents=True)

    print("  Registry cleaned -- ready for fresh training runs.\n")


def run_script(script: Path, idx: int, total: int) -> bool:
    """Run a training script as a subprocess and stream its output."""
    print(DIV)
    print(f"[{idx}/{total}]  RUNNING: {script.name}")
    print(DIV)

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  OK  {script.name} completed in {elapsed/60:.1f} min\n")
        return True
    else:
        print(f"\n  FAILED  {script.name} (exit code {result.returncode}) after {elapsed/60:.1f} min\n")
        return False


def main() -> None:
    print(DIV)
    print("ALL-TRAINING RUNNER")
    print(f"  Scripts : {len(SCRIPTS)}")
    print(f"  Registry: {REGISTRY_DIR}")
    print(DIV + "\n")

    missing = [s for s in SCRIPTS if not s.exists()]
    if missing:
        print("ERROR -- missing scripts:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    clean_registry()

    results: list[tuple[str, bool]] = []
    total = len(SCRIPTS)

    for idx, script in enumerate(SCRIPTS, 1):
        success = run_script(script, idx, total)
        results.append((script.name, success))

    print(DIV)
    print("TRAINING SUMMARY")
    print(DIV)
    for name, ok in results:
        status = "OK   " if ok else "FAILED"
        print(f"  {status}  {name}")

    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n  {len(failed)} script(s) failed -- check logs above.")
        sys.exit(1)
    else:
        print(f"\n  All {total} training runs completed successfully!")
        print(f"  Open the UI to compare variant_1 through variant_{total}.")


if __name__ == "__main__":
    main()
