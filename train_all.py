# train_all.py
"""
Master training script — runs all FL methods sequentially.

Edit the USER CONFIGURATION section below, then:
    python train_all.py
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import time
from datetime import timedelta
from typing import Dict

# ════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ← the only section you need to edit
# ════════════════════════════════════════════════════════════════════

NUM_ROUNDS = 200    # training rounds applied to every method

RUN: Dict[str, bool] = {
    "FedAvg":      True,
    "FedProx":     True,
    "FedNova":     True,
    "FedAvg_DRL":  True,
    "FedProx_DRL": True,
}

RUN_RESULTS_AFTER = True   # run results.py automatically when training finishes

# ════════════════════════════════════════════════════════════════════

_BAR  = "=" * 64
_BAR2 = "-" * 64


def _fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _build_overrides(num_rounds: int) -> Dict[str, Dict]:
    """
    Attributes patched on each module before calling its main().
    SELECTOR_BUFFER_SIZE is kept equal to NUM_ROUNDS so the DDPG replay
    buffer spans the full training horizon.
    """
    return {
        "FedAvg":      {"NUM_ROUNDS": num_rounds},
        "FedProx":     {"NUM_ROUNDS": num_rounds},
        "FedNova":     {"NUM_ROUNDS": num_rounds},
        "FedAvg_DRL":  {"NUM_ROUNDS": num_rounds, "SELECTOR_BUFFER_SIZE": num_rounds},
        "FedProx_DRL": {"NUM_ROUNDS": num_rounds, "SELECTOR_BUFFER_SIZE": num_rounds},
    }


def run_method(name: str, overrides: Dict) -> tuple[bool, float]:
    """
    Import the named module, patch its constants, call main().
    Returns (success, elapsed_seconds).
    """
    print(f"\n{_BAR}")
    print(f"  Starting : {name}   (NUM_ROUNDS={NUM_ROUNDS})")
    print(_BAR)

    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(name)
        for attr, val in overrides.items():
            setattr(mod, attr, val)
        mod.main()
        elapsed = time.perf_counter() - t0
        print(f"\n  {name} finished in {_fmt(elapsed)}")
        return True, elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n  {name} FAILED after {_fmt(elapsed)}: {exc}", file=sys.stderr)
        return False, elapsed


def main():
    active    = [name for name, enabled in RUN.items() if enabled]
    overrides = _build_overrides(NUM_ROUNDS)

    print(_BAR)
    print("  FL-DRL Master Training Run")
    print(_BAR2)
    print(f"  NUM_ROUNDS  = {NUM_ROUNDS}")
    print(f"  Methods     = {active}")
    print(f"  Results     = {'yes' if RUN_RESULTS_AFTER else 'no'}")
    print(_BAR)

    statuses: Dict[str, tuple[bool, float]] = {}
    wall_start = time.perf_counter()

    for name in active:
        statuses[name] = run_method(name, overrides[name])

    wall_total = time.perf_counter() - wall_start

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{_BAR}")
    print("  Training Summary")
    print(_BAR2)
    for name, (ok, elapsed) in statuses.items():
        tag = "OK    " if ok else "FAILED"
        print(f"  {tag}  {name:<14}  {_fmt(elapsed):>10}")
    print(_BAR2)
    print(f"  Total wall time: {_fmt(wall_total)}")
    print(_BAR)

    any_succeeded = any(ok for ok, _ in statuses.values())

    if any_succeeded and RUN_RESULTS_AFTER:
        print(f"\n{_BAR}")
        print("  Generating result plots")
        print(_BAR)
        ret = subprocess.run(
            [sys.executable, "results.py", "--no-model-eval"],
            check=False,
        )
        if ret.returncode != 0:
            print("  results.py exited with errors — run manually to investigate.")
    elif not any_succeeded:
        print("\n  All methods failed — skipping results generation.")

    print(f"\n{_BAR}")
    print("  Done.")
    print(_BAR)


if __name__ == "__main__":
    main()
