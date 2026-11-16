from __future__ import annotations

import subprocess
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent


def command_for(script: str, *arguments) -> list[str]:
    return [sys.executable, str(MODULE_DIR / script), *map(str, arguments)]


def run(command: list[str], dry_run: bool = False) -> None:
    print("\n$ " + subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return
    return_code = subprocess.call(command, cwd=str(MODULE_DIR.parent))
    if return_code:
        raise SystemExit(return_code)


def latest_checkpoint(output_root: Path) -> Path:
    candidates = list(output_root.glob("all_output/*/protein_multihead.pt"))
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()

