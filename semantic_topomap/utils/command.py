from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"[cmd] {printable}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)
