"""Launch the local browser dashboard when app.py is executed as normal Python."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_if_direct(script_path: str) -> None:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is not None:
        return
    script = Path(script_path).resolve()
    command = [
        sys.executable, "-m", "streamlit", "run", str(script),
        "--server.address=127.0.0.1", "--server.headless=false",
        "--browser.gatherUsageStats=false", *sys.argv[1:],
    ]
    try:
        result = subprocess.run(command, cwd=script.parent, check=False)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(result.returncode)
