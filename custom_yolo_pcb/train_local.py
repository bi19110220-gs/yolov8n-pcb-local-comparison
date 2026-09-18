"""Train the original model first, followed by the preserved Trial 044 recipe."""
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'model_code'))
from tools.run_local_vscode_comparison import main

if __name__ == '__main__':
    raise SystemExit(main())
