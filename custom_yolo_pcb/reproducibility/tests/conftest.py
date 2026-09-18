import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PACKAGE), str(PACKAGE / 'model_code'), str(PACKAGE / 'reproducibility')]
