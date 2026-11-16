"""All constants and paths for the solution. One place to change anything."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data"))
TACTICAL_DIR = DATA_DIR / "tactical"
TRAINING_DIR = Path(__file__).resolve().parent / "external" / "metrica"
MODELS_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = ROOT / "evaluation" / "results"

FPS = 25
N_FRAMES = 199_319
PITCH = (105.3, 68.0)

HALF1_START, HALF1_END = 11_185, 90_651
HALF2_START, HALF2_END = 110_808, 190_197

LABELS = ("H", "A", "D")