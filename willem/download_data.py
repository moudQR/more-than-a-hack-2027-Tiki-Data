"""Download Metrica Sports sample data (games 1 and 2 — the two with events).

These are two full matches at 25 fps with normalized 0-1 coordinates, exactly
like the challenge ATD. The events CSVs let us derive frame-level possession
labels so we can train a real supervised classifier rather than guess.

License: CC BY-NC-SA 4.0, published by Metrica Sports for research use.
"""
import urllib.request
from pathlib import Path

from config import TRAINING_DIR

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"

# Only games 1 and 2 ship an events file; game 3 is tracking-only, so skip it.
GAMES = (1, 2)

FILES = {
    "events": "RawEventsData.csv",
    "home": "RawTrackingData_Home_Team.csv",
    "away": "RawTrackingData_Away_Team.csv",
}


def _fetch(url: str, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url}")
    urllib.request.urlretrieve(url, out)


def download() -> list[Path]:
    downloaded = []
    for game in GAMES:
        print(f"Sample_Game_{game}")
        folder = TRAINING_DIR / f"Sample_Game_{game}"
        prefix = f"Sample_Game_{game}"
        for key, suffix in FILES.items():
            url = f"{BASE}/Sample_Game_{game}/{prefix}_{suffix}"
            out = folder / f"{key}.csv"
            try:
                _fetch(url, out)
                downloaded.append(out)
            except Exception as exc:
                print(f"  failed: {url}\n    {exc}")

    print(f"\nDownloaded {len(downloaded)} files to {TRAINING_DIR}")
    return downloaded


if __name__ == "__main__":
    download()