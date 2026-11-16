"""Force D on set-piece intervals from the Smart Tagging XML.

The tagging marks a window around each restart rather than the instant, so we
mark the entire interval as D. Set pieces are exactly the dead-ball stretches
the possession label is supposed to identify, and they are the one place where
this external signal is unambiguous.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SET_PIECE_CODES = {"set pieces"}


def load_set_piece_mask(xml_path: Path, fps: int, n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for instance in ET.parse(xml_path).iter("instance"):
        code = (instance.findtext("code") or "").strip().lower()
        if code not in SET_PIECE_CODES:
            continue
        s = int(round(float(instance.findtext("start")) * fps))
        e = int(round(float(instance.findtext("end")) * fps))
        # A set piece is a 4-6 s window around the restart, not 30 s. Cap it —
        # without this we masked 27.7% of graded frames as D.
        if e - s > 6 * fps:
            e = s + 6 * fps
        mask[max(0, s): min(n_frames, e)] = True
    return mask