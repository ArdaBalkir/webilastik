"""
Prediction tile encoding shared by server.py and headless_cli.py.

Color map: label integer → (R, G, B, A) uint8.
Label IDs come from the classifier's classes_ array (sklearn preserves order).
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np
from PIL import Image

# ── Color map ─────────────────────────────────────────────────────────────────
# Maps label IDs to RGBA colors.  Add entries here as new labels are needed.
# Background is fully transparent so the source image shows through.

_LABEL_COLORS: dict[int, tuple[int, int, int, int]] = {
    1: (255, 0, 0, 200),  # label 1 → red
    2: (0, 0, 0, 200),  # label 2 → black
    3: (50, 120, 220, 200),  # label 3 → blue
    4: (50, 200, 80, 200),  # label 4 → green
    5: (220, 160, 0, 200),  # label 5 → orange
    6: (160, 50, 220, 200),  # label 6 → purple
    7: (0, 200, 200, 200),  # label 7 → cyan
    8: (220, 220, 50, 200),  # label 8 → yellow
}

_UNKNOWN_COLOR: tuple[int, int, int, int] = (128, 128, 128, 200)  # grey fallback
_TRANSPARENT: tuple[int, int, int, int] = (0, 0, 0, 0)


def encode_prediction_png(
    proba: np.ndarray,
    class_labels: Sequence[int],
) -> bytes:
    """
    Convert an (H, W, n_classes) float32 probability array to a colored RGBA PNG.

    Each pixel is colored by its argmax class.  The color is looked up from
    _LABEL_COLORS by the *label ID* (the integer the user gave each class),
    not the class index inside proba.

    Args:
        proba:        (H, W, n_classes) float32, rows sum to 1.
        class_labels: sequence of n_classes label IDs, same order as proba axis 2.
                      Typically clf.classes_.tolist().
    """
    h, w, _ = proba.shape
    argmax = np.argmax(proba, axis=2)  # (H, W) — index into class_labels
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for idx, label_id in enumerate(class_labels):
        color = _LABEL_COLORS.get(int(label_id), _UNKNOWN_COLOR)
        mask = argmax == idx
        rgba[mask] = color

    img = Image.fromarray(rgba)  # RGBA inferred from (H, W, 4) shape
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
