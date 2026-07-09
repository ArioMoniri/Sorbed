"""Skin-tone estimation for equity-aware confidence and warnings.

Early-stage and deep-tissue injuries are defined largely by color change, which
is substantially harder to see on darker skin — a documented source of later,
higher-stage diagnosis. We estimate the patient's skin tone from healthy skin at
the image border using the Individual Typology Angle (ITA), a standard
colorimetric measure, and coarsen it to two bands so the staging engine can lower
confidence and raise an explicit warning rather than risk a false negative.
"""

from __future__ import annotations

import numpy as np
from skimage import color

from sorbed.domain.enums import SkinToneBand

# ITA (degrees) below this threshold indicates darker skin (tan/brown/dark).
_ITA_DARK_THRESHOLD = 28.0


def estimate_skin_tone(rgb: np.ndarray, wound_mask: np.ndarray | None = None) -> SkinToneBand:
    """Estimate a coarse Fitzpatrick band from healthy border skin."""
    h, w = rgb.shape[:2]
    bh, bw = max(1, int(0.06 * h)), max(1, int(0.06 * w))
    border = np.zeros((h, w), dtype=bool)
    border[:bh, :] = border[-bh:, :] = True
    border[:, :bw] = border[:, -bw:] = True
    if wound_mask is not None:
        border &= ~wound_mask
    if not border.any():
        return SkinToneBand.UNKNOWN

    lab = color.rgb2lab(rgb)
    L = lab[..., 0][border]
    b = lab[..., 2][border]
    # Use the median to resist background clutter in the border ring.
    L_med = float(np.median(L))
    b_med = float(np.median(b))
    if abs(b_med) < 1e-3:
        return SkinToneBand.UNKNOWN
    ita = np.degrees(np.arctan2(L_med - 50.0, b_med))
    return SkinToneBand.IV_VI if ita < _ITA_DARK_THRESHOLD else SkinToneBand.I_III
