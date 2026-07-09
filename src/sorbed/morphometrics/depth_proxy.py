"""A shading-based *relative* depth cue — explicitly not a measurement.

True depth cannot be recovered from a single 2D photograph. This returns a
monotone, unitless index in [0, 1]: deeper-looking wounds cast interior shadow,
so the interior sits darker than the rim. It is used only as weak, clearly-
flagged evidence and never as a physical depth. ``is_physical_measurement`` is
always ``False``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import color

from sorbed.domain.metrics import DepthProxy

_METHOD = "shading_gradient_v1"


def shading_depth_proxy(rgb: np.ndarray, wound_mask: np.ndarray) -> DepthProxy | None:
    """Relative interior-vs-rim darkening of the wound, in [0, 1]."""
    if not wound_mask.any():
        return None
    lightness = color.rgb2lab(rgb)[..., 0] / 100.0

    distance = ndimage.distance_transform_edt(wound_mask)
    if distance.max() <= 0:
        return None
    rim = wound_mask & (distance <= max(1.0, 0.15 * distance.max()))
    interior = wound_mask & (distance >= 0.6 * distance.max())
    if not rim.any() or not interior.any():
        return None

    rim_l = float(lightness[rim].mean())
    interior_l = float(lightness[interior].mean())
    index = (rim_l - interior_l) / (rim_l + 1e-6)
    index = float(np.clip(index, 0.0, 1.0))
    return DepthProxy(
        relative_depth_index=round(index, 4),
        method=_METHOD,
        is_physical_measurement=False,
    )
