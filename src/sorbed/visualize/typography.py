"""Font loading for the rendered figures.

Ships the Inter typeface (SIL Open Font License, see ``fonts/OFL.txt``) so every
figure has consistent, modern typography regardless of the host's installed
fonts. Falls back to DejaVu, then to Pillow's built-in bitmap font, if Inter
cannot be loaded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Inter.ttf"
_WEIGHTS = ("Thin", "ExtraLight", "Light", "Regular", "Medium", "SemiBold", "Bold", "ExtraBold")


@lru_cache(maxsize=96)
def font(size: int, weight: str = "Regular") -> ImageFont.ImageFont:
    """Return an Inter font at ``size`` px and the named ``weight``.

    ``weight`` is one of Inter's named instances (Regular, Medium, SemiBold,
    Bold, ...). Results are cached, so repeated calls are cheap.
    """
    if _FONT_PATH.is_file():
        try:
            face = ImageFont.truetype(str(_FONT_PATH), size)
            if weight in _WEIGHTS:
                face.set_variation_by_name(weight)
            return face
        except OSError:
            pass
    bold = weight in {"SemiBold", "Bold", "ExtraBold", "Medium"}
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()
