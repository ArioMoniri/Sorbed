"""A single, comprehensive analysis figure with a clean, minimal layout.

Combines the input, wound mask, tissue match, a shading-based depth heatmap, and
the detection box in one panel row, then lays out the statistics: size and shape,
a tissue-composition bar, PUSH / DESIGN-R sub-scores, depth and periwound cues,
provenance, and the top evidence. Everything shown is computed by the pipeline;
the depth map is explicitly a relative shading cue, not a measurement.
"""

from __future__ import annotations

import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import TissueClass
from sorbed.visualize.detection import render_detection
from sorbed.visualize.overlay import render_mask, render_tissue_overlay
from sorbed.visualize.palette import stage_color, tissue_color

# Minimal light theme.
_BG = (250, 250, 251)
_INK = (30, 32, 37)
_MUTED = (139, 144, 154)
_HAIR = (228, 230, 235)
_TILE = (243, 244, 246)

_MARGIN = 30
_PANEL = 196
_GAP = 20
_HEADER = 74
_PANELS = ("Input", "Mask", "Tissue", "Depth", "Detection")


def render_depth_overlay(
    rgb_u8: np.ndarray, depth_field: np.ndarray, wound_mask: np.ndarray, *, alpha: float = 0.7
) -> Image.Image:
    """Colorized relative-depth heatmap blended over the wound region."""
    d8 = (np.clip(depth_field, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    out = rgb_u8.astype(np.float32)
    m = wound_mask
    out[m] = (1.0 - alpha) * rgb_u8[m] + alpha * heat[m]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def render_dashboard(
    analysis: WoundAnalysis,
    display_rgb_u8: np.ndarray,
    wound_mask: np.ndarray,
    tissue_label_map: np.ndarray,
    depth_field: np.ndarray,
) -> Image.Image:
    """Build the analysis dashboard image."""
    images = [
        Image.fromarray(display_rgb_u8, mode="RGB"),
        render_mask(wound_mask).convert("RGB"),
        render_tissue_overlay(display_rgb_u8, tissue_label_map, wound_mask),
        render_depth_overlay(display_rgb_u8, depth_field, wound_mask),
        render_detection(analysis, display_rgb_u8),
    ]
    cols = len(images)
    width = 2 * _MARGIN + cols * _PANEL + (cols - 1) * _GAP
    height = _HEADER + _PANEL + 40 + 300
    canvas = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(canvas)

    _header(draw, analysis, width)
    y_panels = _HEADER + 20
    for i, (label, img) in enumerate(zip(_PANELS, images, strict=True)):
        x = _MARGIN + i * (_PANEL + _GAP)
        _panel(canvas, draw, img, label, x, y_panels)

    _stats(canvas, draw, analysis, _MARGIN, y_panels + _PANEL + 42, width)
    return canvas


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_w(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def _header(draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, width: int) -> None:
    d = analysis.decision
    draw.text((_MARGIN, 22), "Sorbed", font=_font(22, True), fill=_INK)
    draw.text((_MARGIN, 48), "pressure-injury analysis", font=_font(12), fill=_MUTED)

    stage = d.stage.value.replace("_", " ").title()
    conf = "review" if d.abstained else f"{d.confidence * 100:.0f}%"
    _pill(draw, f"{stage}  ·  {conf}", stage_color(d.stage), width - _MARGIN, 26)

    note = "Decision support, not a diagnosis · clinician review required"
    draw.text(
        (width - _MARGIN - _text_w(draw, note, _font(11)), 52), note, font=_font(11), fill=_MUTED
    )
    draw.line([(_MARGIN, _HEADER), (width - _MARGIN, _HEADER)], fill=_HAIR, width=1)


def _pill(
    draw: ImageDraw.ImageDraw, text: str, color: tuple[int, int, int], right: int, top: int
) -> None:
    font = _font(15, True)
    tw = _text_w(draw, text, font)
    pad_x, h = 14, 28
    x0 = right - tw - 2 * pad_x
    draw.rounded_rectangle([x0, top, right, top + h], radius=h // 2, fill=color)
    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    ink = (0, 0, 0) if lum > 150 else (255, 255, 255)
    draw.text((x0 + pad_x, top + 6), text, font=font, fill=ink)


def _fit(img: Image.Image, box: int) -> Image.Image:
    img = img.convert("RGB")
    scale = min(box / img.width, box / img.height)
    resized = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    tile = Image.new("RGB", (box, box), _TILE)
    tile.paste(resized, ((box - resized.width) // 2, (box - resized.height) // 2))
    return tile


def _panel(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, img: Image.Image, label: str, x: int, y: int
) -> None:
    canvas.paste(_fit(img, _PANEL), (x, y))
    draw.rectangle([x, y, x + _PANEL - 1, y + _PANEL - 1], outline=_HAIR, width=1)
    draw.text((x + 2, y + _PANEL + 8), label.upper(), font=_font(11, True), fill=_MUTED)


def _stats(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, x0: int, y0: int,
    width: int,
) -> None:
    g = analysis.metrics.geometry
    gutter = 34
    col_w = (width - 2 * _MARGIN - 2 * gutter) // 3
    cx = [x0, x0 + col_w + gutter, x0 + 2 * (col_w + gutter)]

    mm = analysis.calibration.mm_per_px
    size_rows = (
        [
            ("Area", f"{g.area_cm2:.2f} cm²"),
            ("Length × width", f"{g.length_mm:.0f} × {g.width_mm:.0f} mm"),
            ("Perimeter", f"{g.perimeter_px * mm:.0f} mm"),
        ]
        if g.area_cm2 is not None and mm
        else [
            ("Area", f"{g.area_px:.0f} px"),
            ("Length × width", f"{g.length_px:.0f} × {g.width_px:.0f} px"),
            ("Perimeter", f"{g.perimeter_px:.0f} px"),
        ]
    )
    size_rows += [
        ("Circularity", f"{g.circularity:.2f}"),
        ("Wound / image", f"{g.wound_fraction_of_image * 100:.1f}%"),
        ("Calibration", analysis.calibration.status.value.replace("_", " ")),
        ("Skin tone", _tone_label(analysis)),
    ]
    _column(draw, "MEASUREMENTS", size_rows, cx[0], y0, col_w)

    draw.text((cx[1], y0), "TISSUE", font=_font(11, True), fill=_MUTED)
    _tissue_strip(draw, analysis, cx[1], y0 + 22, col_w)

    hs, dp, pw = (
        analysis.metrics.healing_scores,
        analysis.metrics.depth_proxy,
        analysis.metrics.periwound,
    )
    score_rows: list[tuple[str, str]] = []
    if hs and hs.push_partial_total is not None:
        score_rows.append(("PUSH (partial)", f"{hs.push_partial_total}"))
    if hs and hs.design_r_size_subscore is not None:
        score_rows.append(("DESIGN-R size", str(hs.design_r_size_subscore)))
    if hs and hs.granulation_percent is not None:
        score_rows.append(("Granulation", f"{hs.granulation_percent:.0f}%"))
    if dp:
        score_rows.append(("Depth (relative)", f"{dp.relative_depth_index:.2f}"))
    if pw and pw.erythema_index is not None:
        score_rows.append(("Periwound erythema", f"{pw.erythema_index:+.1f} a*"))
    score_rows.append(("Segmenter", analysis.provenance.segmentation_backend))
    _column(draw, "SCORES & CUES", score_rows, cx[2], y0, col_w)

    _evidence(draw, analysis, cx[0], y0 + 196, width - 2 * _MARGIN)


def _tone_label(analysis: WoundAnalysis) -> str:
    return {
        "fitzpatrick_i_iii": "Fitzpatrick I–III",
        "fitzpatrick_iv_vi": "Fitzpatrick IV–VI",
    }.get(analysis.skin_tone_band.value, "unknown")


def _column(
    draw: ImageDraw.ImageDraw, title: str, rows: list[tuple[str, str]], x: int, y: int, w: int
) -> None:
    draw.text((x, y), title, font=_font(11, True), fill=_MUTED)
    yy = y + 24
    for label, value in rows:
        draw.text((x, yy), label, font=_font(12), fill=_MUTED)
        vw = _text_w(draw, value, _font(12, True))
        draw.text((x + w - vw, yy), value, font=_font(12, True), fill=_INK)
        yy += 22


def _tissue_strip(
    draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, x: int, y: int, w: int
) -> None:
    fr = [
        (c, f)
        for c, f in sorted(analysis.metrics.tissue.fractions.items(), key=lambda kv: -kv[1])
        if f > 0.005 and c is not TissueClass.BACKGROUND
    ]
    if not fr:
        return
    h = 16
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=_TILE)
    cxp = x
    for i, (c, f) in enumerate(fr):
        seg = max(2, int(w * f))
        last = i == len(fr) - 1
        end = x + w if last else min(x + w, cxp + seg)
        draw.rectangle([cxp, y, end, y + h], fill=tissue_color(c))
        cxp = end
    yy = y + h + 12
    for c, f in fr:
        draw.ellipse([x, yy + 2, x + 10, yy + 12], fill=tissue_color(c))
        draw.text(
            (x + 18, yy), f"{c.value.replace('_', ' ').title()}", font=_font(12), fill=_INK
        )
        pct = f"{f * 100:.0f}%"
        draw.text((x + w - _text_w(draw, pct, _font(12, True)), yy), pct, font=_font(12, True),
                  fill=_MUTED)
        yy += 19


def _evidence(draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, x: int, y: int, w: int) -> None:
    draw.line([(x, y - 12), (x + w, y - 12)], fill=_HAIR, width=1)
    draw.text((x, y), "WHY THIS GRADE", font=_font(11, True), fill=_MUTED)
    yy = y + 22
    char_w = max(60, int(w / 7.2))
    for ev in analysis.decision.evidence[:2]:
        for line in textwrap.wrap(f"•  {ev.description}", width=char_w)[:2]:
            draw.text((x, yy), line, font=_font(12), fill=_INK)
            yy += 17
    for c in analysis.decision.caveats[:2]:
        col = {"info": (70, 120, 190), "warning": (185, 130, 40), "critical": (200, 70, 70)}.get(
            c.severity.value, _MUTED
        )
        for line in textwrap.wrap(f"!  {c.message}", width=char_w)[:2]:
            draw.text((x, yy), line, font=_font(12), fill=col)
            yy += 17
