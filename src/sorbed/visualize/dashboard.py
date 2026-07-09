"""A single comprehensive analysis figure.

Combines the input, wound mask, tissue overlay, a shading-based depth heatmap, and
the detection box into one panel row, then lays out the full statistics: size and
shape, a tissue-composition bar, PUSH / DESIGN-R sub-scores, the depth and
periwound cues, provenance, and the top evidence. Everything shown is computed by
the pipeline; the depth map is explicitly a relative shading cue, not a
measurement.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import TissueClass
from sorbed.visualize.detection import render_detection
from sorbed.visualize.overlay import render_mask, render_tissue_overlay
from sorbed.visualize.palette import stage_color, tissue_color

_PANEL = 216
_PAD = 14
_HEADER = 56
_BG = (22, 23, 26)
_FG = (232, 232, 234)
_MUTED = (150, 152, 158)


def render_depth_overlay(
    rgb_u8: np.ndarray, depth_field: np.ndarray, wound_mask: np.ndarray, *, alpha: float = 0.65
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
    """Build the full analysis dashboard image."""
    panels = [
        ("Input", Image.fromarray(display_rgb_u8, mode="RGB")),
        ("Wound mask", render_mask(wound_mask).convert("RGB")),
        ("Tissue match", render_tissue_overlay(display_rgb_u8, tissue_label_map, wound_mask)),
        ("Depth proxy", render_depth_overlay(display_rgb_u8, depth_field, wound_mask)),
        ("Detection", render_detection(analysis, display_rgb_u8)),
    ]
    cols = len(panels)
    width = cols * _PANEL + (cols + 1) * _PAD
    stats_h = 340
    height = _HEADER + _PANEL + 26 + stats_h
    canvas = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(canvas)

    _header(draw, analysis, width)
    for i, (label, img) in enumerate(panels):
        x = _PAD + i * (_PANEL + _PAD)
        _panel(canvas, draw, img, label, x, _HEADER + 8)

    _stats(canvas, draw, analysis, y0=_HEADER + _PANEL + 34, width=width)
    return canvas


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _header(draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, width: int) -> None:
    d = analysis.decision
    color = stage_color(d.stage)
    draw.rectangle([0, 0, width, _HEADER], fill=(30, 31, 35))
    draw.rectangle([0, 0, 8, _HEADER], fill=color)
    stage = d.stage.value.replace("_", " ").title()
    conf = "withheld" if d.abstained else f"{d.confidence * 100:.0f}% confidence"
    draw.text((18, 9), f"Sorbed  ·  {stage}  ·  {conf}", font=_font(20, True), fill=_FG)
    draw.text(
        (18, 34),
        "Decision support — not a diagnosis. Clinician review required. "
        "Depth shown is a relative shading cue, not a measurement.",
        font=_font(12),
        fill=(200, 170, 90),
    )


def _fit(img: Image.Image, box: int) -> Image.Image:
    img = img.convert("RGB")
    scale = min(box / img.width, box / img.height)
    resized = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    tile = Image.new("RGB", (box, box), (12, 12, 14))
    tile.paste(resized, ((box - resized.width) // 2, (box - resized.height) // 2))
    return tile


def _panel(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, img: Image.Image, label: str, x: int, y: int
) -> None:
    canvas.paste(_fit(img, _PANEL), (x, y))
    draw.text((x, y + _PANEL + 4), label, font=_font(13, True), fill=_MUTED)


def _stats(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, y0: int, width: int
) -> None:
    g = analysis.metrics.geometry
    col_w = (width - 4 * _PAD) // 3
    x1, x2, x3 = _PAD, _PAD * 2 + col_w, _PAD * 3 + 2 * col_w

    # Column 1 — measurements.
    lines1 = [("Measurements", None)]
    if g.area_cm2 is not None:
        lines1 += [
            ("Area", f"{g.area_cm2:.2f} cm²  ({g.area_px:.0f} px)"),
            ("Length × Width", f"{g.length_mm:.0f} × {g.width_mm:.0f} mm"),
            ("Perimeter", f"{(g.perimeter_px * (analysis.calibration.mm_per_px or 0)):.0f} mm"),
        ]
    else:
        lines1 += [
            ("Area", f"{g.area_px:.0f} px (uncalibrated)"),
            ("Length × Width", f"{g.length_px:.0f} × {g.width_px:.0f} px"),
            ("Perimeter", f"{g.perimeter_px:.0f} px"),
        ]
    lines1 += [
        ("Circularity", f"{g.circularity:.2f}"),
        ("Solidity", f"{g.solidity:.2f}"),
        ("Wound / image", f"{g.wound_fraction_of_image * 100:.1f}%"),
        ("Calibration", analysis.calibration.status.value.replace("_", " ")),
        ("Skin tone", analysis.skin_tone_band.value.replace("fitzpatrick_", "Fitz ").upper()),
    ]
    _kv_block(draw, lines1, x1, y0, col_w)

    # Column 2 — tissue composition bar + list.
    draw.text((x2, y0), "Tissue composition", font=_font(14, True), fill=_FG)
    _tissue_bar(draw, analysis, x2, y0 + 24, col_w)

    # Column 3 — scores, depth, periwound, evidence.
    hs = analysis.metrics.healing_scores
    dp = analysis.metrics.depth_proxy
    pw = analysis.metrics.periwound
    lines3 = [("Scores & cues", None)]
    if hs:
        if hs.push_partial_total is not None:
            lines3.append(("PUSH (partial)", f"{hs.push_partial_total}  "
                                              f"(size {hs.push_size_subscore}, "
                                              f"tissue {hs.push_tissue_subscore})"))
        if hs.design_r_size_subscore is not None:
            lines3.append(("DESIGN-R size", str(hs.design_r_size_subscore)))
        if hs.granulation_percent is not None:
            lines3.append(("Granulation", f"{hs.granulation_percent:.0f}%"))
    if dp:
        lines3.append(("Depth proxy", f"{dp.relative_depth_index:.2f} (relative)"))
    if pw and pw.erythema_index is not None:
        lines3.append(("Periwound erythema", f"{pw.erythema_index:+.1f} a*"))
        lines3.append(("Maceration", "suspected" if pw.maceration_suspected else "no"))
    prov = analysis.provenance
    lines3.append(("Segmenter", prov.segmentation_backend))
    _kv_block(draw, lines3, x3, y0, col_w)

    # Evidence + caveats span the bottom.
    ey = y0 + 214
    draw.text((x1, ey), "Why this grade", font=_font(14, True), fill=_FG)
    ey += 22
    for ev in analysis.decision.evidence[:3]:
        draw.text((x1, ey), f"• {ev.description}", font=_font(12), fill=_FG)
        ey += 17
    for c in analysis.decision.caveats[:2]:
        col = {"info": (120, 170, 230), "warning": (220, 180, 90), "critical": (230, 110, 110)}.get(
            c.severity.value, _MUTED
        )
        draw.text((x1, ey), f"! {c.message[:110]}", font=_font(12), fill=col)
        ey += 17


def _kv_block(
    draw: ImageDraw.ImageDraw, lines: list[tuple[str, str | None]], x: int, y: int, w: int
) -> None:
    for label, value in lines:
        if value is None:
            draw.text((x, y), label, font=_font(14, True), fill=_FG)
        else:
            draw.text((x, y), label, font=_font(12), fill=_MUTED)
            draw.text((x + 132, y), value, font=_font(12, True), fill=_FG)
        y += 21


def _tissue_bar(draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, x: int, y: int, w: int) -> None:
    fr = [
        (c, f)
        for c, f in sorted(analysis.metrics.tissue.fractions.items(), key=lambda kv: -kv[1])
        if f > 0.005 and c is not TissueClass.BACKGROUND
    ]
    bar_w = w - 8
    cx = x
    for c, f in fr:
        seg = max(1, int(bar_w * f))
        draw.rectangle([cx, y, cx + seg, y + 24], fill=tissue_color(c))
        cx += seg
    yy = y + 34
    for c, f in fr:
        draw.rectangle([x, yy, x + 14, yy + 12], fill=tissue_color(c))
        draw.text(
            (x + 20, yy),
            f"{c.value.replace('_', ' ').title()}  {f * 100:.0f}%",
            font=_font(12),
            fill=_FG,
        )
        yy += 18
