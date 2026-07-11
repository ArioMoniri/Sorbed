"""Derived clinical statistics for the reports.

Every value here is a deterministic derivation of fields already on a
:class:`WoundAnalysis` (or a :class:`HealingTrend`) — no new measurement and
nothing fabricated. Tissue fractions are normalised to the wound *bed*
(everything except intact skin) before viability ratios are computed, per
wound-bed-preparation practice. Composite indices and proxies are labelled as
such; thresholds are named constants, not inline literals.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import TissueClass
from sorbed.trend.models import HealingTrend

_EPS = 1e-6

# Wound-bed quality (WBQ) weights — a readable composite index (0-100), NOT a
# validated score. Viable tissue rewarded, devitalized tissue penalised.
_WBQ_WEIGHTS: dict[TissueClass, float] = {
    TissueClass.GRANULATION: 1.0,
    TissueClass.EPITHELIAL: 1.0,
    TissueClass.ADIPOSE: 0.3,
    TissueClass.SLOUGH: -0.5,
    TissueClass.ESCHAR: -1.0,
}
# Necrotic-burden red-flag triggers.
_ESCHAR_FLAG = 0.25
_NONVIABLE_FLAG = 0.50
# Edge-shape descriptors.
_IRREGULAR_CIRCULARITY = 0.60
_UNDERMINING_SOLIDITY = 0.85
_ELONGATION_RATIO = 3.0
# Healing-velocity bands (% wound area change per week; negative = shrinking).
_VEL_HEALING = -10.0
_VEL_STALL = -3.0


class ViabilitySplit(BaseModel):
    model_config = ConfigDict(frozen=True)
    viable_pct: float
    non_viable_pct: float
    granulation_pct: float
    epithelial_pct: float
    slough_pct: float
    eschar_pct: float


class TissueArea(BaseModel):
    model_config = ConfigDict(frozen=True)
    tissue: TissueClass
    fraction: float
    area_cm2: float | None


class Flag(BaseModel):
    model_config = ConfigDict(frozen=True)
    label: str
    active: bool
    detail: str = ""


class GradingStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    viability: ViabilitySplit
    tissue_areas: tuple[TissueArea, ...]
    granulation_slough_ratio: float | None
    wound_bed_quality: float          # 0-100 composite index
    volume_proxy: float | None        # area_cm2 * relative depth — PROXY
    standard_size: str                # "L × W, area"
    flags: tuple[Flag, ...]


def _bed_fractions(analysis: WoundAnalysis) -> dict[TissueClass, float]:
    fr = dict(analysis.metrics.tissue.fractions)
    bed = 1.0 - fr.get(TissueClass.INTACT_SKIN, 0.0) - fr.get(TissueClass.BACKGROUND, 0.0)
    if bed <= _EPS:
        return dict.fromkeys(fr, 0.0)
    return {k: (v / bed if k not in (TissueClass.INTACT_SKIN, TissueClass.BACKGROUND) else 0.0)
            for k, v in fr.items()}


def viability_split(analysis: WoundAnalysis) -> ViabilitySplit:
    b = _bed_fractions(analysis)
    gran = b.get(TissueClass.GRANULATION, 0.0)
    epi = b.get(TissueClass.EPITHELIAL, 0.0)
    slough = b.get(TissueClass.SLOUGH, 0.0)
    eschar = b.get(TissueClass.ESCHAR, 0.0)
    viable = gran + epi
    non_viable = slough + eschar
    return ViabilitySplit(
        viable_pct=round(viable * 100, 1), non_viable_pct=round(non_viable * 100, 1),
        granulation_pct=round(gran * 100, 1), epithelial_pct=round(epi * 100, 1),
        slough_pct=round(slough * 100, 1), eschar_pct=round(eschar * 100, 1),
    )


def tissue_areas_cm2(analysis: WoundAnalysis) -> tuple[TissueArea, ...]:
    fr = analysis.metrics.tissue.fractions
    areas_mm2 = analysis.metrics.tissue.areas_mm2 or {}
    total_cm2 = analysis.metrics.geometry.area_cm2
    out: list[TissueArea] = []
    for tc in (TissueClass.GRANULATION, TissueClass.EPITHELIAL, TissueClass.SLOUGH,
               TissueClass.ESCHAR, TissueClass.ADIPOSE):
        f = fr.get(tc, 0.0)
        if f <= 0.004:
            continue
        cm2: float | None = None
        if tc in areas_mm2:
            cm2 = round(areas_mm2[tc] / 100.0, 2)
        elif total_cm2 is not None:
            cm2 = round(total_cm2 * f, 2)
        out.append(TissueArea(tissue=tc, fraction=round(f, 3), area_cm2=cm2))
    return tuple(out)


def wound_bed_quality(analysis: WoundAnalysis) -> float:
    b = _bed_fractions(analysis)
    score = sum(w * b.get(tc, 0.0) for tc, w in _WBQ_WEIGHTS.items())
    return round(max(0.0, min(1.0, score)) * 100, 0)


def _ratio(analysis: WoundAnalysis) -> float | None:
    b = _bed_fractions(analysis)
    slough = b.get(TissueClass.SLOUGH, 0.0)
    gran = b.get(TissueClass.GRANULATION, 0.0)
    if slough < 0.01:
        return None if gran < 0.01 else 99.0  # sentinel: no slough
    return round(gran / slough, 1)


def _flags(analysis: WoundAnalysis, vs: ViabilitySplit) -> tuple[Flag, ...]:
    g = analysis.metrics.geometry
    b = _bed_fractions(analysis)
    eschar = b.get(TissueClass.ESCHAR, 0.0)
    non_viable = vs.non_viable_pct / 100.0
    necrotic = eschar >= _ESCHAR_FLAG or non_viable >= _NONVIABLE_FLAG
    elong = (g.major_axis_px / g.minor_axis_px) if g.minor_axis_px else 0.0
    return (
        Flag(label="Yüksek nekrotik yük · High necrotic burden", active=necrotic,
             detail=f"nekroz {vs.non_viable_pct:.0f}% · eskar {vs.eschar_pct:.0f}%"),
        Flag(label="Düzensiz kenar · Irregular edge", active=g.circularity < _IRREGULAR_CIRCULARITY,
             detail=f"dairesellik {g.circularity:.2f}"),
        Flag(label="Alttan oyulma şüphesi · Undermining suspected",
             active=g.solidity < _UNDERMINING_SOLIDITY,
             detail=f"doluluk {g.solidity:.2f} · 2B'den doğrulanamaz"),
        Flag(label="Uzamış yara · Elongated", active=elong >= _ELONGATION_RATIO,
             detail=f"en-boy {elong:.1f}"),
    )


def grading_stats(analysis: WoundAnalysis) -> GradingStats:
    g = analysis.metrics.geometry
    vs = viability_split(analysis)
    depth = analysis.metrics.depth_proxy
    vol = None
    if g.area_cm2 is not None and depth is not None and depth.relative_depth_index is not None:
        vol = round(g.area_cm2 * depth.relative_depth_index, 3)
    if g.length_mm and g.width_mm and g.area_cm2 is not None:
        size = f"{g.length_mm / 10:.1f} × {g.width_mm / 10:.1f} cm · {g.area_cm2:.2f} cm²"
    else:
        size = f"{g.length_px:.0f} × {g.width_px:.0f} px"
    return GradingStats(
        viability=vs, tissue_areas=tissue_areas_cm2(analysis),
        granulation_slough_ratio=_ratio(analysis), wound_bed_quality=wound_bed_quality(analysis),
        volume_proxy=vol, standard_size=size, flags=_flags(analysis, vs),
    )


# --------------------------------------------------------------------------- #
# Follow-up
# --------------------------------------------------------------------------- #
def healing_velocity_band(trend: HealingTrend) -> tuple[str, str]:
    """Return ``(band, colour)`` from the weekly % area change."""
    r = trend.healing_rate_pct_per_week
    if r is None:
        return ("Belirsiz · Indeterminate", "#8A93A0")
    if r <= _VEL_HEALING:
        return ("Hızlı iyileşme · Healing", "#16A34A")
    if r <= _VEL_STALL:
        return ("Yavaş iyileşme · Slow", "#65A30D")
    if r < abs(_VEL_STALL):
        return ("Duraklamış · Stalled", "#D97706")
    return ("Kötüleşiyor · Deteriorating", "#DC2626")
