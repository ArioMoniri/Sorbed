"""HTML templates for the grading and follow-up reports.

Both builders return a complete, self-contained HTML document (fonts, images,
and vector charts inlined) ready for :func:`sorbed.report.pdf.html_to_pdf`. They
are pure assembly over already-rendered ``data:`` URIs, inline SVG charts, and a
:class:`~sorbed.guidelines.match.GuidelineContext`, so they carry no model or
I/O dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import PressureInjuryStage, TissueClass
from sorbed.guidelines.match import GuidelineContext, TopicBlock
from sorbed.report import svg
from sorbed.report.theme import (
    ALERT_COLORS,
    base_css,
    confidence_tier,
    stage_theme,
)
from sorbed.trend.alerts import HealingAssessment
from sorbed.trend.models import HealingTrend
from sorbed.visualize.palette import TISSUE_COLORS

_TISSUE_LABEL: dict[str, str] = {
    "granulation": "Granülasyon",
    "slough": "Fibrin/slough",
    "eschar": "Eskar (nekroz)",
    "epithelial": "Epitel",
    "adipose": "Yağ dokusu",
    "muscle": "Kas",
    "tendon_bone": "Tendon/kemik",
    "intact_skin": "Sağlam deri",
    "unknown": "Belirsiz",
}

_STAGE_SHORT: dict[str, str] = {
    "stage_1": "Evre 1",
    "stage_2": "Evre 2",
    "stage_3": "Evre 3",
    "stage_4": "Evre 4",
    "unstageable": "Sınıflandırılamayan",
    "deep_tissue_injury": "Derin doku",
    "mucosal_not_stageable": "Mukozal",
    "not_pressure_injury": "Basınç yarası değil",
    "indeterminate": "Belirsiz",
}


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _tlabel(tc: TissueClass) -> str:
    return _TISSUE_LABEL.get(tc.value, tc.value.replace("_", " ").title())


def _fmt(v: float | None, nd: int = 1, dash: str = "—") -> str:
    return dash if v is None else f"{v:.{nd}f}"


def _doc(css: str, body: str, title: str) -> str:
    return (
        "<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title><style>{css}</style></head>"
        f"<body>{body}</body></html>"
    )


def _brandrow(ctx: GuidelineContext, kicker: str) -> str:
    ref = ""
    if ctx.available and ctx.meta:
        m = ctx.meta
        parts = [escape(m.doc_name)]
        if m.doc_no:
            parts.append(f"{escape(m.doc_no)} · Rev {escape(m.revision_no or '')}")
        if m.revision_date:
            parts.append(escape(m.revision_date))
        ref = "<br>".join(parts)
    return (
        "<div class='brandrow'>"
        "<div class='brand'><span class='logo'></span>"
        "<div><div class='name'>Sorbed</div>"
        f"<div class='kick'>{escape(kicker)}</div></div></div>"
        f"<div class='docref'>{ref}</div></div>"
    )


def _confbox(conf: float) -> str:
    tier, color = confidence_tier(conf)
    pct = round(conf * 100)
    return (
        "<div class='confbox'>"
        f"<div class='confhead'><span class='muted'>{escape(tier)}</span>"
        f"<span class='confpct' style='color:{color}'>{pct}%</span></div>"
        f"<div class='confbar'><div class='conffill' style='width:{pct}%;background:{color}'></div></div>"
        "</div>"
    )


def _pill(label: str, color: str | None = None) -> str:
    dot = f"<span class='dot' style='background:{color}'></span>" if color else ""
    return f"<span class='pill'>{dot}{escape(label)}</span>"


def _metric(k: str, v: str, u: str = "") -> str:
    unit = f" <span class='u'>{escape(u)}</span>" if u else ""
    return f"<div class='metric'><div class='k'>{escape(k)}</div><div class='v mono'>{v}{unit}</div></div>"


def _tissue_bar(fractions: Mapping[TissueClass, float]) -> str:
    items = [(tc, f) for tc, f in fractions.items()
             if f > 0.004 and tc not in (TissueClass.BACKGROUND, TissueClass.INTACT_SKIN)]
    items.sort(key=lambda kv: kv[1], reverse=True)
    total = sum(f for _, f in items) or 1.0
    segs, legend = [], []
    for tc, f in items:
        color = _hex(TISSUE_COLORS.get(tc, (130, 130, 130)))
        segs.append(f"<span style='width:{f / total * 100:.2f}%;background:{color}'></span>")
        legend.append(
            f"<div class='item'><span class='swatch' style='background:{color}'></span>"
            f"{escape(_tlabel(tc))} <b class='mono'>{f * 100:.0f}%</b></div>"
        )
    if not segs:
        return "<p class='muted small'>Doku bileşimi belirlenemedi.</p>"
    return (f"<div class='tissuebar'>{''.join(segs)}</div>"
            f"<div class='legend'>{''.join(legend)}</div>")


def _cite_line(block: TopicBlock | None) -> str:
    if not block or not block.citations:
        return ""
    refs = ", ".join(f"§{c.section} (s.{c.page})" for c in block.citations)
    return f"<div class='ref'>Kaynak: {escape(refs)}</div>"


def _directive_figure_block(block: TopicBlock | None, fig_uri: str | None) -> str:
    if not block:
        return ""
    quote = f"<div class='q'>{escape(block.text)}</div>{_cite_line(block)}"
    if fig_uri:
        cap = escape(block.caption or "")
        return (
            "<div class='directivefig'>"
            f"<figure class='fig'><img src='{fig_uri}'><figcaption>{cap}</figcaption></figure>"
            f"<div style='padding-top:2px'>{quote}</div>"
            "</div>"
        )
    return f"<div class='cite'>{quote}</div>"


def _card(title: str, inner: str, *, cls: str = "", sub: str = "") -> str:
    extra = f" {cls}" if cls else ""
    subhtml = f"<span class='cardsub'>{escape(sub)}</span>" if sub else ""
    return (
        f"<section class='card{extra}'><div class='cardtitle'><span class='bar'></span>"
        f"<h2>{escape(title)}</h2>{subhtml}</div>{inner}</section>"
    )


def _stage_likelihood(analysis: WoundAnalysis) -> str:
    probs = analysis.decision.ml_stage_probabilities or {}
    if not probs:
        return ""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:4]
    rows = []
    for stage, p in ranked:
        st = stage if isinstance(stage, PressureInjuryStage) else PressureInjuryStage(stage)
        color = stage_theme(st)[0]
        rows.append((_STAGE_SHORT.get(st.value, st.value), float(p), color))
    return svg.hbars_svg(rows)


# --------------------------------------------------------------------------- #
# Grading report
# --------------------------------------------------------------------------- #
def build_grading_report_html(
    *,
    analysis: WoundAnalysis,
    guideline_ctx: GuidelineContext,
    images: Mapping[str, str],
    patient_ref: str | None = None,
    generated: str = "",
) -> str:
    d = analysis.decision
    g = analysis.metrics.geometry
    hs = analysis.metrics.healing_scores
    accent, soft, stage_label = stage_theme(d.stage)
    css = base_css(accent, soft)

    sub = escape((d.narrative or "").split(". ")[0][:150]) if d.narrative else ""
    tone = analysis.skin_tone_band.value.replace("_", " ").title() if analysis.skin_tone_band else "—"
    calib = "kalibre · cm²" if g.area_cm2 is not None else "kalibresiz · px"
    pills = "".join([
        _pill(f"Cilt tonu · {tone}"),
        _pill(calib),
        _pill(f"ID {str(analysis.analysis_id)[:8]}"),
    ])
    if patient_ref:
        pills = _pill(patient_ref) + pills
    hero = (
        "<div class='hero'>"
        f"{_brandrow(guideline_ctx, 'pressure-injury analysis')}"
        "<div class='gradewrap'>"
        "<div class='gradeblock'>"
        "<div class='eyebrow'>Otomatik evreleme · Auto grade</div>"
        f"<div class='grade'>{escape(stage_label)}</div>"
        f"<div class='sub'>{sub}</div>"
        f"<div class='pillrow'>{pills}</div>"
        "</div>"
        f"{_confbox(d.confidence)}"
        "</div></div>"
    )
    disclaimer = (
        "<div class='disc'><b>Karar desteği — tanı değildir.</b> "
        f"{escape(analysis.disclaimer or 'Klinisyen doğrulaması gerektirir.')}</div>"
    )

    # Left column: imagery. Photo large, model panels beneath.
    photo = (f"<figure class='big'><img src='{images['photo']}'>"
             "<figcaption>Yüklenen görüntü · Uploaded photo</figcaption></figure>"
             if "photo" in images else "")
    small = []
    labels = {"overlay": "Doku · Tissue", "detection": "Tespit · Detection",
              "depth": "Derinlik · Depth"}
    for key in ("overlay", "detection", "depth"):
        if key in images:
            small.append(f"<figure><img src='{images[key]}'>"
                         f"<figcaption>{escape(labels[key])}</figcaption></figure>")
    left = (f"<div class='imgstack'>{photo}<div class='trio'>{''.join(small)}</div></div>")

    # Right column: measurements + stage likelihood.
    area_v = f"{g.area_cm2:.2f}" if g.area_cm2 is not None else f"{g.area_px:.0f}"
    area_u = "cm²" if g.area_cm2 is not None else "px"
    lw = (f"{g.length_mm:.0f}×{g.width_mm:.0f}" if g.length_mm and g.width_mm
          else f"{g.length_px:.0f}×{g.width_px:.0f}")
    lw_u = "mm" if g.length_mm else "px"
    gran = _fmt(hs.granulation_percent if hs else None, 0)
    push = str(hs.push_partial_total) if hs and hs.push_partial_total is not None else "—"
    metrics = "".join([
        _metric("Yüzey alanı", area_v, area_u),
        _metric("U×G", lw, lw_u),
        _metric("Çevre", f"{g.perimeter_px:.0f}", "px"),
        _metric("Granülasyon", gran, "%"),
        _metric("PUSH", push, "/17"),
        _metric("Baskın doku", escape(_tlabel(analysis.metrics.tissue.dominant))),
    ])
    like = _stage_likelihood(analysis)
    if like:
        extra = f"<div class='mininote'>Evre olasılıkları · Stage likelihood</div>{like}"
    else:
        # No ML stage posterior available — fill the column with the tissue mix,
        # which is always informative and grounds the RYB card below.
        extra = ("<div class='mininote'>Doku bileşimi · Tissue composition</div>"
                 + _tissue_bar(analysis.metrics.tissue.fractions))
    right = f"<div class='grid2 tight'>{metrics}</div>{extra}"

    top_card = _card(
        "Görüntü, ölçüm ve model çıktısı · Image, metrics & model output",
        f"<div class='splitcol'><div>{left}</div><div>{right}</div></div>",
        cls="avoidbreak",
    )

    # RYB tissue-colour model (directive §4.5.9) + per-tissue readings.
    tissue_card = ""
    if guideline_ctx.available and guideline_ctx.tissue_model:
        readings = "".join(
            f"<div class='item'><span class='swatch' style='background:"
            f"{_hex(TISSUE_COLORS.get(tr.tissue, (130, 130, 130)))}'></span>"
            f"{escape(_tlabel(tr.tissue))} · {tr.ryb} — {escape(tr.directive_note)}</div>"
            for tr in guideline_ctx.tissue_readings
        )
        inner = (
            f"<div class='cite'><div class='q'>{escape(guideline_ctx.tissue_model.text)}</div>"
            f"{_cite_line(guideline_ctx.tissue_model)}</div>"
            f"<div class='legend' style='margin-top:8px'>{readings}</div>"
        )
        tissue_card = _card("Doku rengi modeli · Tissue-colour model (RYB · §4.5.9)",
                            inner, cls="avoidbreak")

    stage_card = ""
    if guideline_ctx.available and guideline_ctx.stage:
        stage_card = _card(
            "Evrelendirme kriteri · Staging criterion",
            _directive_figure_block(guideline_ctx.stage, images.get("directive_stage")),
            cls="avoidbreak",
        )

    care_card = ""
    if guideline_ctx.available:
        bits = []
        if guideline_ctx.reassessment_open:
            ref = ""
            if guideline_ctx.reassessment_citation:
                c = guideline_ctx.reassessment_citation
                ref = f" <span class='muted'>(§{c.section}, s.{c.page})</span>"
            bits.append(f"<p><b>Yeniden değerlendirme (açık yara):</b> "
                        f"{escape(guideline_ctx.reassessment_open)}{ref}</p>")
        if guideline_ctx.care and guideline_ctx.care.text:
            bits.append(f"<div class='cite' style='margin-top:6px'><div class='q'>"
                        f"{escape(guideline_ctx.care.text[:820])}</div>{_cite_line(guideline_ctx.care)}</div>")
        if bits:
            care_card = _card("Bakım ve yeniden değerlendirme · Care & reassessment",
                              "".join(bits), cls="avoidbreak")

    body = (hero + disclaimer + top_card + tissue_card + stage_card + care_card
            + _footer(guideline_ctx, generated))
    return _doc(css, body, "Sorbed · Basınç Yaralanması Değerlendirme")


# --------------------------------------------------------------------------- #
# Follow-up report
# --------------------------------------------------------------------------- #
def build_followup_report_html(
    *,
    trend: HealingTrend,
    assessment: HealingAssessment,
    guideline_ctx: GuidelineContext,
    visit_thumbs: Mapping[str, str],
    patient_ref: str | None = None,
    generated: str = "",
    attribution: str = "",
) -> str:
    latest_stage = trend.points[-1].stage
    accent, soft, _ = stage_theme(latest_stage)
    css = base_css(accent, soft)
    bannercolor = ALERT_COLORS.get(assessment.level, ALERT_COLORS["info"])

    hero = (
        "<div class='hero'>"
        f"{_brandrow(guideline_ctx, 'pressure-injury analysis')}"
        "<div class='gradeblock'>"
        "<div class='eyebrow'>İyileşme takibi · Healing follow-up</div>"
        f"<div class='grade' style='font-size:23px'>"
        f"{escape(patient_ref or trend.patient_ref or 'Wound trajectory')}</div>"
        f"<div class='sub'>{len(trend.points)} ziyaret · "
        f"{trend.points[-1].day - trend.points[0].day:.0f} gün · birim: {trend.unit}</div>"
        "</div></div>"
    )

    # Verdict banner + healing gauge side by side.
    gauge = svg.gauge_svg(trend.percent_area_reduction, accent="#16A34A"
                          if trend.percent_area_reduction >= 0 else "#DC2626",
                          caption="alan azalması · PAR")
    banner = (
        "<div class='verdictrow avoidbreak'>"
        f"<div class='banner' style='background:{bannercolor}'>"
        f"<div class='lvl'>{escape(assessment.level.upper())}</div>"
        f"<div class='hd'>{escape(assessment.headline)}</div>"
        f"<div class='sm'>{escape(assessment.summary)}</div></div>"
        f"<div class='gaugebox'>{gauge}</div>"
        "</div>"
    )

    tiles = "".join([
        _metric("Baz→son alan", f"{trend.baseline_area:.3g}→{trend.latest_area:.3g}", trend.unit),
        _metric("Haftalık hız", _fmt(trend.healing_rate_pct_per_week, 1), "%/hf"),
        _metric("4-haftalık PAR", _fmt(trend.par_at_4_weeks, 0), "%"),
        _metric("PUSH eğilimi", escape((trend.push_trend or "—").title())),
        _metric("Kapanma (proj.)", _fmt(trend.projected_days_to_closure, 0), "gün"),
        _metric("İyileşme olasılığı",
                "Evet" if trend.likely_to_heal else "Hayır" if trend.likely_to_heal is False else "—"),
    ])
    tiles_card = _card("Özet · Summary", f"<div class='grid3 tight'>{tiles}</div>")

    # SVG charts.
    days = [p.day for p in trend.points]
    areas = [(p.area_cm2 if trend.unit == "cm2" and p.area_cm2 is not None else p.area_px)
             for p in trend.points]
    proj = None
    if trend.projected_days_to_closure is not None:
        proj = (days[-1] + trend.projected_days_to_closure, 0.0)
    area_svg = svg.line_chart_svg(days, areas, accent=accent, unit=trend.unit, projection=proj)

    push_pts = [(p.day, float(p.push_total)) for p in trend.points if p.push_total is not None]
    charts_html = [
        f"<figure class='chart'>{area_svg}<figcaption>Yüzey alanı · Wound area "
        f"({trend.unit}) — kesikli çizgi: projeksiyon</figcaption></figure>",
        f"<figure class='chart'>{svg.tissue_stack_svg(trend)}"
        "<figcaption>Doku bileşimi · Tissue mix (%)</figcaption></figure>",
    ]
    if len(push_pts) >= 2:
        push_svg = svg.line_chart_svg([x for x, _ in push_pts], [y for _, y in push_pts],
                                      accent="#0E9C9B", unit="")
        charts_html.append(
            f"<figure class='chart'>{push_svg}<figcaption>PUSH toplamı · PUSH total "
            "(↓ = iyileşme)</figcaption></figure>"
        )
    charts_card = _card("Eğilim grafikleri · Trend charts",
                        f"<div class='chartgrid'>{''.join(charts_html)}</div>", cls="avoidbreak")

    # Alerts
    alert_rows = []
    for a in assessment.alerts:
        color = ALERT_COLORS.get(a.level, ALERT_COLORS["info"])
        refs = ("  ·  " + ", ".join(f"§{r}" for r in a.directive_refs)) if a.directive_refs else ""
        alert_rows.append(
            f"<div class='alert'><div class='stripe' style='background:{color}'></div>"
            f"<div><div class='at'>{escape(a.title)}</div>"
            f"<div class='ad'>{escape(a.detail)}</div>"
            f"<div class='aref'>{escape(a.level.upper())}{escape(refs)}</div></div></div>"
        )
    alerts_card = _card("Uyarılar · Clinical alerts",
                        "".join(alert_rows) or "<p class='muted small'>Uyarı yok.</p>",
                        cls="avoidbreak")

    # Visit timeline table
    rows = []
    for p in trend.points:
        acc = stage_theme(p.stage)[0]
        thumb = visit_thumbs.get(p.label, "")
        img = (f"<img src='{thumb}' class='vthumb'>" if thumb else "")
        area = f"{p.area_cm2:.2f}" if p.area_cm2 is not None else f"{p.area_px:.0f}"
        gran = p.fraction(TissueClass.GRANULATION) * 100
        nec = (p.fraction(TissueClass.SLOUGH) + p.fraction(TissueClass.ESCHAR)) * 100
        push = str(p.push_total) if p.push_total is not None else "—"
        rows.append(
            f"<tr><td>{img}</td><td><b>{escape(p.label)}</b><br>"
            f"<span class='muted small'>gün {p.day:.0f}</span></td>"
            f"<td><span class='chip' style='background:{acc}'>"
            f"{escape(_STAGE_SHORT.get(p.stage.value, p.stage.value))}</span></td>"
            f"<td class='mono'>{area} {trend.unit}</td>"
            f"<td class='mono'>{gran:.0f}%</td><td class='mono'>{nec:.0f}%</td>"
            f"<td class='mono'>{push}</td><td class='mono'>{round(p.confidence * 100)}%</td></tr>"
        )
    table = (
        "<table class='visits'><thead><tr><th></th><th>Ziyaret</th><th>Evre</th>"
        "<th>Alan</th><th>Gran.</th><th>Nekroz</th><th>PUSH</th><th>Güven</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    visits_card = _card("Ziyaret zaman çizelgesi · Visit timeline", table, cls="avoidbreak")

    directive_card = ""
    if guideline_ctx.available:
        blocks = []
        for label, block in (("PUSH · §4.5.8", guideline_ctx.push),
                             ("Doku rengi · §4.5.9", guideline_ctx.tissue_model),
                             ("Boyut · §4.5.11", guideline_ctx.size)):
            if block and block.text:
                blocks.append(
                    f"<div class='cite col'><h3>{escape(label)}</h3>"
                    f"<div class='q' style='margin-top:4px'>{escape(block.text[:460])}</div>"
                    f"{_cite_line(block)}</div>"
                )
        if blocks:
            directive_card = _card("Talimat dayanağı · Directive basis",
                                   f"<div class='citegrid'>{''.join(blocks)}</div>", cls="avoidbreak")

    body = (hero + banner + tiles_card + charts_card + alerts_card
            + visits_card + directive_card + _footer(guideline_ctx, generated, attribution))
    return _doc(css, body, "Sorbed · İyileşme Takibi")


def _footer(ctx: GuidelineContext, generated: str, attribution: str = "") -> str:
    bits = [
        "Sorbed karar destek yazılımıdır; tıbbi cihaz veya tanı aracı değildir. "
        "Tüm çıktılar klinisyen doğrulaması gerektirir.",
    ]
    if ctx.available and ctx.meta:
        bits.append(
            f"Klinik dayanak: {escape(ctx.meta.doc_name)} "
            f"({escape(ctx.meta.doc_no or '')} Rev {escape(ctx.meta.revision_no or '')}, "
            f"{escape(ctx.meta.revision_date or '')}). Alıntılar ve şekiller kaynak dokümandan alınmıştır."
        )
    if attribution:
        bits.append(escape(attribution))
    if generated:
        bits.append(f"Oluşturulma: {escape(generated)}")
    return "<div class='foot'>" + "<br>".join(bits) + "</div>"
