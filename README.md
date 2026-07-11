# Sorbed

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Sorbed takes a photograph of a pressure injury (a bedsore) and returns an explainable, provisional grade — a stage, the tissue it saw in the wound bed, size measurements when the image carries a scale, and a plain-language account of *why* it decided what it did. The whole thing runs offline on a laptop CPU with no model downloads, because the default pipeline is classical computer vision, not a neural network waiting on weights. Learned backends exist and can be switched on, but they are never required.

> **This is decision-support software, not a medical device.** It does not diagnose. Every number it prints is an estimate computed from a flat 2D image by an algorithm that never touched the patient. A pressure-injury stage is a clinical determination made by a qualified professional — Sorbed's output is one input to that judgment and nothing more. Read [`DISCLAIMER.md`](DISCLAIMER.md) before you point it at a real wound.

## What it does 🩹

Give it an image and an optional scale, and it writes a folder of artifacts:

```bash
sorbed analyze wound.jpg --mm-per-px 0.15 --out reports
```

```
Provisional grade: Stage 3  (62%)
Area: 8.40 cm²   Size: 41 × 29 mm
Tissue: granulation 58%, slough 27%, eschar 9%
! Depth-dependent grade — confirm Stage 3 vs 4 at the bedside.
```

Alongside the console summary it drops a binary mask, a tissue overlay, an annotated clinician guide, a clean schematic, the canonical JSON, and a self-contained HTML report into `reports/`. Plain phone photos are fine; the wound can be any size or body location. If the photo has no scale reference, the physical measurements come back as `null` and the pixel-space metrics are reported instead — Sorbed degrades honestly rather than inventing a scale.

## Pipeline overview

Each stage is a pure, typed transform that only *adds* to an immutable context, so any point can be snapshotted and inspected.

```
 image bytes ─▶ Ingest ─▶ Decode & Normalize ─▶ Preprocess (calibrate) ─▶
   Wound Segmentation ─▶ Tissue Segmentation ─▶ Morphometrics ─▶
     Staging Engine (rule ⊕ ML ⊕ arbiter) ─▶ Explainability ─▶ Outputs
```

The staging engine leans on a deterministic rule layer that encodes NPIAP/EPUAP staging doctrine; a learned head, when enabled, only proposes candidates that an arbiter reconciles against the rules. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Install

Requires Python 3.11+.

```bash
pip install -e .                 # core: the classical pipeline, CLI, no downloads
pip install -e .[formats]        # HEIC/HEIF, DICOM, camera RAW, BigTIFF decoders
pip install -e .[ml]             # optional ONNX learned backends
pip install -e .[api]            # the FastAPI service
pip install -e .[pdf]            # PDF grading & follow-up reports (headless Chromium)
pip install -e .[dev]            # ruff, mypy, pytest, hypothesis
pip install -e .[all]            # formats + ml + pdf + api + hf
```

After installing the `[pdf]` extra, fetch the Chromium build once:

```bash
playwright install chromium
```

## Quickstart

**CLI:**

```bash
sorbed analyze wound.heic --marker-mm 20 --out reports   # calibrate from an ArUco marker
sorbed inspect wound.dcm                                  # decode & show metadata + scale, no grading
sorbed formats                                            # which decoders are installed
sorbed schema --out analysis.schema.json                 # the JSON contract
```

**Python:**

```python
from sorbed.pipeline import analyze_image, AnalyzeOptions

bundle = analyze_image("wound.jpg", options=AnalyzeOptions(mm_per_px=0.15))
d = bundle.analysis.decision
print(d.stage, round(d.confidence, 2), d.abstained)
print(bundle.analysis.metrics.tissue.fractions)
```

## Outputs

Every artifact is a *rendering* of one canonical JSON record — nothing downstream computes anything the JSON does not already contain.

| Artifact | File | What it is |
|---|---|---|
| Mask | `*_mask.png` | Binary wound-boundary mask. |
| Overlay | `*_overlay.png` | Per-tissue coloring drawn over the original photo. |
| Guide | `*_guide.png` | Annotated photo: contour, length × width, scale bar, legend, grade banner. |
| Schematic | `*_schematic.png` | Clean synthetic drawing of the sore, free of photographic noise. |
| Schematic guide | `*_schematic_guide.png` | The schematic with measurements and labels. |
| JSON | `*.json` | Canonical `WoundAnalysis` — the single source of truth. |
| HTML | `*.html` | Self-contained clinician report. |
| PDF (grade) | `*_grading.pdf` | Directive-grounded grading report (needs `[pdf]`). |
| PDF (follow-up) | `*_followup.pdf` | Longitudinal healing report with trend charts + alerts. |

Pick a subset with `--format mask,json` or take them all with `--format all` (the default).

## Clinical directive grounding 📋

Sorbed can ground every report in a hospital's *own* pressure-injury directive rather than a generic rulebook. A **directive pack** is that guideline — for example an institutional NPIAP/EPUAP-aligned protocol — distilled from its PDF into citable, numbered sections plus the guideline's own figures:

```bash
python scripts/build_directive_pack.py --pdf HD_T86_REV10.pdf --slug my_hospital
export SORBED_DIRECTIVE_PACK=var/directive_packs/my_hospital
```

The analysis is then mapped onto the directive's criteria: the staging definition **and** the guideline's own illustration for the detected stage, its tissue-colour (red-yellow-black) model, sizing and PUSH guidance, and the reassessment cadence — each with a section/page citation. Institutional directive content is loaded at runtime and is never vendored into this repository. This is what makes the model "tuned to a local clinical protocol": swap the pack, and the same engine speaks your unit's guideline.

## PDF reports & healing follow-up 📈

Two print-ready, self-contained, **bilingual (Turkish · English)** PDFs (modern layout, Manrope, grade- and confidence-based colouring), rendered by headless Chromium:

- **Grading report** — the grade and calibrated confidence; the uploaded photo beside every generated output (binary mask, tissue overlay, detection box, relative-depth cue, synthetic schematic, and a full analysis dashboard); a **clinical-statistics** card (bed-normalised tissue viability, tissue areas in cm², a wound-bed-quality index, granulation/slough ratio, red-flag chips, standard L×W×area); a **guideline-comparison** panel putting the directive's own figure + verbatim text + an NPIAP English gloss beside Sorbed's schematic and findings; the directive's full **staging-ladder reference** (every stage's figure + text chunk); and the tissue-colour model and care/reassessment cadence — all with section/page citations.
- **Follow-up report** — a better/worse verdict banner with a healing gauge and a healing-velocity band, vector trend charts (wound area with a projected-closure line, tissue mix over visits, PUSH total), directive-cited **clinical alerts** ("wound surface enlarging", "PUSH total falling", "devitalized tissue increasing", "stage progression"), and a per-visit timeline.

Derived clinical statistics live in `sorbed.report.stats` (named constants, bed-normalised, every proxy explicitly flagged); nothing is hardcoded.

Longitudinal analytics track granulation and full tissue composition, surface area (cm² when calibrated), PUSH, percent area reduction, the validated 4-week PAR predictor, and the Gilman perimeter-normalized healing rate — see [`docs/TREND.md`](docs/TREND.md).

## Clinical workflow: autograde on upload + QA 🏥

The target workflow mirrors how wound imaging actually flows through an EHR. Today a nurse photographs the wound, uploads it, and *types in a stage* — which a central quality office frequently has to re-check and correct. Sorbed is designed to slot into that loop as decision support:

1. **On upload**, autograde the image and pre-fill a provisional stage, size, tissue composition, and a directive-cited rationale — so the human starts from a structured draft instead of a blank field.
2. **Human-in-the-loop:** the nurse confirms or edits; low-confidence or out-of-distribution images (bad lighting, obscured bed) **abstain** and route to review rather than forcing a stage.
3. **Follow-up** runs automatically across a patient's visits, surfacing the healing trajectory and better/worse alerts to the QA office instead of manual chart review.

Everything remains decision support — a clinician owns the final determination.

## How the grade is decided 🧭

The stage comes from an explicit, auditable rule engine, not a black box. Staging is driven by *which tissues are present*: visible fat forces at least Stage 3, any structural tissue forces Stage 4, a bed obscured by slough or eschar forces Unstageable, and maroon-to-purple discoloration on intact skin points to Deep Tissue Injury. Each grade carries a ranked evidence trace whose items reference real metric fields (for example `metrics.tissue.fractions.eschar`), a confidence, caveats, and a deterministic narrative.

Two design choices matter most. Depth-dependent grades are damped and flagged for clinician review, because you cannot assert a depth a photo cannot show. And when the evidence does not support any confident grade, the engine returns **Indeterminate** and abstains rather than guessing. Abstention is a feature. The full clinical specification is in [`docs/CLINICAL.md`](docs/CLINICAL.md).

## Skin-tone equity ⚖️

Stage 1 and Deep Tissue Injury are defined partly by color changes that are genuinely harder to see in darkly pigmented skin, and naive color analysis inherits — and can amplify — that bias. Sorbed estimates an ITA-based skin-tone band, lowers its confidence and raises an explicit warning on darker skin, and prompts assessment of temperature, firmness, and edema, which a camera cannot capture. A low or negative result on dark skin does **not** rule out injury. See [`docs/EQUITY.md`](docs/EQUITY.md).

## Formats

| Family | Formats | Needs |
|---|---|---|
| Common | PNG, JPEG, WEBP, BMP, GIF, TIFF | core |
| Phone | HEIC / HEIF | `[formats]` (pillow-heif) |
| Clinical | DICOM (pixel-spacing calibration + PHI stripping) | `[formats]` (pydicom) |
| Camera | RAW | `[formats]` (rawpy) |

Run `sorbed formats` to see what is installed on your machine.

## Models & the data reality

Wound imaging is data-poor, and pressure-injury *staging* is its poorest corner: no large, public, permissively-licensed staging dataset could be verified to exist, and the strong published numbers come from private single-center sets under noisy ground truth (human raters agree only 23–58% of the time). That is exactly why Sorbed ships staging as decision support with explicit uncertainty, keeps a weight-free classical backend that always works offline, and — when learned ONNX backends are enabled — downloads weights only on request, verifies them by sha256, and records each in a provenance registry. The honest, cited accounting lives in [`docs/MODELS.md`](docs/MODELS.md).

**Segmentation.** For *wound-area segmentation* — unlike staging — real public benchmarks exist. Sorbed's learned segmenter is trained on the **AZH Chronic Wound / MICCAI-2021 FUSeg** foot-ulcer datasets (real clinical photographs), and the training path supports the architecture that leads that benchmark: a U-Net with an EfficientNet encoder and **scSE** (spatial-and-channel Squeeze-and-Excitation) decoder attention — the mechanism the FUSegNet line uses to reach SOTA.

| Approach | Data-based DSC | Notes |
|---|---|---|
| FUSegNet (EfficientNet-b7 + P-scSE) | **92.70%** | Dhar et al., *Biomed. Signal Process. Control* 2024 |
| x-FUSegNet (5-fold ensemble) | 89.23% | tops the FUSeg-2021 challenge leaderboard |
| LinkNet-EffB1 + UNet-EffB2 (ensemble) | 92.07% | Mahbod et al. |
| DeepLabV3+ / PSPNet / MANet | 91–92% | strong baselines |
| U-Net + scSE | 91.85% | scSE over plain U-Net (90.88%) |

Train with the attention on real data (see [`docs/TRAINING.md`](docs/TRAINING.md)):

```bash
python scripts/train_segmenter.py --images imgs/ --masks masks/ \
    --encoder efficientnet-b4 --decoder-attention scse
```

For a broader, cited survey of 2024–2026 model and system designs — promptable foundation models (SAM / MedSAM / MedSAM-2), on-device staging (YOLOv8), skin-tone equity, and the EHR/regulatory picture (FDA SaMD, EU MDR + AI Act, Singapore HSA) — see [`docs/MODELS.md`](docs/MODELS.md).

## API

An optional FastAPI service exposes the same pipeline over HTTP (install `[api]`):

```bash
uvicorn sorbed.api.app:app --reload
# POST /v1/analyze   GET /v1/health   GET /v1/formats
```

## Project layout

```
Sorbed/
├── src/sorbed/
│   ├── domain/          pydantic v2 data contracts (the types every stage speaks)
│   ├── io/              format sniffing, decoders, normalization
│   ├── preprocess/      calibration, fiducial detection, color normalization
│   ├── segmentation/    wound boundary — classical + pluggable backends
│   ├── tissue/          tissue-type classification
│   ├── morphometrics/   geometry, tissue %, healing sub-scores
│   ├── staging/         features · rules · ml_head · arbiter · engine
│   ├── explain/         the evidence trace
│   ├── visualize/       masks, overlays, guides, schematics
│   ├── report/          HTML / PDF / JSON assembly
│   ├── pipeline/        the orchestrator
│   ├── api/             FastAPI service
│   └── cli/             the Typer command line
├── docs/                CLINICAL · MODELS · ARCHITECTURE · USAGE · EQUITY
├── scripts/             watchdog, training
└── tests/
```

## Limitations

- **Depth is inferred, not measured.** Stage 3 vs 4 and Unstageable hinge on depth and structures a single photo cannot reliably convey. The shading "depth proxy" is a weak, clearly-flagged cue — never a real measurement.
- **Undermining and tunneling are invisible** on the surface and must be entered by a clinician.
- **Scale depends on calibration.** Without DICOM spacing, a fiducial, or a manual `--mm-per-px`, measurements are relative pixel counts, not centimeters.
- **Skin-tone bias** on early-stage and deep-tissue detection, as above.
- **Image quality** — lighting, white balance, focus, angle, and occlusion (dressings, hair, shadows) all move the result.
- **Some scale items are partial.** PUSH, BWAT, and DESIGN-R sub-scores that need palpation or probing are left `null` and reported as partial totals.

## Contributing & policies

[`CONTRIBUTING.md`](CONTRIBUTING.md) · [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) · [`SECURITY.md`](SECURITY.md) · [`CHANGELOG.md`](CHANGELOG.md) · [`DISCLAIMER.md`](DISCLAIMER.md)

Development tasks are wrapped in the [`Makefile`](Makefile): `make check` runs the same gate CI does (lint, the integrity watchdog, and tests). A pre-commit config is provided — `pre-commit install`.

## License

Apache-2.0. © 2026 Ariorad Moniri. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

*Not affiliated with, endorsed by, or a product of NPIAP, EPUAP, PPPIA, or JSPU. Sorbed encodes publicly documented staging criteria from those bodies' guidelines but is an independent open-source project. Obtain the official instruments from the source bodies for verbatim scoring anchors.*
