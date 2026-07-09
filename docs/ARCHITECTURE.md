# Architecture

Sorbed is built as a **pipeline of pure, typed transforms.** Each stage takes an immutable analysis context, computes something from it, and returns a new context with additional fields — it never mutates what came before, and it never reaches sideways for state it wasn't handed. Because every stage's input and output is a serializable pydantic model, any point in the pipeline can be snapshotted, replayed, cached, or inspected. This is the property that makes the system auditable: a clinician (or a developer, or a test) can stop the pipeline at any stage and read exactly what was known at that moment.

## Data flow

The pipeline is a directed sequence. Raw bytes come in on the left; a mask, an annotated overlay, a canonical JSON record, and a human-readable report come out on the right.

```
                 ┌─────────────┐
 image bytes ──▶ │   Ingest    │  sniff format / magic bytes
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   Decode    │  per-format adapters → normalized array
                 │ & Normalize │
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │ Preprocess  │  color normalization, calibration (fiducial → mm_per_px)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   Wound     │  where is the wound? → boundary mask
                 │Segmentation │
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   Tissue    │  what is in the bed? → per-tissue regions
                 │Segmentation │
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │Morphometrics│  area, perimeter, tissue % (px and, if calibrated, cm)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  Staging    │  rule ⊕ ML ⊕ arbiter → StageDecision
                 │   Engine    │
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │Explainability│ evidence trace over the above
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   Outputs   │  mask PNG · overlay/guide PNG · canonical JSON · HTML/PDF
                 └─────────────┘
```

The staging engine is deliberately drawn as `rule ⊕ ML ⊕ arbiter`: a deterministic rule layer and a machine-learning head produce candidate stages, and an arbiter reconciles them, preferring the rules where the guideline is categorical and deferring to a clinician where confidence is low.

## Package layout

The source tree under `src/sorbed/` mirrors the pipeline. Each subpackage owns one responsibility:

| Subpackage | Responsibility |
|---|---|
| `config` | Runtime configuration and settings. |
| `domain` | Pure pydantic v2 data contracts — the types every stage speaks. No logic, no I/O. |
| `io` | Format sniffing, a decoder registry, per-format decoders, and array normalization. |
| `preprocess` | Calibration, fiducial detection, and color normalization. |
| `segmentation` | Wound boundary segmentation — a classical backend plus pluggable model backends. |
| `tissue` | Tissue-type segmentation and classification. |
| `morphometrics` | Geometry: area, perimeter, tissue percentages, and physical measurements when calibrated. |
| `staging` | The staging engine: `features`, `rules`, `ml_head`, `arbiter`, and the `engine` that composes them. |
| `explain` | Builds the evidence trace that justifies each decision. |
| `visualize` | Renders masks, overlays, and clinician guide images. |
| `report` | Assembles HTML/PDF reports. |
| `pipeline` | The orchestrator that wires the stages together. |
| `api` | FastAPI service surface. |
| `cli` | Typer command-line interface. |
| `models` | The model registry — provenance, sha256 verification, and licenses. |

## Domain contracts

The `domain` package holds the pydantic v2 models that flow through the pipeline. The central contract is **`WoundAnalysis`**, the canonical record of a single analysis. Around it sit:

- **`Metrics`**, composed of **`GeometryMetrics`** (area, perimeter, length, width), **`TissueComposition`** (per-tissue percentages and confidence), and **`DepthProxy`** (the depth cue, including its source and whether it is actually known);
- **`StageDecision`**, which carries the chosen stage together with an ordered list of **`Evidence`** items explaining how it was reached;
- **`Report`**, the presentation-layer model that the HTML/PDF renderers consume.

Because these are ordinary pydantic models, the entire analysis serializes to and from JSON losslessly.

## Key invariants

These properties are enforced across the codebase and are what make the system trustworthy:

- **The context grows monotonically.** Stages only *add* fields; they never overwrite or delete what an earlier stage established. The history of a computation is always fully present.
- **`mm_per_px` is Optional and is never fabricated.** If no scale fiducial was found, calibration is absent, and every physical (centimeter) measurement is reported as **null**. Pixel-based metrics still compute normally — the system degrades honestly rather than inventing a scale.
- **The canonical `WoundAnalysis` JSON is the single source of truth.** The mask PNG, the overlay, the HTML, and the PDF are all *renderings* of that JSON. Nothing downstream computes anything the canonical record doesn't already contain.
- **Model inference lives behind adapters.** No pipeline stage imports `torch` directly; segmentation and staging call into backend interfaces. This is what lets the ML backends be swapped, disabled, or run out-of-process.
- **A weight-free classical backend guarantees end-to-end operation with zero downloads.** The pipeline always produces a complete result, even with no models fetched and no network access.
- **Every emitted number traces to a real pixel computation.** There are no placeholder values, demo constants, or hand-waved estimates. If a value can't be computed from the actual image, it is null, not fabricated.

## Technology stack

The implementation targets **Python 3.12** with a conventional, well-supported scientific stack: **numpy**, **OpenCV**, and **scikit-image** for image processing and morphometrics; **pydantic** (v2) for the domain contracts; **Typer** for the CLI; and **FastAPI** for the service API. The design keeps the core dependency-light so that the classical path runs anywhere, with the ML backends layered on top only when explicitly enabled.
