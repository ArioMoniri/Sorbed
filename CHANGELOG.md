# Changelog

All notable changes to Sorbed are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding, governance, and community health files (Apache-2.0
  license, code of conduct, contributing guide, security policy, clinical
  disclaimer).
- Core analysis pipeline: multi-format image ingestion, preprocessing,
  segmentation, tissue analysis, morphometrics, explainable staging engine,
  visualization, and reporting.
- Command-line interface and FastAPI service.
- `scripts/watchdog.py` integrity guard that fails the build on placeholder,
  stub, or fabricated-output code paths.
- Clinical **directive grounding**: build a citable "directive pack" from a
  hospital's pressure-injury guideline PDF (`scripts/build_directive_pack.py`)
  and ground every report in its numbered sections and figures
  (`sorbed.guidelines`).
- **PDF reports** (`sorbed.report`, optional `[pdf]` extra): a directive-grounded
  grading report and a longitudinal follow-up report with a healing verdict,
  inline SVG trend charts, and directive-cited alerts. Modern design system with
  the vendored Manrope (OFL) typeface and grade/confidence colouring.
- Healing **follow-up alerts** (`sorbed.trend.alerts`): a better/worse verdict
  plus ranked, directive-cited signals (area/PAR, PUSH, granulation, necrosis,
  stage progression).
- Segmentation trainer gains `--decoder-attention scse` (spatial-and-channel
  Squeeze-and-Excitation), the FUSegNet-style attention, plus documented
  EfficientNet-encoder training toward the AZH/FUSeg benchmark.

<!--
Template for future releases:

## [x.y.z] - YYYY-MM-DD
### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
-->

[Unreleased]: https://github.com/ArioMoniri/Sorbed/commits/main
