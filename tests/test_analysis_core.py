"""Segmentation, tissue, and morphometrics tests against known geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sorbed.domain.enums import CalibrationStatus, TissueClass
from sorbed.domain.image import Calibration
from sorbed.imaging import RasterImage, build_metadata
from sorbed.morphometrics.geometry import measure_geometry
from sorbed.segmentation.classical import ClassicalSegmenter
from sorbed.tissue.color_model import ColorTissueClassifier
from tests.synth import ESCHAR, GRANULATION, SLOUGH, blank_image, make_wound


def _raster(rgb: np.ndarray, mm_per_px: float | None = None) -> RasterImage:
    meta = build_metadata(pixels=rgb, source_format="png", bit_depth=8, channels=3)
    if mm_per_px is None:
        cal = Calibration(status=CalibrationStatus.UNCALIBRATED)
    else:
        cal = Calibration(status=CalibrationStatus.MANUAL, mm_per_px=mm_per_px)
    return RasterImage(pixels=rgb, metadata=meta, calibration=cal)


def test_segmenter_recovers_known_ellipse():
    wound = make_wound(axes=(80, 55), fill=GRANULATION, seed=2)
    seg = ClassicalSegmenter().segment(_raster(wound.rgb))
    inter = int((seg.wound_mask & wound.wound_mask).sum())
    union = int((seg.wound_mask | wound.wound_mask).sum())
    iou = inter / union
    assert iou > 0.85
    assert seg.confidence > 0.4


def test_segmenter_blank_image_is_empty_and_low_confidence():
    seg = ClassicalSegmenter().segment(_raster(blank_image()))
    assert seg.area_px == 0
    assert seg.confidence < 0.3


def test_tissue_classifier_reads_colors():
    for fill, expected in [
        (GRANULATION, TissueClass.GRANULATION),
        (SLOUGH, TissueClass.SLOUGH),
        (ESCHAR, TissueClass.ESCHAR),
    ]:
        wound = make_wound(fill=fill, seed=3)
        analysis = ColorTissueClassifier().classify(wound.rgb, wound.wound_mask)
        assert analysis.composition.dominant is expected
        assert analysis.composition.fraction_of(expected) > 0.6


def test_tissue_fractions_sum_to_one():
    wound = make_wound(fill=GRANULATION, inner=((28, 20), SLOUGH), seed=4)
    comp = ColorTissueClassifier().classify(wound.rgb, wound.wound_mask).composition
    assert sum(comp.fractions.values()) == pytest.approx(1.0, abs=1e-3)


def test_geometry_area_recovered_within_tolerance():
    axes = (80, 55)
    wound = make_wound(axes=axes, fill=GRANULATION, seed=5)
    geom = measure_geometry(wound.wound_mask, Calibration(status=CalibrationStatus.UNCALIBRATED))
    true_area = math.pi * axes[0] * axes[1]
    assert geom.area_px == pytest.approx(true_area, rel=0.03)
    assert geom.area_mm2 is None  # uncalibrated
    assert 0.0 <= geom.circularity <= 1.0


def test_geometry_calibrated_area_in_cm2():
    axes = (80, 55)
    wound = make_wound(axes=axes, seed=6)
    cal = Calibration(status=CalibrationStatus.MANUAL, mm_per_px=0.2)
    geom = measure_geometry(wound.wound_mask, cal)
    assert geom.area_mm2 is not None
    expected_mm2 = math.pi * axes[0] * axes[1] * 0.2**2
    assert geom.area_mm2 == pytest.approx(expected_mm2, rel=0.03)
    assert geom.area_cm2 == pytest.approx(expected_mm2 / 100.0, rel=0.03)


def test_geometry_area_monotonic_under_dilation():
    from scipy import ndimage

    wound = make_wound(seed=7)
    base = measure_geometry(wound.wound_mask, Calibration(status=CalibrationStatus.UNCALIBRATED))
    grown = ndimage.binary_dilation(wound.wound_mask, iterations=3)
    grown_geom = measure_geometry(grown, Calibration(status=CalibrationStatus.UNCALIBRATED))
    assert grown_geom.area_px > base.area_px
