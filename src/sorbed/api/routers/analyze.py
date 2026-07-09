"""The analysis endpoint: image upload in, WoundAnalysis out.

Patient images are processed entirely in memory. The uploaded bytes are decoded,
analyzed, and discarded when the request ends; nothing is written to disk by this
handler.
"""

from __future__ import annotations

import base64
import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from sorbed.api.deps import ApiSettings, get_analyzer, get_api_settings
from sorbed.api.schemas import AnalyzeResponse
from sorbed.io import ImageFormat, load_image, sniff_format
from sorbed.pipeline import AnalyzeOptions
from sorbed.visualize.guide import render_guide

router = APIRouter(tags=["analyze"])

_TRUTHY = {"1", "true", "yes", "on"}


def _wants_guide(request: Request, form_value: bool) -> bool:
    """Resolve ``include_guide`` from the query string, falling back to the form."""
    raw = request.query_params.get("include_guide")
    if raw is not None:
        return raw.strip().lower() in _TRUTHY
    return form_value


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="The wound photograph to analyze."),  # noqa: B008
    mm_per_px: float | None = Form(default=None),
    marker_mm: float | None = Form(default=None),
    coin_mm: float | None = Form(default=None),
    include_guide: bool = Form(default=False),
    api_settings: ApiSettings = Depends(get_api_settings),  # noqa: B008
) -> AnalyzeResponse:
    """Analyze one uploaded image and return the serialized WoundAnalysis.

    Validation order: reject oversized uploads (413), then undetectable formats
    (415), then surface pipeline failures as 422.
    """
    max_bytes = api_settings.max_upload_bytes

    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds the {max_bytes} byte limit",
        )

    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds the {max_bytes} byte limit",
        )
    if not data:
        raise HTTPException(status_code=422, detail="uploaded file is empty")

    fmt = sniff_format(data, filename=file.filename)
    if fmt is ImageFormat.UNKNOWN:
        raise HTTPException(
            status_code=415,
            detail="could not detect a supported image format from the upload",
        )

    options = AnalyzeOptions(
        mm_per_px=mm_per_px,
        marker_length_mm=marker_mm,
        coin_diameter_mm=coin_mm,
    )
    try:
        image = load_image(data, mm_per_px=mm_per_px)
        bundle = get_analyzer(request).analyze(image, options)
    except HTTPException:
        raise
    except Exception as exc:  # report any pipeline failure to the caller
        raise HTTPException(status_code=422, detail=f"analysis failed: {exc}") from exc

    payload = bundle.analysis.model_dump(mode="json")

    guide_b64: str | None = None
    if _wants_guide(request, include_guide):
        display_u8 = bundle.display_image.to_uint8_rgb()
        guide_img: Image.Image = render_guide(
            bundle.analysis, display_u8, bundle.wound_mask, bundle.tissue_label_map
        )
        buf = io.BytesIO()
        guide_img.save(buf, format="PNG")
        guide_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return AnalyzeResponse(analysis=payload, guide_png_base64=guide_b64)
