"""Report assembly: canonical JSON, annotated images, and HTML."""

from __future__ import annotations

from sorbed.report.builder import write_report
from sorbed.report.html_report import build_html

__all__ = ["build_html", "write_report"]
