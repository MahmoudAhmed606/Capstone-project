"""
Report generator.

Reads ``outputs/session_events.json`` + ``session_summary.json`` and
produces ``report.html`` (and optionally ``report.pdf`` via WeasyPrint).

Usage
-----
    python reports/generate_report.py
    python reports/generate_report.py --events outputs/session_events.json --pdf
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate proctoring HTML/PDF report.")
    p.add_argument("--events",  default="outputs/session_events.json")
    p.add_argument("--summary", default="outputs/session_summary.json")
    p.add_argument("--output",  default="outputs/report.html")
    p.add_argument("--pdf",     action="store_true", help="Also generate report.pdf")
    return p.parse_args()


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(p.read_text())


def generate_report(
    events_path:  str = "outputs/session_events.json",
    summary_path: str = "outputs/session_summary.json",
    output_path:  str = "outputs/report.html",
    pdf:          bool = False,
) -> Path:
    """
    Render the Jinja2 HTML template and write the report.

    Returns the path to the generated HTML file.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        raise ImportError("jinja2 not installed. Run: pip install jinja2")

    events_data  = _load_json(events_path)
    summary_data = _load_json(summary_path)

    event_counts  = summary_data.get("event_counts", {})
    high_conf     = summary_data.get("high_confidence_violations", [])
    total_events  = summary_data.get("total_events", len(events_data.get("events", [])))

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    # Add tojson filter for detail dicts
    import json as _json
    env.filters["tojson"] = lambda v: _json.dumps(v)

    template = env.get_template("report.html.j2")
    html = template.render(
        session_id=events_data.get("session_id", "—"),
        session_start=events_data.get("session_start", "—"),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        total_events=total_events,
        high_conf_count=len(high_conf),
        event_counts=event_counts,
        events=events_data.get("events", []),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("HTML report → %s", out)

    if pdf:
        _export_pdf(html, out.with_suffix(".pdf"))

    return out


def _export_pdf(html: str, pdf_path: Path) -> None:
    try:
        from weasyprint import HTML as WPHTML
    except ImportError:
        log.error("weasyprint not installed (pip install weasyprint). Skipping PDF.")
        return
    WPHTML(string=html).write_pdf(str(pdf_path))
    log.info("PDF report → %s", pdf_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    generate_report(args.events, args.summary, args.output, args.pdf)