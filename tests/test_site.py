"""The website builds from a reports directory and every page links its assets."""

import json
import re
from pathlib import Path

import pytest

from trendlab.site import build_site

REPORTS = Path(__file__).resolve().parents[1] / "reports"


@pytest.mark.skipif(not (REPORTS / "sections").exists(), reason="reports/ not generated")
def test_site_builds_and_links_resolve(tmp_path):
    index = build_site(REPORTS, tmp_path / "site")
    out = index.parent
    pages = [index, out / "methodology.html", *sorted((out / "studies").glob("*.html"))]
    assert len(pages) == 12
    for page in pages:
        html = page.read_text()
        assert "<title>" in html and 'lang="en"' in html
        for ref in re.findall(r'(?:src|href)="([^"#:]+)"', html):
            assert (page.parent / ref).resolve().exists(), f"{page.name}: missing {ref}"
        for payload in re.findall(
            r'<script type="application/json" class="chart-payload">(.*?)</script>', html, re.S
        ):
            json.loads(payload.replace("<\\/", "</"))


@pytest.mark.skipif(not (REPORTS / "sections").exists(), reason="reports/ not generated")
def test_summary_numbers_come_from_results(tmp_path):
    index = build_site(REPORTS, tmp_path / "site").read_text()
    res = json.loads((REPORTS / "results.json").read_text())["e01_bias_waterfall"]
    claimed = f"{res['original_algo_return'] * 100:+.0f}%"
    corrected = f"{res['corrected_algo_return'] * 100:+.0f}%"
    assert claimed in index and corrected in index
