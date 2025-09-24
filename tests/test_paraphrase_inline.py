import pathlib
import sys
from typing import List

from bs4 import BeautifulSoup

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import app


def build_config():
    return app.EngineConfig(
        remote_url="",
        remote_timeout=5.0,
        use_hf_local=False,
        hf_model_id="",
        hf_task="",
        hf_device="cpu",
        hf_max_new_tokens=0,
        hf_chunk_size=0,
        hf_num_beams=0,
        hf_temperature=0.0,
        hf_do_sample=False,
        use_xai=False,
        xai_key="",
        xai_model="",
        use_gemini=False,
        gemini_keys=[],
        cf_zone_id="",
        cf_api_token="",
        date_selector="",
        date_attribute="",
        paraphrase_year_threshold=0,
        paraphrase_month_threshold=0,
        paraphrase_day_threshold=0,
        use_background_paraphrase=False,
    )


def test_inline_placeholders_preserved(monkeypatch):
    config = build_config()

    def fake_paraphrase(text: str, _config: app.EngineConfig, **_kwargs):
        return text.replace("example", "sample").replace(
            " entities", " entities &nbsp; &amp;nbsp;"
        )

    monkeypatch.setattr(app, "paraphrase_text", fake_paraphrase)
    monkeypatch.setattr(app, "paraphrase_in_chunks", fake_paraphrase)

    soup = BeautifulSoup(
        (
            "<p>This is an <em>important</em> example with an <a href='#'>inline link</a> to test "
            "entities across a significantly longer paragraph that should be paraphrased.</p>"
        ),
        "html.parser",
    )
    paragraph = soup.p

    app.paraphrase_element(paragraph, config)

    rendered = str(paragraph)

    assert "<em>important</em>" in rendered
    link = paragraph.find("a")
    assert link is not None
    assert link.get("href") == "#"
    assert "sample" in paragraph.get_text()
    assert "  " in paragraph.get_text()
    assert "&nbsp;" not in rendered
    assert "&amp;nbsp;" not in rendered


def test_private_placeholder_token_used(monkeypatch):
    config = build_config()

    captured: List[str] = []

    def fake_passthrough(text: str, _config: app.EngineConfig, **_kwargs):
        captured.append(text)
        return text

    monkeypatch.setattr(app, "paraphrase_text", fake_passthrough)
    monkeypatch.setattr(app, "paraphrase_in_chunks", fake_passthrough)

    soup = BeautifulSoup(
        (
            "<p>This paragraph contains <em>emphasized</em> inline elements and "
            "an <a href='#'>important link</a> so the placeholder logic engages.</p>"
        ),
        "html.parser",
    )

    app.paraphrase_element(soup.p, config)

    combined = "".join(captured)
    assert any(
        app.INLINE_PLACEHOLDER_BASE <= ord(ch) <= app.INLINE_PLACEHOLDER_MAX
        for ch in combined
    )


class DummyRateLimit(Exception):
    def __init__(self, retry_delay: float | None = None):
        self.code = 429
        if retry_delay is not None:
            self.retry_delay = retry_delay


def test_gemini_waits_for_rate_limits(monkeypatch):
    config = build_config()
    config.use_gemini = True
    config.gemini_keys = ["key-1"]

    app.GEMINI_KEY_COOLDOWNS.clear()
    app.GEMINI_KEY_BACKOFF.clear()

    fake_now = 0.0

    def fake_time() -> float:
        return fake_now

    def fake_sleep(seconds: float) -> None:
        nonlocal fake_now
        fake_now += seconds

    call_count = {"value": 0}

    def fake_gemini(text: str, api_key: str) -> str:
        call_count["value"] += 1
        if call_count["value"] < 3:
            raise DummyRateLimit()
        return "paraphrased"

    monkeypatch.setattr(app, "paraphrase_with_gemini", fake_gemini)
    monkeypatch.setattr(app.time, "time", fake_time)
    monkeypatch.setattr(app.time, "sleep", fake_sleep)

    result = app.paraphrase_text("This content should be paraphrased.", config)

    assert result == "paraphrased"
    assert call_count["value"] == 3
    assert fake_now >= app.GEMINI_MIN_COOLDOWN
