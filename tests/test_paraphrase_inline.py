import pathlib
import sys

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
        gemini_wait_for_available=0.0,
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


def test_inline_segments_dont_split_words(monkeypatch):
    config = build_config()

    def fake_paraphrase(text: str, _config: app.EngineConfig, **_kwargs):
        # Simulate a model that inserts spaces around placeholders
        return text.replace("[[INLINE_0]]", " [[INLINE_0]] ")

    monkeypatch.setattr(app, "paraphrase_text", fake_paraphrase)
    monkeypatch.setattr(app, "paraphrase_in_chunks", fake_paraphrase)

    soup = BeautifulSoup(
        (
            "<p>This sentence is intentionally lengthy so the paraphraser runs on the segment "
            "while verifying that s<em>tipulated</em> remains a single word in the output even "
            "if spacing is altered by the model.</p>"
        ),
        "html.parser",
    )
    paragraph = soup.p

    app.paraphrase_element(paragraph, config)

    text = paragraph.get_text()
    assert "stipulated" in text
    assert "s tipulated" not in text
    assert "  " not in text


def test_gemini_waits_instead_of_spamming_keys(monkeypatch):
    config = build_config()
    config.use_gemini = True
    config.gemini_keys = ["key-1", "key-2"]
    config.gemini_wait_for_available = 10.0

    monkeypatch.setattr(app, "GEMINI_KEY_COOLDOWNS", {})
    monkeypatch.setattr(app, "GEMINI_KEY_BACKOFF", {})

    current_time = {"value": 0.0}
    sleep_calls = []

    def fake_time() -> float:
        return current_time["value"]

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)
        current_time["value"] += duration

    monkeypatch.setattr(app.time, "time", fake_time)
    monkeypatch.setattr(app.time, "sleep", fake_sleep)

    call_history = []

    class DummyDelay:
        def total_seconds(self) -> float:
            return 2.0

    class GeminiRateLimit(RuntimeError):
        pass

    def fake_paraphrase_with_gemini(text: str, api_key: str) -> str:
        call_history.append((api_key, current_time["value"]))
        if len(call_history) == 1:
            exc = GeminiRateLimit("429 Resource exhausted; retry after 2 seconds")
            exc.code = 429
            exc.retry_delay = DummyDelay()
            raise exc
        return f"gemini:{api_key}"

    monkeypatch.setattr(app, "paraphrase_with_gemini", fake_paraphrase_with_gemini)

    result = app.paraphrase_text("Sample text to paraphrase", config)

    assert result == "gemini:key-2"
    assert call_history == [("key-1", 0.0), ("key-2", 2.0)]
    assert sleep_calls == [2.0]
