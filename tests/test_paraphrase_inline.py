
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
