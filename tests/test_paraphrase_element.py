import importlib.util
import pathlib
import sys

from bs4 import BeautifulSoup
import pytest


APP_MODULE_NAME = "app_module_under_test"
APP_PATH = pathlib.Path(__file__).resolve().parent.parent / "app.py"
spec = importlib.util.spec_from_file_location(APP_MODULE_NAME, APP_PATH)
app = importlib.util.module_from_spec(spec)
sys.modules[APP_MODULE_NAME] = app
assert spec.loader is not None
spec.loader.exec_module(app)


def _build_config() -> app.EngineConfig:
    return app.build_engine_config({})


def test_inline_tag_paraphrase_continuation_lowercase(monkeypatch):
    html = (
        "<p>"
        "This introduction contains sufficient characters to trigger paraphrasing before the link. "
        "<a href='#'>link</a>"
        " Continuation Text After Inline Element Should Become Lowercase Even If It Starts Uppercase."
        "</p>"
    )
    soup = BeautifulSoup(html, "html.parser")
    paragraph = soup.p
    config = _build_config()

    def fake_paraphrase_text(text: str, _config: app.EngineConfig) -> str:
        if "Continuation" in text:
            return (
                "Continuation Text After Inline Element Should Become Lowercase "
                "Even If It Starts Uppercase."
            )
        return (
            "Rewritten introduction that remains obviously different and lengthy "
            "enough to keep the paraphraser engaged."
        )

    monkeypatch.setattr(app, "paraphrase_text", fake_paraphrase_text)
    monkeypatch.setattr(app, "paraphrase_in_chunks", lambda text, cfg, max_len=150: fake_paraphrase_text(text, cfg))

    app.paraphrase_element(paragraph, config)

    link_node = paragraph.find("a")
    assert link_node is not None
    trailing_text = link_node.next_sibling
    assert isinstance(trailing_text, str)

    first_alpha = next((ch for ch in trailing_text if ch.isalpha()), "")
    assert first_alpha == first_alpha.lower() != ""
