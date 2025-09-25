from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import json
import gzip
import shutil
from pathlib import Path
from urllib.parse import quote_plus, urlsplit
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Iterator, List, Set, Tuple

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

try:
    from transformers import pipeline  # type: ignore

    TRANSFORMERS_AVAILABLE = True
    HF_IMPORT_ERROR = None
except Exception as exc:
    TRANSFORMERS_AVAILABLE = False
    HF_IMPORT_ERROR = exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="Clone Proxy with Remote Paraphrasing")
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

DB_PATH = "config.db"
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
PAGE_CACHE_DIR = "page_cache"
os.makedirs(PAGE_CACHE_DIR, exist_ok=True)

ARTICLE_SELECTORS = [
    ".article-body",
    ".story-body",
    ".article-content",
    ".post-content",
    ".entry-content",
    ".content",
    ".main-content",
    "article",
]


REMOTE_PARAPHRASE_DEFAULT = "http://154.53.41.163:8000/paraphrase"


GEMINI_KEY_COOLDOWNS: Dict[str, float] = {}
GEMINI_KEY_BACKOFF: Dict[str, float] = {}
GEMINI_MIN_COOLDOWN = 1.0
GEMINI_MAX_COOLDOWN = 60.0
GEMINI_MAX_WAIT_FOR_AVAILABLE = 5.0


# ------------------------------------------------------------------------------
# Database & config helpers
# ------------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE,
            password_hash TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS sites (
            id INTEGER PRIMARY KEY,
            original_domain TEXT,
            cloned_domain TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS configs (
            id INTEGER PRIMARY KEY,
            site_id INTEGER,
            key TEXT,
            value TEXT,
            FOREIGN KEY (site_id) REFERENCES sites (id)
        )
        """
    )

    if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            ("admin@admin.com", password_hash),
        )

    if c.execute("SELECT COUNT(*) FROM sites").fetchone()[0] == 0:
        c.execute(
            "INSERT INTO sites (id, original_domain, cloned_domain) VALUES (?, ?, ?)",
            (1, "https://example.com", "https://yourdomain.com"),
        )

    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def save_config(cursor: sqlite3.Cursor, key: str, value: str, site_id: int = 1) -> None:
    cursor.execute("DELETE FROM configs WHERE site_id = ? AND key = ?", (site_id, key))
    cursor.execute(
        "INSERT INTO configs (site_id, key, value) VALUES (?, ?, ?)",
        (site_id, key, value),
    )


def as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_gemini_keys(value: str | None) -> List[str]:
    if not value:
        return []
    return [line.strip() for line in value.splitlines() if line.strip()]


def build_cache_key(text: str, engine_id: str, variant: str = "") -> str:
    base_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    variant_hash = hashlib.md5(f"{engine_id}:{variant}".encode("utf-8")).hexdigest()
    return f"{base_hash}_{variant_hash}"


def get_cached_paraphrase(cache_key: str) -> str | None:
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as file:
            return file.read()
    return None


def set_cached_paraphrase(cache_key: str, paraphrased: str) -> None:
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
    with open(cache_file, "w", encoding="utf-8") as file:
        file.write(paraphrased)


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    retry_delay = getattr(exc, "retry_delay", None)
    if retry_delay is not None:
        try:
            if hasattr(retry_delay, "total_seconds"):
                seconds = float(retry_delay.total_seconds())
            else:
                seconds = float(retry_delay)
            if seconds > 0:
                return seconds
        except Exception:
            pass

    retry_info = getattr(exc, "retry_info", None)
    if retry_info is not None:
        retry_delay = getattr(retry_info, "retry_delay", None)
        if retry_delay is not None:
            seconds = float(getattr(retry_delay, "seconds", 0))
            nanos = float(getattr(retry_delay, "nanos", 0))
            total = seconds + nanos / 1_000_000_000
            if total > 0:
                return total

    metadata = getattr(exc, "metadata", None)
    if isinstance(metadata, dict):
        for header in ("retry-after", "Retry-After"):
            if header in metadata:
                try:
                    return float(metadata[header])
                except Exception:
                    continue

    message = " ".join(str(part) for part in getattr(exc, "args", ()) if part)
    if not message:
        message = str(exc)
    if message:
        match = re.search(r"retry[-\s]?after[^\d]*(\d+(?:\.\d+)?)", message, re.IGNORECASE)
        if match:
            try:
                seconds = float(match.group(1))
                if seconds > 0:
                    return seconds
            except Exception:
                pass

    return None


def _classify_gemini_exception(exc: Exception) -> Tuple[bool, float | None]:
    is_rate_limited = False
    try:
        from google.api_core import exceptions as google_exceptions  # type: ignore

        if isinstance(exc, google_exceptions.ResourceExhausted):
            is_rate_limited = True
        else:
            too_many_requests = getattr(google_exceptions, "TooManyRequests", None)
            if too_many_requests is not None and isinstance(exc, too_many_requests):
                is_rate_limited = True
    except Exception:
        pass

    if not is_rate_limited:
        code = getattr(exc, "code", None)
        if code == 429:
            is_rate_limited = True
        else:
            message = str(exc).lower()
            if "429" in message or "rate limit" in message or "resource exhausted" in message:
                is_rate_limited = True

    retry_delay = _extract_retry_delay_seconds(exc) if is_rate_limited else None
    return is_rate_limited, retry_delay


def clear_directory(directory: str) -> None:
    base = Path(directory)
    if not base.exists() or not base.is_dir():
        return

    for entry in base.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except Exception as exc:
            logger.warning("Unable to delete %s: %s", entry, exc)


# ------------------------------------------------------------------------------
# Auth helper
# ------------------------------------------------------------------------------

def verify_user(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE email = ?", (credentials.username,))
    row = c.fetchone()
    conn.close()

    valid = (
        row
        and hashlib.sha256(credentials.password.encode()).hexdigest() == row["password_hash"]
    )
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ------------------------------------------------------------------------------
# Paraphrasing helpers
# ------------------------------------------------------------------------------

HF_PIPELINES: Dict[tuple[str, str, str], object] = {}


def resolve_device(device: str) -> tuple[int | None, dict]:
    device = (device or "cpu").strip().lower()
    if device.startswith("cuda"):
        if device == "cuda":
            return 0, {}
        try:
            _, index = device.split(":", 1)
            return int(index), {}
        except ValueError:
            return 0, {}
    if device.startswith("mps"):
        return None, {"device_map": "auto"}
    return -1, {}


def get_hf_pipeline(model_id: str, task: str, device: str):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "transformers is not installed. Install it with 'pip install transformers sentencepiece'."
        ) from HF_IMPORT_ERROR

    cache_key = (model_id, task, device)
    if cache_key in HF_PIPELINES:
        return HF_PIPELINES[cache_key]

    device_index, extra_kwargs = resolve_device(device)
    logger.info(
        "Loading Hugging Face pipeline for model '%s' (task=%s, device=%s)",
        model_id,
        task,
        device,
    )
    if device_index is None:
        generator = pipeline(task, model=model_id, tokenizer=model_id, **extra_kwargs)
    else:
        generator = pipeline(
            task,
            model=model_id,
            tokenizer=model_id,
            device=device_index,
            **extra_kwargs,
        )

    HF_PIPELINES[cache_key] = generator
    return generator


def chunk_text_for_hf(text: str, max_chars: int) -> Iterator[str]:
    if len(text) <= max_chars:
        yield text
        return

    sentences = re.split(r"(?<=[.!?])\s+", text)
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        if not sentence:
            continue
        if current_len + len(sentence) + 1 <= max_chars:
            current.append(sentence)
            current_len += len(sentence) + 1
        else:
            if current:
                yield " ".join(current)
            current = [sentence]
            current_len = len(sentence)
    if current:
        yield " ".join(current)


@dataclass
class EngineConfig:
    remote_url: str
    remote_timeout: float
    use_hf_local: bool
    hf_model_id: str
    hf_task: str
    hf_device: str
    hf_max_new_tokens: int
    hf_chunk_size: int
    hf_num_beams: int
    hf_temperature: float
    hf_do_sample: bool
    use_xai: bool
    xai_key: str
    xai_model: str
    use_gemini: bool
    gemini_keys: List[str]
    gemini_wait_for_available: float
    cf_zone_id: str
    cf_api_token: str
    date_selector: str
    date_attribute: str
    paraphrase_year_threshold: int
    paraphrase_month_threshold: int
    paraphrase_day_threshold: int
    use_background_paraphrase: bool


def paraphrase_with_remote(text: str, url: str, timeout: float) -> str:
    cache_key = build_cache_key(text, "remote", url)
    cached = get_cached_paraphrase(cache_key)
    if cached:
        logger.info("Using cached remote paraphrase")
        return cached

    payload = {"text": text}
    logger.info("Calling remote paraphrase endpoint %s", url)
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()

    data = resp.json()
    result: str | None = None
    if isinstance(data, dict):
        for key in ("paraphrased", "result", "text", "output"):
            value = data.get(key)
            if isinstance(value, str):
                result = value
                break
    elif isinstance(data, str):
        result = data

    if not result:
        raise RuntimeError(f"Unexpected response structure from {url!r}: {data!r}")

    set_cached_paraphrase(cache_key, result)
    return result


def paraphrase_with_huggingface_local(text: str, config: EngineConfig) -> str:
    cache_key = build_cache_key(
        text,
        "hf",
        f"{config.hf_model_id}:{config.hf_max_new_tokens}:{config.hf_num_beams}:"
        f"{config.hf_do_sample}:{config.hf_temperature}:{config.hf_chunk_size}",
    )
    cached = get_cached_paraphrase(cache_key)
    if cached:
        logger.info("Using cached local Hugging Face paraphrase")
        return cached

    generator = get_hf_pipeline(config.hf_model_id, config.hf_task, config.hf_device)

    paraphrased_segments: List[str] = []
    for segment in chunk_text_for_hf(text, max(256, config.hf_chunk_size)):
        generation_kwargs = {
            "max_new_tokens": config.hf_max_new_tokens,
            "num_beams": config.hf_num_beams,
            "do_sample": config.hf_do_sample,
        }
        if config.hf_do_sample:
            generation_kwargs["temperature"] = max(0.1, config.hf_temperature)

        start = time.time()
        result = generator(segment, truncation=True, **generation_kwargs)
        elapsed = time.time() - start
        logger.info("Local Hugging Face generation finished in %.2fs", elapsed)
        generated_text = result[0]["generated_text"]
        paraphrased_segments.append(generated_text or segment)

    paraphrased = "\n\n".join(paraphrased_segments)
    set_cached_paraphrase(cache_key, paraphrased)
    return paraphrased


def paraphrase_with_gemini(text: str, api_key: str) -> str:
    import google.generativeai as genai

    cache_key = build_cache_key(text, "gemini", api_key[:10])
    cached = get_cached_paraphrase(cache_key)
    if cached:
        logger.info("Using cached Gemini paraphrase")
        return cached

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

    prompt = (
        "Paraphrase the following text while keeping the meaning and structure clear. "
        "Keep HTML entities untouched and avoid adding extra commentary.\n\n" + text
    )

    start_time = time.time()
    response = model.generate_content(prompt)
    elapsed = time.time() - start_time
    logger.info("Gemini API call took %.2f seconds", elapsed)

    paraphrased = response.text if response.text else text
    set_cached_paraphrase(cache_key, paraphrased)
    return paraphrased


def paraphrase_with_xai(text: str, api_key: str, model: str) -> str:
    cache_key = build_cache_key(text, "xai", f"{model}:{api_key[:10]}")
    cached = get_cached_paraphrase(cache_key)
    if cached:
        logger.info("Using cached xAI paraphrase")
        return cached

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    prompt = (
        "Paraphrase the following text while keeping the meaning, tone, and approximate length. "
        "Do not add explanations or remove HTML tags if present.\n\n" + text
    )

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "max_tokens": 800,
        "temperature": 0.7,
    }

    logger.info("Calling xAI model %s", model)
    response = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        logger.warning("Empty response from xAI; returning original text")
        return text

    paraphrased = content if content is not None else text
    set_cached_paraphrase(cache_key, paraphrased)
    return paraphrased


# ------------------------------------------------------------------------------
# Core paraphrasing logic
# ------------------------------------------------------------------------------

def build_engine_config(config_map: Dict[str, str]) -> EngineConfig:
    remote_url_raw = config_map.get("hf_remote_url")
    if remote_url_raw is None:
        remote_url = REMOTE_PARAPHRASE_DEFAULT
    else:
        remote_url = remote_url_raw.strip()
    requested_timeout = as_float(config_map.get("hf_remote_timeout"), default=45.0)
    remote_timeout = max(5.0, min(120.0, requested_timeout))
    if remote_timeout != requested_timeout:
        logger.warning(
            "Clamping remote paraphrase timeout from %s to %s seconds to keep workers responsive",
            requested_timeout,
            remote_timeout,
        )

    gemini_key_raw = config_map.get("gemini_key", "")

    return EngineConfig(
        remote_url=remote_url,
        remote_timeout=remote_timeout,
        use_hf_local=as_bool(config_map.get("use_huggingface_local"), default=False),
        hf_model_id=config_map.get("hf_model_id", "humarin/chatgpt_paraphraser_on_T5_base"),
        hf_task=config_map.get("hf_task", "text2text-generation"),
        hf_device=config_map.get("hf_device", "cpu"),
        hf_max_new_tokens=as_int(config_map.get("hf_max_new_tokens"), default=256),
        hf_chunk_size=as_int(config_map.get("hf_chunk_size"), default=1200),
        hf_num_beams=as_int(config_map.get("hf_num_beams"), default=4),
        hf_temperature=as_float(config_map.get("hf_temperature"), default=0.7),
        hf_do_sample=as_bool(config_map.get("hf_do_sample"), default=False),
        use_xai=as_bool(config_map.get("use_xai"), default=False),
        xai_key=config_map.get("xai_key", ""),
        xai_model=config_map.get("xai_model", "grok-code-fast-1"),
        use_gemini=as_bool(config_map.get("use_gemini"), default=False),
        gemini_keys=parse_gemini_keys(gemini_key_raw),
        gemini_wait_for_available=max(0.0, as_float(config_map.get("gemini_wait_for_available"), default=30.0)),
        cf_zone_id=config_map.get("cf_zone_id", ""),
        cf_api_token=config_map.get("cf_api_token", ""),
        date_selector=config_map.get("date_selector", "").strip(),
        date_attribute=config_map.get("date_attribute", "").strip(),
        paraphrase_year_threshold=as_int(config_map.get("paraphrase_year_threshold"), default=2025),
        paraphrase_month_threshold=as_int(config_map.get("paraphrase_month_threshold"), default=1),
        paraphrase_day_threshold=as_int(config_map.get("paraphrase_day_threshold"), default=1),
        use_background_paraphrase=as_bool(config_map.get("use_background_paraphrase"), default=False),
    )


def paraphrase_text(text: str, config: EngineConfig) -> str:
    errors: List[str] = []

    if config.remote_url:
        try:
            return paraphrase_with_remote(text, config.remote_url, config.remote_timeout)
        except Exception as exc:
            logger.exception("Remote paraphrasing failed: %s", exc)
            errors.append(f"Remote: {exc}")

    if config.use_hf_local:
        try:
            return paraphrase_with_huggingface_local(text, config)
        except Exception as exc:
            logger.exception("Local Hugging Face paraphrasing failed: %s", exc)
            errors.append(f"Local HF: {exc}")

    if config.use_xai and config.xai_key:
        try:
            return paraphrase_with_xai(text, config.xai_key, config.xai_model or "grok-code-fast-1")
        except Exception as exc:
            logger.exception("xAI paraphrasing failed: %s", exc)
            errors.append(f"xAI: {exc}")

    if config.use_gemini:
        if not config.gemini_keys:
            errors.append("Gemini: no API keys configured")
        else:
            total_waited = 0.0
            wait_budget = max(0.0, config.gemini_wait_for_available)

            while True:
                remaining_budget = max(0.0, wait_budget - total_waited)
                ordered_keys = sorted(
                    enumerate(config.gemini_keys, start=1),
                    key=lambda item: GEMINI_KEY_COOLDOWNS.get(item[1], 0.0),
                )
                attempted = False
                had_non_rate_limit_error = False

                for idx, api_key in ordered_keys:
                    cooldown_until = GEMINI_KEY_COOLDOWNS.get(api_key, 0.0)
                    wait_remaining = cooldown_until - time.time()
                    if wait_remaining > 0:
                        continue

                    attempted = True
                    GEMINI_KEY_BACKOFF.setdefault(api_key, GEMINI_MIN_COOLDOWN)

                    try:
                        result = paraphrase_with_gemini(text, api_key)
                    except Exception as exc:
                        is_rate_limited, retry_delay = _classify_gemini_exception(exc)
                        if is_rate_limited:
                            fallback_delay = GEMINI_KEY_BACKOFF.get(api_key, GEMINI_MIN_COOLDOWN)
                            delay = max(retry_delay or 0.0, fallback_delay)
                            delay = min(max(delay, GEMINI_MIN_COOLDOWN), GEMINI_MAX_COOLDOWN)
                            GEMINI_KEY_COOLDOWNS[api_key] = time.time() + delay
                            GEMINI_KEY_BACKOFF[api_key] = min(delay * 2, GEMINI_MAX_COOLDOWN)
                            logger.warning(
                                "Gemini key %d hit rate limit; cooling down for %.2f seconds",
                                idx,
                                delay,
                            )
                            errors.append(
                                f"Gemini[{idx}]: rate limited (cooldown {delay:.1f}s)"
                            )
                            if remaining_budget > 0:
                                break
                            continue
                        else:
                            had_non_rate_limit_error = True
                            GEMINI_KEY_BACKOFF[api_key] = GEMINI_MIN_COOLDOWN
                            logger.exception(
                                "Gemini paraphrasing failed with key %d: %s", idx, exc
                            )
                            errors.append(f"Gemini[{idx}]: {exc}")
                            continue

                    GEMINI_KEY_COOLDOWNS.pop(api_key, None)
                    GEMINI_KEY_BACKOFF[api_key] = GEMINI_MIN_COOLDOWN
                    return result

                if had_non_rate_limit_error or wait_budget <= 0:
                    break

                now = time.time()
                wait_candidates: List[float] = []
                for api_key in config.gemini_keys:
                    cooldown_until = GEMINI_KEY_COOLDOWNS.get(api_key, 0.0)
                    remaining = cooldown_until - now
                    if remaining > 0:
                        wait_candidates.append(remaining)

                if not wait_candidates:
                    break

                remaining_budget = max(0.0, wait_budget - total_waited)
                if remaining_budget <= 0:
                    break

                wait_for = min(min(wait_candidates), remaining_budget)
                if wait_for <= 0:
                    break

                if not attempted:
                    logger.info(
                        "All Gemini keys are cooling down; waiting %.2f seconds for the earliest key",
                        wait_for,
                    )
                else:
                    logger.info(
                        "Gemini rate limits encountered; waiting %.2f seconds before retrying",
                        wait_for,
                    )

                time.sleep(wait_for)
                total_waited += wait_for

            # end while True

    if errors:
        logger.warning("All paraphrasing engines failed; returning original text. Reasons: %s", "; ".join(errors))
    return text



def paraphrase_element(element, config: EngineConfig) -> None:
    if element is None:
        return

    protected_inline = {
        "a",
        "em",
        "i",
        "strong",
        "b",
        "span",
        "u",
    }

    for child in list(element.children):
        if isinstance(child, Tag):
            paraphrase_element(child, config)

    contents = list(element.contents)
    if not contents:
        return

    rebuilt: List = []
    segment_nodes: List = []

    def normalize_entities(value: str) -> str:
        return re.sub(r"&(?:amp;)?nbsp;", " ", value)

    def append_nodes_from_html(html: str) -> None:
        fragment = BeautifulSoup(html, "html.parser")
        if fragment.body is not None:
            nodes = list(fragment.body.contents)
        else:
            nodes = list(fragment.contents)
        for node in nodes:
            rebuilt.append(node)

    def flush_segment() -> None:
        nonlocal segment_nodes
        if not segment_nodes:
            return

        placeholder_map: List[Tuple[str, str]] = []
        combined_parts: List[str] = []
        inline_counter = 0
        text_length = 0
        original_text_fragments: List[str] = []

        for node in segment_nodes:
            if isinstance(node, NavigableString):
                text = str(node)
                combined_parts.append(text)
                text_length += len(text.strip())
                original_text_fragments.append(text)
            elif isinstance(node, Tag) and node.name and node.name.lower() in protected_inline:
                placeholder = f"[[INLINE_{inline_counter}]]"
                inline_counter += 1
                placeholder_map.append((placeholder, str(node)))
                combined_parts.append(placeholder)
                original_text_fragments.append(node.get_text("", strip=False))

        combined = "".join(combined_parts)
        if not combined:
            segment_nodes = []
            return

        leading_match = re.match(r"^\s*", combined)
        trailing_match = re.search(r"\s*$", combined)
        leading_ws = leading_match.group(0) if leading_match else ""
        trailing_ws = trailing_match.group(0) if trailing_match else ""
        core = combined.strip()

        should_paraphrase = text_length >= 50 and core
        result_core = core

        if should_paraphrase:
            logger.info("Paraphrasing text segment with %d characters", text_length)
            if len(core) > 150:
                result_core = paraphrase_in_chunks(core, config, max_len=150) or core
            else:
                result_core = paraphrase_text(core, config) or core

            original_text = "".join(original_text_fragments)
            original_alpha = next((ch for ch in original_text if ch.isalpha()), None)
            if original_alpha and result_core:
                for idx, ch in enumerate(result_core):
                    if ch.isalpha():
                        if original_alpha.isupper():
                            desired = ch.upper()
                        elif original_alpha.islower():
                            desired = ch.lower()
                        else:
                            desired = ch
                        if desired != ch:
                            result_core = (
                                result_core[:idx] + desired + result_core[idx + 1 :]
                            )
                        break

        placeholder_context: Dict[str, Tuple[str, str]] = {}
        search_start = 0
        for placeholder, _html in placeholder_map:
            idx = combined.find(placeholder, search_start)
            if idx == -1:
                continue
            pre_ws_start = idx - 1
            while pre_ws_start >= 0 and combined[pre_ws_start].isspace():
                pre_ws_start -= 1
            pre_ws = combined[pre_ws_start + 1 : idx]
            post_index = idx + len(placeholder)
            post_ws_end = post_index
            while post_ws_end < len(combined) and combined[post_ws_end].isspace():
                post_ws_end += 1
            post_ws = combined[post_index:post_ws_end]
            placeholder_context[placeholder] = (pre_ws, post_ws)
            search_start = idx + len(placeholder)

        for placeholder, html in placeholder_map:
            pre_ws, post_ws = placeholder_context.get(placeholder, ("", ""))

            idx = result_core.find(placeholder)
            while idx != -1:
                trim_start = idx
                while trim_start > 0 and result_core[trim_start - 1].isspace():
                    trim_start -= 1
                if trim_start < idx:
                    current_ws = result_core[trim_start:idx]
                    if pre_ws:
                        if current_ws != pre_ws:
                            result_core = (
                                result_core[:trim_start]
                                + pre_ws
                                + result_core[idx:]
                            )
                            idx = trim_start + len(pre_ws)
                            idx = result_core.find(placeholder, idx)
                            continue
                    else:
                        result_core = result_core[:trim_start] + result_core[idx:]
                        idx = trim_start
                        idx = result_core.find(placeholder, idx)
                        continue

                end_idx = idx + len(placeholder)
                trim_end = end_idx
                while trim_end < len(result_core) and result_core[trim_end].isspace():
                    trim_end += 1
                if trim_end > end_idx:
                    current_ws = result_core[end_idx:trim_end]
                    if post_ws:
                        if current_ws != post_ws:
                            result_core = (
                                result_core[:end_idx]
                                + post_ws
                                + result_core[trim_end:]
                            )
                            idx = result_core.find(placeholder, idx)
                            continue
                    else:
                        result_core = result_core[:end_idx] + result_core[trim_end:]
                        idx = result_core.find(placeholder, idx)
                        continue

                idx = result_core.find(placeholder, idx + len(placeholder))

            result_core = result_core.replace(placeholder, html)

        replacement_html = f"{leading_ws}{result_core}{trailing_ws}" if core else combined
        replacement_html = normalize_entities(replacement_html)
        append_nodes_from_html(replacement_html)
        segment_nodes = []

    for node in contents:
        if isinstance(node, NavigableString):
            segment_nodes.append(node)
        elif isinstance(node, Tag) and node.name and node.name.lower() in protected_inline:
            segment_nodes.append(node)
        else:
            flush_segment()
            rebuilt.append(node)

    flush_segment()

    if rebuilt:
        element.clear()
        for node in rebuilt:
            element.append(node)

def ensure_lazy_social_scripts(soup: BeautifulSoup) -> None:
    if soup is None:
        return
    body = soup.body
    if body is None:
        return

    existing_srcs: Set[str] = {sc.get("src") for sc in soup.find_all("script") if sc.get("src")}
    to_append: Set[str] = set()
    for div in soup.select("div.lazy-social-embed"):
        src = (div.get("data-src") or "").strip()
        if src and src not in existing_srcs:
            to_append.add(src)

    for src in sorted(to_append):
        script_tag = soup.new_tag("script", src=src)
        script_tag.attrs["async"] = ""
        body.append(script_tag)


DATE_META_KEYS = {
    "article:published_time",
    "article:modified_time",
    "og:published_time",
    "og:updated_time",
    "pubdate",
    "publishdate",
    "publish-date",
    "publication_date",
    "date",
    "datepublished",
    "datecreated",
    "datemodified",
    "dc.date",
    "dc.date.issued",
    "dc.date.published",
}

JSONLD_DATE_KEYS = {"datePublished", "dateCreated", "dateModified", "uploadDate"}

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d %B %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %b %Y",
    "%B %d %Y",
]

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


# Sentence splitter used for chunking long text runs before paraphrasing
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_chunks(text: str, max_len: int = 150) -> List[str]:
    parts: List[str] = []
    # First split on sentence boundaries
    for seg in SENT_SPLIT_RE.split(text):
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) <= max_len:
            parts.append(seg)
        else:
            # Hard-wrap segments that are still too long
            for i in range(0, len(seg), max_len):
                chunk = seg[i : i + max_len].strip()
                if chunk:
                    parts.append(chunk)
    return parts


def paraphrase_in_chunks(text: str, config: "EngineConfig", max_len: int = 150) -> str:
    pieces = split_into_chunks(text, max_len=max_len)
    if not pieces:
        return text
    out: List[str] = []
    for piece in pieces:
        out.append(paraphrase_text(piece, config) or piece)
    return " ".join(out)


def parse_date_components(value: str) -> tuple[int | None, int | None, int | None]:
    if not value:
        return None, None, None
    candidate = value.strip()
    if not candidate:
        return None, None, None
    normalized = candidate.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        return dt.year, dt.month, dt.day
    except ValueError:
        pass
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.year, dt.month, dt.day
        except ValueError:
            continue
    match = YEAR_PATTERN.search(candidate)
    if match:
        year = int(match.group())
        if 1900 <= year <= 2100:
            return year, None, None
    return None, None, None


def parse_year_from_string(value: str) -> int | None:
    year, _month, _day = parse_date_components(value)
    return year


def extract_dates_from_jsonld(data) -> list[tuple[int | None, int | None, int | None]]:
    results: list[tuple[int | None, int | None, int | None]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key in JSONLD_DATE_KEYS and isinstance(value, str):
                results.append(parse_date_components(value))
            elif isinstance(value, (list, dict)):
                results.extend(extract_dates_from_jsonld(value))
    elif isinstance(data, list):
        for item in data:
            results.extend(extract_dates_from_jsonld(item))
    return results


def extract_publication_date(soup: BeautifulSoup, config: EngineConfig) -> tuple[int | None, int | None, int | None, str | None]:
    selectors = [s.strip() for s in (config.date_selector or "").splitlines() if s.strip()]
    candidates: list[tuple[str, str]] = []

    def add_candidate(value: str | None, source: str) -> None:
        if value:
            candidates.append((value, source))

    for selector in selectors:
        for element in soup.select(selector):
            if config.date_attribute:
                add_candidate(element.get(config.date_attribute), f"selector:{selector}[{config.date_attribute}]")
            add_candidate(element.get_text(" ", strip=True), f"selector:{selector}")

    for meta in soup.find_all("meta"):
        name = (meta.get("property") or meta.get("name") or meta.get("itemprop") or meta.get("http-equiv") or "").lower()
        if name in DATE_META_KEYS:
            add_candidate(meta.get("content"), f"meta:{name}")

    for time_tag in soup.find_all("time"):
        add_candidate(time_tag.get("datetime"), "time:datetime")
        add_candidate(time_tag.get_text(" ", strip=True), "time:text")

    for script in soup.find_all("script"):
        type_attr = script.get("type")
        if not type_attr or "ld+json" not in type_attr.lower():
            continue
        try:
            data = json.loads(script.string or script.get_text() or "{}")
        except Exception:
            continue
        for year, month, day in extract_dates_from_jsonld(data):
            if year:
                return year, month, day, "json-ld"

    for value, source in candidates:
        year, month, day = parse_date_components(value)
        if year:
            return year, month, day, source

    fallback_candidates = soup.select("[class*='date'], [class*='time'], [id*='date'], [id*='time']")
    for element in fallback_candidates:
        year, month, day = parse_date_components(element.get_text(" ", strip=True))
        if year:
            return year, month, day, "fallback"

    return None, None, None, None


def should_paraphrase_article(html: str, config: EngineConfig) -> tuple[bool, int | None, str | None, int | None, int | None]:
    soup = BeautifulSoup(html, "html.parser")
    year, month, day, source = extract_publication_date(soup, config)

    threshold_year = config.paraphrase_year_threshold or 2025
    threshold_month = max(1, min(12, config.paraphrase_month_threshold or 1))
    threshold_day = max(1, min(31, config.paraphrase_day_threshold or 1))
    threshold_tuple = (threshold_year, threshold_month, threshold_day)

    if year is None:
        return False, None, source, None, None

    if month is None:
        month = threshold_month
    if day is None:
        day = threshold_day

    current_tuple = (year, month, day)
    should = current_tuple >= threshold_tuple
    return should, year, source, month, day



@dataclass
class CachedPage:
    html: str
    compressed: bytes


def get_page_cache_path(path: str, query: str) -> Path:
    key_source = f"{path}?{query}" if query else path
    digest = hashlib.md5(key_source.encode("utf-8")).hexdigest()
    return Path(PAGE_CACHE_DIR) / f"{digest}.html.gz"


def load_cached_page(cache_path: Path) -> CachedPage | None:
    try:
        compressed = cache_path.read_bytes()
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.exception("Failed to read cached page %s: %s", cache_path, exc)
        return None

    try:
        html = gzip.decompress(compressed).decode("utf-8")
        return CachedPage(html=html, compressed=compressed)
    except (OSError, gzip.BadGzipFile):
        try:
            html = compressed.decode("utf-8")
        except Exception as decode_exc:
            logger.exception("Failed to decode cached page %s: %s", cache_path, decode_exc)
            return None
        logger.warning("Cached page %s was not gzip-compressed; rewriting cache entry", cache_path)
        try:
            cache_page(cache_path, html)
            compressed = cache_path.read_bytes()
        except Exception as write_exc:
            logger.exception("Failed to rewrite cached page %s as gzip: %s", cache_path, write_exc)
            compressed = gzip.compress(html.encode("utf-8"))
        return CachedPage(html=html, compressed=compressed)
    except Exception as exc:
        logger.exception("Failed to load cached page %s: %s", cache_path, exc)
        return None


def cache_page(cache_path: Path, content: str) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.parent / f"{cache_path.name}.tmp"
        compressed = gzip.compress(content.encode("utf-8"))
        temp_path.write_bytes(compressed)
        temp_path.replace(cache_path)
        logger.info("Cached page at %s", cache_path)
    except Exception as exc:
        logger.exception("Failed to write cached page %s: %s", cache_path, exc)




def purge_downstream_cache(full_url: str, engine_config: EngineConfig) -> None:
    zone_id = (engine_config.cf_zone_id or "").strip()
    api_token = (engine_config.cf_api_token or "").strip()
    if not (zone_id and api_token):
        logger.debug("Skipping CDN purge for %s; Cloudflare credentials not configured", full_url)
        return
    try:
        response = requests.post(
            f"https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            json={"files": [full_url]},
            timeout=15,
        )
        if response.status_code // 100 != 2:
            logger.warning("Cloudflare purge failed for %s: %s %s", full_url, response.status_code, response.text)
        else:
            logger.info("Purged CDN cache for %s", full_url)
    except Exception as exc:
        logger.exception("Cloudflare purge raised for %s: %s", full_url, exc)


ANCHOR_LEADING_SPACE_RE = re.compile(r"(\w)(<a\b)", re.IGNORECASE)
ANCHOR_TRAILING_SPACE_RE = re.compile(r"(</a>)(\w)", re.IGNORECASE)


def normalize_anchor_spacing(html: str) -> str:
    html = ANCHOR_LEADING_SPACE_RE.sub(r"\1 \2", html)
    html = ANCHOR_TRAILING_SPACE_RE.sub(r"\1 \2", html)
    return html


def render_page(
    upstream_html: str,
    engine_config: EngineConfig,
    replacements: Dict[str, str],
    original_domain: str,
    cloned_domain: str,
    include_paraphrase: bool,
) -> str:
    soup = BeautifulSoup(upstream_html, "html.parser")

    if include_paraphrase:
        for selector in ARTICLE_SELECTORS:
            elements = soup.select(selector)
            if not elements:
                continue
            logger.info("Found %d elements for selector '%s'", len(elements), selector)
            for element in elements:
                paraphrase_element(element, engine_config)

    ensure_lazy_social_scripts(soup)

    rendered = str(soup)
    for find, replace in replacements.items():
        rendered = rendered.replace(find, replace)
    rendered = rendered.replace(original_domain, cloned_domain)
    rendered = normalize_anchor_spacing(rendered)
    return rendered


def process_background_paraphrase(
    target_url: str,
    path: str,
    query_string: str,
    upstream_html: str,
    engine_config: EngineConfig,
    replacements: Dict[str, str],
    original_domain: str,
    cloned_domain: str,
    cache_path_str: str,
    should_paraphrase: bool,
) -> None:
    cache_path = Path(cache_path_str) if cache_path_str else None
    if not should_paraphrase:
        logger.info("Background paraphrase skipped for %s?%s due to date threshold_year", path, query_string)
        return
    if cache_path is None:
        logger.warning("No cache path resolved for background paraphrase %s?%s", path, query_string)
        return
    try:
        rendered = render_page(
            upstream_html,
            engine_config,
            replacements,
            original_domain,
            cloned_domain,
            include_paraphrase=True,
        )
        cache_page(cache_path, rendered)
        cloned_path = f"{cloned_domain.rstrip('/')}/{path.lstrip('/')}"
        if query_string:
            cloned_path = f"{cloned_path}?{query_string}"
        purge_downstream_cache(cloned_path, engine_config)
    except Exception as exc:
        logger.exception(
            "Background paraphrasing failed for %s?%s (target %s): %s",
            path,
            query_string,
            target_url,
            exc,
        )




# ------------------------------------------------------------------------------
# Admin UI routes
# ------------------------------------------------------------------------------

@app.get("/admin/", response_class=HTMLResponse)
async def admin_dashboard(request: Request, user: str = Depends(verify_user)) -> HTMLResponse:
    conn = get_db()
    c = conn.cursor()

    site = c.execute("SELECT original_domain, cloned_domain FROM sites WHERE id = 1").fetchone()
    config_rows = c.execute("SELECT key, value FROM configs WHERE site_id = 1").fetchall()
    conn.close()

    config_map = {row["key"]: row["value"] for row in config_rows}

    replacements: List[Dict[str, str]] = []
    for rep in config_rows:
        if rep["key"].startswith("replace_"):
            replacements.append(
                {
                    "id": rep["key"],
                    "find": rep["key"].split("_", 1)[1],
                    "replace": rep["value"],
                }
            )

    context = {
        "request": request,
        "original": site["original_domain"] if site else "https://example.com",
        "cloned": site["cloned_domain"] if site else "https://yourdomain.com",
        "config": config_map,
        "replacements": replacements,
        "hf_available": TRANSFORMERS_AVAILABLE,
        "hf_import_error": HF_IMPORT_ERROR,
        "remote_default": REMOTE_PARAPHRASE_DEFAULT,
    }
    return templates.TemplateResponse("dashboard.html", context)


@app.post("/admin/config/")
async def update_config(
    original: str = Form(...),
    cloned: str = Form(...),
    user: str = Depends(verify_user),
) -> RedirectResponse:
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE sites SET original_domain = ?, cloned_domain = ? WHERE id = 1", (original, cloned))
    conn.commit()
    conn.close()
    return RedirectResponse("/admin/?tab=config", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/api/")
async def update_api_config(
    hf_remote_url: str = Form(""),
    hf_remote_timeout: str = Form("45"),
    use_background_paraphrase: str | None = Form(None),
    use_huggingface_local: str | None = Form(None),
    use_huggingface: str | None = Form(None),
    hf_model_id: str = Form("humarin/chatgpt_paraphraser_on_T5_base"),
    hf_task: str = Form("text2text-generation"),
    hf_device: str = Form("cpu"),
    hf_max_new_tokens: str = Form("256"),
    hf_chunk_size: str = Form("1200"),
    hf_num_beams: str = Form("4"),
    hf_temperature: str = Form("0.7"),
    hf_do_sample: str = Form("false"),
    gemini_key: str = Form(""),
    use_gemini: str = Form("false"),
    xai_key: str = Form(""),
    use_xai: str = Form("false"),
    xai_model: str = Form("grok-code-fast-1"),
    cf_zone_id: str = Form(""),
    cf_api_token: str = Form(""),
    date_selector: str = Form(""),
    date_attribute: str = Form(""),
    paraphrase_year_threshold: str = Form("2025"),
    paraphrase_month_threshold: str = Form("8"),
    paraphrase_day_threshold: str = Form("1"),
    user: str = Depends(verify_user),
) -> RedirectResponse:
    conn = get_db()
    c = conn.cursor()

    remote_url = hf_remote_url.strip()
    remote_timeout = hf_remote_timeout.strip() or "45"

    save_config(c, "hf_remote_url", remote_url)
    save_config(c, "hf_remote_timeout", remote_timeout)

    background_enabled = (use_background_paraphrase or "").strip().lower() in {"on", "true", "1", "yes"}
    save_config(c, "use_background_paraphrase", "true" if background_enabled else "false")
    if not background_enabled:
        clear_directory(PAGE_CACHE_DIR)

    hf_local_form_value = use_huggingface_local if use_huggingface_local is not None else use_huggingface
    save_config(
        c,
        "use_huggingface_local",
        "true" if (hf_local_form_value or "").lower() in {"on", "true", "1", "yes"} else "false",
    )
    save_config(c, "hf_model_id", hf_model_id.strip())
    save_config(c, "hf_task", hf_task.strip() or "text2text-generation")
    save_config(c, "hf_device", hf_device.strip() or "cpu")
    save_config(c, "hf_max_new_tokens", hf_max_new_tokens.strip() or "256")
    save_config(c, "hf_chunk_size", hf_chunk_size.strip() or "1200")
    save_config(c, "hf_num_beams", hf_num_beams.strip() or "4")
    save_config(c, "hf_temperature", hf_temperature.strip() or "0.7")
    save_config(c, "hf_do_sample", "true" if hf_do_sample in {"on", "true"} else "false")

    save_config(c, "gemini_key", gemini_key.strip())
    save_config(c, "use_gemini", "true" if use_gemini in {"on", "true"} else "false")

    save_config(c, "xai_key", xai_key.strip())
    save_config(c, "use_xai", "true" if use_xai in {"on", "true"} else "false")
    save_config(c, "xai_model", xai_model.strip() or "grok-code-fast-1")

    save_config(c, "cf_zone_id", cf_zone_id.strip())
    save_config(c, "cf_api_token", cf_api_token.strip())

    save_config(c, "date_selector", date_selector.strip())
    save_config(c, "date_attribute", date_attribute.strip())
    save_config(c, "paraphrase_year_threshold", paraphrase_year_threshold.strip() or "2025")
    save_config(c, "paraphrase_month_threshold", paraphrase_month_threshold.strip() or "8")
    save_config(c, "paraphrase_day_threshold", paraphrase_day_threshold.strip() or "1")

    conn.commit()
    conn.close()

    HF_PIPELINES.clear()

    return RedirectResponse("/admin/?tab=api", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/replacements/")
async def update_replacement(
    find: str = Form(...),
    replace: str = Form(...),
    id: str = Form(""),
    user: str = Depends(verify_user),
) -> RedirectResponse:
    conn = get_db()
    c = conn.cursor()

    if id:
        save_config(c, id, replace)
    else:
        save_config(c, f"replace_{find}", replace)

    conn.commit()
    conn.close()
    return RedirectResponse("/admin/?tab=replacement", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/delete-replacement/")
async def delete_replacement(id: str = Form(...), user: str = Depends(verify_user)) -> RedirectResponse:
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM configs WHERE site_id = 1 AND key = ?", (id,))
    conn.commit()
    conn.close()
    return RedirectResponse("/admin/?tab=replacement", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/clear-cache/")
async def clear_cache(user: str = Depends(verify_user)) -> RedirectResponse:
    for directory in (CACHE_DIR, PAGE_CACHE_DIR):
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    return RedirectResponse("/admin/?tab=api", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/purge/")
async def purge_cache_url(
    purge_url: str = Form(...),
    user: str = Depends(verify_user),
) -> RedirectResponse:
    target = (purge_url or "").strip()
    if not target:
        return RedirectResponse("/admin/?tab=api&purge_error=1", status_code=status.HTTP_303_SEE_OTHER)

    conn = get_db()
    c = conn.cursor()
    site = c.execute("SELECT original_domain, cloned_domain FROM sites WHERE id = 1").fetchone()
    config_rows = c.execute("SELECT key, value FROM configs WHERE site_id = 1").fetchall()
    conn.close()

    config_map = {row["key"]: row["value"] for row in config_rows}
    engine_config = build_engine_config(config_map)

    cloned_domain = site["cloned_domain"] if site else "https://yourdomain.com"
    base_url = (cloned_domain or "https://yourdomain.com").rstrip("/")

    if target.startswith("http://") or target.startswith("https://"):
        full_url = target
    else:
        full_url = f"{base_url}/{target.lstrip('/')}"

    parsed = urlsplit(full_url)
    relative_path = parsed.path.lstrip("/")
    query = parsed.query

    if engine_config.use_background_paraphrase:
        cache_path = get_page_cache_path(relative_path, query)
        try:
            cache_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Unable to delete cached page %s: %s", cache_path, exc)
        temp_path = cache_path.parent / f"{cache_path.name}.tmp"
        if temp_path.exists():
            try:
                logger.info("Deleting temp cache %s", temp_path)
                temp_path.unlink()
            except Exception as exc:
                logger.warning("Unable to delete temp cache %s: %s", temp_path, exc)

    purge_downstream_cache(full_url, engine_config)

    label = parsed.path or "/"
    if parsed.query:
        label = f"{label}?{parsed.query}"
    redirect_url = f"/admin/?tab=api&purged={quote_plus(label)}"
    return RedirectResponse(redirect_url, status_code=status.HTTP_303_SEE_OTHER)


# ------------------------------------------------------------------------------
# Proxy endpoint
# ------------------------------------------------------------------------------

@app.get("/{path:path}")
async def proxy_site(path: str, request: Request, background_tasks: BackgroundTasks) -> Response:
    try:
        conn = get_db()
        c = conn.cursor()

        site = c.execute("SELECT original_domain, cloned_domain FROM sites WHERE id = 1").fetchone()
        original_domain = site["original_domain"] if site else "https://example.com"
        cloned_domain = site["cloned_domain"] if site else "https://yourdomain.com"

        config_rows = c.execute("SELECT key, value FROM configs WHERE site_id = 1").fetchall()
        config_map = {row["key"]: row["value"] for row in config_rows}

        replacements = {
            row["key"].split("_", 1)[1]: row["value"]
            for row in config_rows
            if row["key"].startswith("replace_")
        }

        conn.close()

        engine_config = build_engine_config(config_map)

        query_string = request.url.query or ""
        cache_path: Path | None = None
        cached_page: CachedPage | None = None
        if engine_config.use_background_paraphrase:
            cache_path = get_page_cache_path(path, query_string)
            logger.info("Generated cache path %s%s", path, f"?{query_string}" if query_string else "")
            cached_page = load_cached_page(cache_path)
            if cached_page is not None:
                logger.info("Serving cached page for %s?%s", path, query_string)
                accept_encoding = request.headers.get("accept-encoding", "")
                supports_gzip = "gzip" in accept_encoding.lower()
                headers = {"Vary": "Accept-Encoding"}
                if supports_gzip:
                    headers["Content-Encoding"] = "gzip"
                    return Response(
                        content=cached_page.compressed,
                        status_code=200,
                        media_type="text/html; charset=utf-8",
                        headers=headers,
                    )
                return Response(
                    content=cached_page.html,
                    status_code=200,
                    media_type="text/html; charset=utf-8",
                    headers=headers,
                )

        target_url = f"{original_domain.rstrip('/')}/{path.lstrip('/')}"
        if query_string:
            target_url = f"{target_url}?{query_string}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0 Safari/537.36"
            )
        }
        upstream_response = requests.get(target_url, headers=headers, timeout=15)
        content_type = upstream_response.headers.get("Content-Type", "").lower()

        should_paraphrase = True
        publication_year: int | None = None
        publication_source: str | None = None

        if "text/html" in content_type:
            should_paraphrase, publication_year, publication_source, publication_month, publication_day = should_paraphrase_article(upstream_response.text, engine_config)
            threshold_year = engine_config.paraphrase_year_threshold or 2025
            threshold_month = max(1, min(12, engine_config.paraphrase_month_threshold or 1))
            threshold_day = max(1, min(31, engine_config.paraphrase_day_threshold or 1))
            if should_paraphrase:
                logger.info("Paraphrasing enabled for %s?%s (date=%04d-%02d-%02d >= threshold=%04d-%02d-%02d)", path, query_string, publication_year, publication_month, publication_day, threshold_year, threshold_month, threshold_day)
            else:
                if publication_year is None:
                    logger.info("Skipping paraphrase for %s?%s: publication date not found (threshold=%04d-%02d-%02d)", path, query_string, threshold_year, threshold_month, threshold_day)
                else:
                    logger.info("Skipping paraphrase for %s?%s: publication date %04d-%02d-%02d below threshold %04d-%02d-%02d", path, query_string, publication_year, publication_month or threshold_month, publication_day or threshold_day, threshold_year, threshold_month, threshold_day)
                if publication_source:
                    logger.debug("Publication date source: %s", publication_source)
                if cache_path and cache_path.exists():
                    try:
                        logger.info("Removing cached page %s because paraphrasing is disabled for this article", cache_path)
                        cache_path.unlink()
                    except Exception as exc:
                        logger.warning("Unable to delete cached page %s: %s", cache_path, exc)
                    temp_path = cache_path.parent / f"{cache_path.name}.tmp"
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception as exc:
                            logger.warning("Unable to delete temp cache %s: %s", temp_path, exc)
            rendered = render_page(
                upstream_response.text,
                engine_config,
                replacements,
                original_domain,
                cloned_domain,
                include_paraphrase=should_paraphrase and not engine_config.use_background_paraphrase,
            )
            if engine_config.use_background_paraphrase and should_paraphrase and upstream_response.status_code == 200:
                if cache_path is None:
                    cache_path = get_page_cache_path(path, query_string)
                background_tasks.add_task(
                    process_background_paraphrase,
                    target_url,
                    path,
                    query_string,
                    upstream_response.text,
                    engine_config,
                    replacements,
                    original_domain,
                    cloned_domain,
                    str(cache_path),
                    should_paraphrase,
                )
            return HTMLResponse(content=rendered, status_code=upstream_response.status_code)

        response_headers = dict(upstream_response.headers)
        response_headers.pop("Content-Encoding", None)
        response_headers.pop("Transfer-Encoding", None)
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    except Exception as exc:
        logger.exception("Error in proxy_site: %s", exc)
        return HTMLResponse(content=f"Error fetching content: {exc}", status_code=500)


# expose ASGI app
application = app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
