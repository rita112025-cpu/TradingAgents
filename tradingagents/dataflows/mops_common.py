"""Shared transport and Taiwan helpers for the MOPS adapters.

Both MOPS adapters (monthly revenue on the legacy ``mopsov`` host, material
announcements on the new JSON API) need the same safety rules and Taiwan
facts. Keeping them here means a limit, error type, board mapping, or clock
is defined once:

* HTTP: ``requests`` with TLS verification left on (the MOPS hosts' certificates
  lack a Subject Key Identifier, which urllib's strict X.509 mode on Python
  3.13+ rejects), a timeout, a fixed User-Agent, a streamed body abandoned at
  the size cap, and any non-200 status or transport error raised as
  :class:`MopsUnavailableError`.
* Taiwan ticker parsing through the market profile (``resolve_market``), never
  a second suffix check.
* One "now": Taiwan time, which decides live versus historical for every
  MOPS adapter.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone

import requests

from .errors import NoMarketDataError, VendorError
from .market_profiles import MARKET_TAIWAN, resolve_market

USER_AGENT = "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"
TIMEOUT_SECONDS = 20.0
MAX_BODY_BYTES = 4 * 1024 * 1024   # largest live MOPS response seen is ~0.5 MB
CHUNK_BYTES = 64 * 1024

# Taiwan has used UTC+8 without daylight saving since 1979; a fixed offset
# avoids depending on a tz database (absent on Windows without tzdata).
TAIPEI = timezone(timedelta(hours=8), "Asia/Taipei")

# Yahoo suffix -> MOPS board code. Taiwan detection itself is the market
# profile's job; this only maps a known-Taiwan suffix to the MOPS board.
BOARD_BY_SUFFIX = {".TW": "sii", ".TWO": "otc"}
# The market name MOPS reports for each board.
BOARD_MARKET_NAME = {"sii": "上市公司", "otc": "上櫃公司"}


class MopsUnavailableError(VendorError):
    """MOPS could not be read (HTTP failure, oversized body, or layout change)."""


# --------------------------------------------------------------------------
# Clock
# --------------------------------------------------------------------------

def taipei_now() -> datetime:
    """Current Taiwan time, the single "now" for every MOPS adapter.

    Adapters bind it as their own ``_now`` so tests patch one seam per module.
    """
    return datetime.now(TAIPEI)


# --------------------------------------------------------------------------
# Labels and tickers
# --------------------------------------------------------------------------

def norm_header(label: str) -> str:
    """Minimal header normalization: NFKC (full-width -> half-width, NBSP ->
    space) and removal of all whitespace. MOPS labels are CJK, where whitespace
    only comes from formatting. No fuzzy matching: labels compare exactly.
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", label))


def split_taiwan_ticker(ticker: str, source: str) -> tuple[str, str]:
    """``2330.TW`` -> (``"2330"``, ``"sii"``); ``6488.TWO`` -> (``"6488"``, ``"otc"``).

    Raises :class:`NoMarketDataError` naming ``source`` for anything that is
    not a Taiwan-listed ticker with a numeric company code, so callers can
    refuse before making any request.
    """
    symbol = (ticker or "").strip().upper()
    if resolve_market(symbol) != MARKET_TAIWAN:
        raise NoMarketDataError(
            ticker, detail=f"{source} is only available for Taiwan-listed tickers "
                           "(.TW / .TWO); not queried",
        )
    for suffix, board in BOARD_BY_SUFFIX.items():
        if symbol.endswith(suffix):
            code = symbol[: -len(suffix)]
            if re.fullmatch(r"\d{4,6}", code):
                return code, board
            raise NoMarketDataError(
                ticker, detail=f"{code!r} is not a numeric MOPS company code; not queried",
            )
    raise NoMarketDataError(
        ticker, detail="Taiwan ticker without a MOPS board suffix (.TW/.TWO); not queried",
    )


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _fetch_failed(url: str, exc: Exception) -> MopsUnavailableError:
    return MopsUnavailableError(f"MOPS fetch failed ({type(exc).__name__}) for {url}")


def _read_capped(resp, url: str) -> bytes:
    """Return the body of a streamed response, enforcing status and size cap."""
    with resp:
        if resp.status_code != 200:
            raise MopsUnavailableError(f"MOPS fetch failed (HTTP {resp.status_code}) for {url}")
        too_large = MopsUnavailableError(
            f"MOPS response exceeded {MAX_BODY_BYTES} bytes for {url}; refusing to parse"
        )
        declared = resp.headers.get("Content-Length", "")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            raise too_large
        body = bytearray()
        try:
            for chunk in resp.iter_content(CHUNK_BYTES):
                body.extend(chunk)
                if len(body) > MAX_BODY_BYTES:
                    raise too_large
        except requests.RequestException as exc:  # connection dropped mid-body
            raise _fetch_failed(url, exc) from exc
    return bytes(body)


def http_get(url: str, *, accept: str = "text/html", timeout: float = TIMEOUT_SECONDS) -> bytes:
    """GET ``url`` and return the raw body, or raise :class:`MopsUnavailableError`."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": accept},
            timeout=timeout,
            stream=True,
        )
    except requests.RequestException as exc:  # timeout, DNS, TLS, connection reset
        raise _fetch_failed(url, exc) from exc
    return _read_capped(resp, url)


def http_post_json(url: str, body: dict, *, timeout: float = TIMEOUT_SECONDS) -> dict:
    """POST ``body`` as JSON and return the decoded JSON object.

    Raises :class:`MopsUnavailableError` on transport errors, non-200 status,
    oversized bodies, malformed JSON, or a top-level value that is not an object.
    """
    try:
        resp = requests.post(
            url,
            json=body,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            stream=True,
        )
    except requests.RequestException as exc:  # timeout, DNS, TLS, connection reset
        raise _fetch_failed(url, exc) from exc
    raw = _read_capped(resp, url)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise MopsUnavailableError(f"MOPS returned malformed JSON for {url}") from exc
    if not isinstance(decoded, dict):
        raise MopsUnavailableError(f"MOPS returned an unexpected JSON shape for {url}")
    return decoded
