"""MOPS-specific pieces shared by the MOPS adapters.

Generic behaviour lives elsewhere: HTTP limits and streaming in
``bounded_http``, and the Taiwan clock, board mapping, ticker parsing and label
normalization in ``taiwan_common``. What remains here is MOPS's own vocabulary:
its error type, the market names MOPS reports per board, and transport
wrappers that raise that error with the ``MOPS`` message prefix.
"""

from __future__ import annotations

from .bounded_http import DEFAULT_TIMEOUT_SECONDS, bounded_request, decode_json_object
from .errors import VendorError

_SOURCE = "MOPS"

# The market name MOPS reports for each board code.
BOARD_MARKET_NAME = {"sii": "上市公司", "otc": "上櫃公司"}


class MopsUnavailableError(VendorError):
    """MOPS could not be read (HTTP failure, oversized body, or layout change)."""


def http_get(url: str, *, accept: str = "text/html", timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    """GET a MOPS page and return the raw body, or raise :class:`MopsUnavailableError`."""
    return bounded_request(
        "GET", url, error_cls=MopsUnavailableError, source=_SOURCE, accept=accept, timeout=timeout,
    ).body


def http_post_json(url: str, body: dict, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """POST JSON to a MOPS API and return the decoded object, or raise
    :class:`MopsUnavailableError` (transport, status, size, malformed JSON, or a
    top-level value that is not an object)."""
    resp = bounded_request(
        "POST", url, error_cls=MopsUnavailableError, source=_SOURCE,
        accept="application/json", json_body=body, timeout=timeout,
    )
    return decode_json_object(resp, error_cls=MopsUnavailableError, source=_SOURCE)
