"""Bounded HTTP for keyless official data sources.

One transport with the safety rules every scraper-style adapter needs, and no
knowledge of any particular source:

* ``requests`` with TLS verification left at its default (on). Some official
  Taiwan hosts ship certificates without a Subject Key Identifier, which
  urllib's strict X.509 mode on Python 3.13+ rejects; ``requests`` accepts them
  while still verifying the chain.
* A timeout and a fixed User-Agent on every request.
* The body is streamed and abandoned as soon as it passes ``max_bytes`` (a
  declared ``Content-Length`` over the cap is refused before reading).
* Any transport error or non-200 status raises.

The caller supplies the exception class and a ``source`` label, so each data
source keeps its own error type and message prefix (``"<source> fetch failed
..."``) without depending on another source's names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests
from requests.structures import CaseInsensitiveDict

DEFAULT_USER_AGENT = "tradingagents/0.4 (+https://github.com/TauricResearch/TradingAgents)"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_BODY_BYTES = 4 * 1024 * 1024
CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class BoundedResponse:
    """A fully read, size-checked 200 response."""

    url: str
    status: int
    headers: CaseInsensitiveDict
    body: bytes


def _fetch_failed(source: str, url: str, exc: Exception, error_cls: type[Exception]) -> Exception:
    return error_cls(f"{source} fetch failed ({type(exc).__name__}) for {url}")


def bounded_request(
    method: str,
    url: str,
    *,
    error_cls: type[Exception],
    source: str,
    accept: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    json_body: Any = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> BoundedResponse:
    """Send a GET or POST and return the whole body, or raise ``error_cls``.

    ``params`` become the query string, ``data`` a form body, ``json_body`` a
    JSON body; only the ones given are sent.
    """
    method = method.upper()
    if method not in ("GET", "POST"):
        raise ValueError(f"unsupported method {method!r}")
    kwargs: dict[str, Any] = {
        "headers": {"User-Agent": user_agent, "Accept": accept},
        "timeout": timeout,
        "stream": True,
    }
    if params is not None:
        kwargs["params"] = params
    if data is not None:
        kwargs["data"] = data
    if json_body is not None:
        kwargs["json"] = json_body

    send = requests.get if method == "GET" else requests.post
    try:
        resp = send(url, **kwargs)
    except requests.RequestException as exc:  # timeout, DNS, TLS, connection reset
        raise _fetch_failed(source, url, exc, error_cls) from exc

    with resp:
        if resp.status_code != 200:
            raise error_cls(f"{source} fetch failed (HTTP {resp.status_code}) for {url}")
        too_large = error_cls(
            f"{source} response exceeded {max_bytes} bytes for {url}; refusing to parse"
        )
        declared = resp.headers.get("Content-Length", "")
        if declared.isdigit() and int(declared) > max_bytes:
            raise too_large
        body = bytearray()
        try:
            for chunk in resp.iter_content(CHUNK_BYTES):
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise too_large
        except requests.RequestException as exc:  # connection dropped mid-body
            raise _fetch_failed(source, url, exc, error_cls) from exc
        headers = CaseInsensitiveDict(resp.headers)
    return BoundedResponse(url=url, status=resp.status_code, headers=headers, body=bytes(body))


def decode_json_object(
    resp: BoundedResponse,
    *,
    error_cls: type[Exception],
    source: str,
    encoding: str = "utf-8",
) -> dict:
    """Decode ``resp.body`` as a JSON object, or raise ``error_cls``."""
    try:
        decoded = json.loads(resp.body.decode(encoding))
    except (UnicodeDecodeError, ValueError) as exc:
        raise error_cls(f"{source} returned malformed JSON for {resp.url}") from exc
    if not isinstance(decoded, dict):
        raise error_cls(f"{source} returned an unexpected JSON shape for {resp.url}")
    return decoded
