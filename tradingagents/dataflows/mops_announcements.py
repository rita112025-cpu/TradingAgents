"""MOPS material announcements (重大訊息) for Taiwan-listed companies.

Reads the new MOPS site's JSON API, which the MOPS single-page app itself
calls, so the News Analyst can see official material announcements for
``.TW`` (TWSE, ``sii``) and ``.TWO`` (TPEx, ``otc``) tickers.

Source facts that shape the design (verified against the live API):

* List: ``POST https://mops.twse.com.tw/mops/api/t05st01`` with JSON
  ``{"companyId", "year" (ROC), "month", "firstDay", "lastDay"}``. The UI marks
  the day bounds optional, but every request without them is rejected with
  ``傳入參數異常``, so the adapter always sends them.
* Detail: ``POST .../t05st01_detail`` with the ``parameters`` object the list
  row carries (``companyId``, ``marketKind``, ``enterDate``, ``serialNumber``).
* Envelope ``{"code", "message", "result"}``: ``200`` success, ``406``
  ``查無相符資料`` (no matching records), anything else (``500``
  ``傳入參數異常`` / ``公司代號格式錯誤``) is a vendor failure.
* ``result.titles`` labels every column, so fields are located by label.
* Each row carries its own publication timestamp (``發言日期`` ``發言時間``,
  Taiwan time). ``enterDate`` in the detail reference can differ from it and
  is never used for filtering.
* The host certificate lacks a Subject Key Identifier (rejected by urllib's
  strict mode on Python 3.13+); the shared transport in ``mops_common`` uses
  ``requests`` with verification on.

Point-in-time: a historical run knows only ``curr_date``, not the decision
time, and MOPS publishes material announcements after the market close. A
historical run therefore uses announcements dated strictly before
``curr_date`` and never even requests ``curr_date`` itself; a live run uses
what was published up to the actual run time.

States kept apart in the output: list unavailable (raises), detail
unavailable (per announcement), no announcements (queried, none found), and
same-day withheld (historical runs).
"""

from __future__ import annotations

import calendar
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .mops_common import (
    BOARD_MARKET_NAME,
    TAIPEI,
    TIMEOUT_SECONDS,
    MopsUnavailableError,
    http_post_json,
    norm_header,
    split_taiwan_ticker,
    taipei_now as _now,  # test seam; the shared Taiwan-time clock
)

logger = logging.getLogger(__name__)

_API_BASE = "https://mops.twse.com.tw/mops/api/"
_LIST_API = "t05st01"
_DETAIL_API = "t05st01_detail"

_MAX_LOOK_BACK_DAYS = 90       # bounds the number of monthly list requests
_MAX_DETAIL_LIMIT = 5          # bounds detail requests and prompt size
_DETAIL_TEXT_LIMIT = 2000      # characters of 說明 kept per announcement
_REQUEST_SPACING_SECONDS = 0.5 # politeness gap between sequential requests

_CODE_OK = "200"
_CODE_NO_MATCH = "406"

_LIST_REQUIRED = {
    "code": "公司代號",
    "name": "公司名稱",
    "date": "發言日期",
    "time": "發言時間",
    "subject": "主旨",
}
_LIST_OPTIONAL = {"detail_ref": "詳細資料"}

_DETAIL_REQUIRED = {
    "date": "發言日期",
    "time": "發言時間",
    "subject": "主旨",
    "clause": "符合條款",
    "fact_date": "事實發生日",
    "text": "說明",
}

_ROC_DATE_RE = re.compile(r"(\d{2,3})/(\d{1,2})/(\d{1,2})")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")

WITHHELD_SAME_DAY_NOTICE = (
    "same-day announcements withheld because historical decision time is unknown"
)


class MopsBoardMismatchError(MopsUnavailableError):
    """MOPS reports the company on a different board than the ticker suffix.

    Never corrected or accepted: the ticker may point at a different listing,
    so the whole result is refused.
    """


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _post_json(api: str, body: dict, timeout: float = TIMEOUT_SECONDS) -> dict:
    """POST to a MOPS JSON API through the shared transport (test seam)."""
    return http_post_json(_API_BASE + api, body, timeout=timeout)


class _Client:
    """Sequential MOPS API caller with a politeness gap between requests."""

    def __init__(self):
        self._calls = 0

    def call(self, api: str, body: dict) -> dict | None:
        """Return ``result`` on success, None for "no matching records", or raise."""
        if self._calls:
            _sleep(_REQUEST_SPACING_SECONDS)
        self._calls += 1
        envelope = _post_json(api, body)
        code = str(envelope.get("code", "")).strip()
        message = str(envelope.get("message", "")).strip()
        if code == _CODE_NO_MATCH:
            return None
        if code != _CODE_OK:
            raise MopsUnavailableError(
                f"MOPS {api} returned code {code or '<missing>'}: {message or '<no message>'}"
            )
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise MopsUnavailableError(f"MOPS {api} returned code 200 without a result object")
        return result


# --------------------------------------------------------------------------
# Response validation
# --------------------------------------------------------------------------

def _title_map(api: str, titles, required: dict[str, str], optional: dict[str, str]) -> dict[str, int]:
    """Map field keys to column indexes by exact normalized title label."""
    if not isinstance(titles, list) or not titles:
        raise MopsUnavailableError(f"MOPS {api} layout mismatch: missing titles")
    positions: dict[str, list[int]] = {}
    for index, title in enumerate(titles):
        if not isinstance(title, dict) or not isinstance(title.get("main"), str):
            raise MopsUnavailableError(f"MOPS {api} layout mismatch: malformed title #{index}")
        if title.get("sub"):
            # Sub-columns would change the row width in a way we cannot map.
            raise MopsUnavailableError(
                f"MOPS {api} layout mismatch: title {title['main']!r} has sub-columns"
            )
        positions.setdefault(norm_header(title["main"]), []).append(index)

    mapping: dict[str, int] = {}
    for key, label in (required | optional).items():
        found = positions.get(label, [])
        if len(found) > 1:
            raise MopsUnavailableError(
                f"MOPS {api} layout mismatch: title {label!r} appears {len(found)} times"
            )
        if not found:
            if key in required:
                raise MopsUnavailableError(
                    f"MOPS {api} layout mismatch: required title {label!r} not found"
                )
            continue
        mapping[key] = found[0]
    return mapping


def _check_company(api: str, result: dict, code: str, board: str) -> str:
    """Validate companyId and board; return MOPS's market name."""
    company = str(result.get("companyId", "")).strip()
    if company != code:
        raise MopsUnavailableError(
            f"MOPS {api} layout mismatch: companyId {company!r} does not match {code!r}"
        )
    market = str(result.get("marketName", "")).strip()
    expected = BOARD_MARKET_NAME[board]
    if market != expected:
        raise MopsBoardMismatchError(
            f"MOPS reports company {code} as {market or '<missing>'}, but the ticker "
            f"suffix implies {expected}; refusing to use it"
        )
    return market


def _roc_date(value: str) -> date:
    m = _ROC_DATE_RE.fullmatch(value.strip())
    if not m:
        raise ValueError(value)
    return date(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))


def _timestamp(api: str, date_text, time_text) -> datetime:
    if not isinstance(date_text, str) or not isinstance(time_text, str):
        raise MopsUnavailableError(f"MOPS {api} layout mismatch: non-text timestamp")
    t = _TIME_RE.fullmatch(time_text.strip())
    try:
        d = _roc_date(date_text)
        if not t:
            raise ValueError(time_text)
        return datetime(d.year, d.month, d.day, int(t.group(1)), int(t.group(2)),
                        int(t.group(3)), tzinfo=TAIPEI)
    except ValueError as exc:
        raise MopsUnavailableError(
            f"MOPS {api} returned an unparseable timestamp {date_text!r} {time_text!r}"
        ) from exc


def _text(api: str, value, label: str) -> str:
    if not isinstance(value, str):
        raise MopsUnavailableError(f"MOPS {api} layout mismatch: {label!r} is not text")
    return value.strip()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class _Announcement:
    published: datetime
    code: str
    name: str
    market: str
    subject: str
    detail_ref: object = None
    detail_status: str = "not fetched (beyond detail cap)"
    detail: dict[str, str] = field(default_factory=dict)


def _parse_list(result: dict, code: str, board: str) -> list[_Announcement]:
    market = _check_company(_LIST_API, result, code, board)
    titles = result.get("titles")
    mapping = _title_map(_LIST_API, titles, _LIST_REQUIRED, _LIST_OPTIONAL)
    data = result.get("data")
    if not isinstance(data, list):
        raise MopsUnavailableError(f"MOPS {_LIST_API} layout mismatch: data is not a list")
    announcements = []
    for row in data:
        if not isinstance(row, list) or len(row) != len(titles):
            raise MopsUnavailableError(
                f"MOPS {_LIST_API} layout mismatch: row width does not match {len(titles)} titles"
            )
        row_code = _text(_LIST_API, row[mapping["code"]], "公司代號")
        if row_code != code:
            raise MopsUnavailableError(
                f"MOPS {_LIST_API} layout mismatch: row for company {row_code!r} in a "
                f"{code!r} query"
            )
        announcements.append(_Announcement(
            published=_timestamp(_LIST_API, row[mapping["date"]], row[mapping["time"]]),
            code=row_code,
            name=_text(_LIST_API, row[mapping["name"]], "公司名稱"),
            market=market,
            subject=_text(_LIST_API, row[mapping["subject"]], "主旨"),
            detail_ref=row[mapping["detail_ref"]] if "detail_ref" in mapping else None,
        ))
    return announcements


def _detail_body(ann: _Announcement, code: str, board: str) -> dict | None:
    """Validated detail request body from the list row, or None if unusable.

    The endpoint is fixed; a reference naming another API is never followed.
    A reference that names the other board is a board mismatch.
    """
    ref = ann.detail_ref
    if not isinstance(ref, dict) or ref.get("apiName") != _DETAIL_API:
        return None
    params = ref.get("parameters")
    if not isinstance(params, dict):
        return None
    kind = str(params.get("marketKind", "")).strip()
    if kind and kind != board:
        raise MopsBoardMismatchError(
            f"MOPS detail reference for company {code} names board {kind!r}, but the "
            f"ticker suffix implies {board!r}; refusing to use it"
        )
    body = {
        "companyId": str(params.get("companyId", "")).strip(),
        "marketKind": kind,
        "enterDate": str(params.get("enterDate", "")).strip(),
        "serialNumber": str(params.get("serialNumber", "")).strip(),
    }
    if (body["companyId"] != code or body["marketKind"] != board
            or not re.fullmatch(r"\d{7}", body["enterDate"])
            or not re.fullmatch(r"\d{1,4}", body["serialNumber"])):
        return None
    return body


def _fetch_detail(client: _Client, ann: _Announcement, code: str, board: str) -> None:
    """Attach detail to ``ann`` or mark it unavailable. Board mismatch raises."""
    body = _detail_body(ann, code, board)
    if body is None:
        ann.detail_status = "detail unavailable: no usable detail reference in the list row"
        return
    try:
        result = client.call(_DETAIL_API, body)
        if result is None:
            ann.detail_status = "detail unavailable: MOPS returned no matching detail"
            return
        _check_company(_DETAIL_API, result, code, board)
        titles = result.get("titles")
        mapping = _title_map(_DETAIL_API, titles, _DETAIL_REQUIRED, {})
        data = result.get("data")
        if not isinstance(data, list) or len(data) != 1:
            raise MopsUnavailableError(
                f"MOPS {_DETAIL_API} layout mismatch: expected exactly one detail row"
            )
        row = data[0]
        if not isinstance(row, list) or len(row) != len(titles):
            raise MopsUnavailableError(
                f"MOPS {_DETAIL_API} layout mismatch: row width does not match titles"
            )
        published = _timestamp(_DETAIL_API, row[mapping["date"]], row[mapping["time"]])
        if published != ann.published:
            ann.detail_status = "detail unavailable: detail timestamp does not match the list"
            return
        ann.detail = {
            key: _text(_DETAIL_API, row[mapping[key]], label)
            for key, label in _DETAIL_REQUIRED.items()
            if key not in ("date", "time")
        }
        ann.detail_status = "fetched"
    except MopsBoardMismatchError:
        raise
    except MopsUnavailableError as exc:
        logger.warning("MOPS detail unavailable for %s %s: %s", code, body, exc)
        ann.detail_status = f"detail unavailable: {exc}"


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------

def _month_requests(start: date, end: date, code: str) -> list[dict]:
    """One list request per calendar month covering ``[start, end]``."""
    bodies = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last = calendar.monthrange(year, month)[1]
        first_day = start.day if (year, month) == (start.year, start.month) else 1
        last_day = end.day if (year, month) == (end.year, end.month) else last
        bodies.append({
            "companyId": code,
            "year": str(year - 1911),
            "month": str(month),
            "firstDay": str(first_day),
            "lastDay": str(last_day),
        })
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return bodies


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def _cell(value: str) -> str:
    return re.sub(r"\s+", " ", value).replace("|", "\\|").strip()


def _iso_or_raw(value: str) -> str:
    try:
        return _roc_date(value).isoformat()
    except ValueError:
        return value


def _clip(text: str) -> str:
    if len(text) <= _DETAIL_TEXT_LIMIT:
        return text
    return (text[:_DETAIL_TEXT_LIMIT]
            + f" …[truncated by adapter: {len(text) - _DETAIL_TEXT_LIMIT} more characters]")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def get_material_announcements(
    ticker: str,
    curr_date: str,
    look_back_days: int = 30,
    detail_limit: int = _MAX_DETAIL_LIMIT,
) -> str:
    """Official MOPS material announcements for a Taiwan-listed ticker.

    Lists announcements published in the ``look_back_days`` before
    ``curr_date`` and fetches full detail for at most the newest
    ``detail_limit`` (capped at 5). Historical runs exclude ``curr_date``
    itself; live runs include what was published up to now.

    Raises :class:`NoMarketDataError` for non-Taiwan tickers without any
    request, :class:`MopsBoardMismatchError` when MOPS lists the company on
    the other board, and :class:`MopsUnavailableError` when the list cannot be
    retrieved or validated. A failed detail request only marks that
    announcement's detail unavailable.
    """
    code, board = split_taiwan_ticker(ticker, "MOPS material announcements")
    try:
        as_of = datetime.strptime(curr_date, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"curr_date must be yyyy-mm-dd, got {curr_date!r}") from exc
    look_back_days = min(max(1, int(look_back_days)), _MAX_LOOK_BACK_DAYS)
    detail_limit = min(max(0, int(detail_limit)), _MAX_DETAIL_LIMIT)

    now = _now().astimezone(TAIPEI)
    today = now.date()
    live = as_of >= today
    if live:
        window_start = today - timedelta(days=look_back_days)
        window_end = today
    else:
        window_start = as_of - timedelta(days=look_back_days)
        window_end = as_of - timedelta(days=1)   # curr_date is never requested

    client = _Client()
    announcements: list[_Announcement] = []
    confirmed_market = None
    for body in _month_requests(window_start, window_end, code):
        result = client.call(_LIST_API, body)
        if result is None:
            continue
        rows = _parse_list(result, code, board)
        confirmed_market = BOARD_MARKET_NAME[board]
        announcements.extend(rows)

    # Client-side point-in-time filter on the publication timestamp (never
    # enterDate), independent of the day bounds sent to MOPS.
    def visible(a: _Announcement) -> bool:
        d = a.published.date()
        if d < window_start:
            return False
        return a.published <= now if live else d <= window_end

    announcements = sorted(
        (a for a in announcements if visible(a)), key=lambda a: a.published, reverse=True
    )
    for ann in announcements[:detail_limit]:
        _fetch_detail(client, ann, code, board)

    header = [
        f"# Material Announcements (MOPS 重大訊息, official filings) for {ticker.strip().upper()} "
        f"(MOPS company code {code}, expected board {BOARD_MARKET_NAME[board]})",
        f"# Point-in-time as of: {curr_date}",
        "# Source: MOPS 歷史重大訊息 API (POST https://mops.twse.com.tw/mops/api/t05st01; "
        "detail via t05st01_detail)",
    ]
    if live:
        header.append(
            f"# Live run: announcements published from {window_start} up to "
            f"{now:%Y-%m-%d %H:%M:%S} Taiwan time (UTC+8) are included."
        )
    else:
        header.append(
            f"# Historical run: announcements published from {window_start} to {window_end} "
            f"(Taiwan time) are included; {WITHHELD_SAME_DAY_NOTICE} ({curr_date} is not "
            f"queried). Withheld is not evidence that {curr_date} had no announcements."
        )
    header.append(
        "# 發言日期/發言時間 is when MOPS published the announcement; 事實發生日 (in the "
        "detail) is the event date it reports and can be earlier."
    )
    header.append(
        "# Material announcements are factual filings, not a bullish or bearish signal by themselves."
    )

    if not announcements:
        board_note = (
            "" if confirmed_market else
            " MOPS returned no matching records for every queried month, so the board "
            "could not be confirmed from the response."
        )
        return "\n".join(header) + (
            f"\n\n## Result: no material announcements found\n"
            f"MOPS was queried and returned no material announcements for company {code} "
            f"in this window.{board_note}"
        )

    fetched = sum(1 for a in announcements if a.detail_status == "fetched")
    attempted = min(detail_limit, len(announcements))
    lines = [
        f"## {len(announcements)} announcements (newest first)",
        f"Detail was requested for the newest {attempted} (cap {detail_limit}); "
        f"{fetched} fetched, {attempted - fetched} unavailable. Older announcements "
        f"are list-level only. Detail content is only what MOPS returned.",
        "",
        "| # | 發言日期 | 發言時間 | 公司代號 | 公司名稱 | 市場別 | 主旨 | Detail |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for i, a in enumerate(announcements, 1):
        lines.append(
            f"| {i} | {a.published:%Y-%m-%d} | {a.published:%H:%M:%S} | {a.code} "
            f"| {_cell(a.name)} | {a.market} | {_cell(a.subject)} | {_cell(a.detail_status)} |"
        )
    details = [
        (i, a) for i, a in enumerate(announcements, 1) if a.detail_status == "fetched"
    ]
    if details:
        lines += ["", "## Announcement details"]
        for i, a in details:
            lines += [
                "",
                f"### {i}. {a.published:%Y-%m-%d %H:%M:%S} {_cell(a.subject)}",
                f"- 符合條款: {_cell(a.detail['clause']) or 'n/a'}",
                f"- 事實發生日 (event date): {_iso_or_raw(a.detail['fact_date']) or 'n/a'}",
                "- 說明:",
            ]
            lines += [f"  > {line}" if line.strip() else "  >"
                      for line in _clip(a.detail["text"]).splitlines()]
    return "\n".join(header) + "\n\n" + "\n".join(lines)
