"""Taiwan institutional investor flows (三大法人買賣超) from TWSE and TPEx.

Official daily whole-market tables, one per board:

* ``.TW`` (TWSE listed): ``GET https://www.twse.com.tw/rwd/zh/fund/T86`` with
  ``date=YYYYMMDD&selectType=ALLBUT0999&response=json``.
* ``.TWO`` (TPEx listed): ``POST https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade``
  with form ``type=Daily&sect=EW&date=YYYY/MM/DD&response=csv``. CSV is used
  because its headers carry the investor group (``外資自營商-買進股數``); the JSON
  variant only repeats the leaf labels.

Source facts that shape the design (verified against the live sites):

* No key, cookie or token. Both serve only whole-market tables, so a day's
  table is fetched once and shared across tickers through a process cache.
* Units are shares. TWSE states ``單位：股``; every TPEx label ends in ``股數``.
* Supported layouts: TWSE from 2017-12-18 (19 columns) and TPEx from
  2018-01-15 (24 columns). Earlier dates use older layouts whose foreign
  investor scope is not stated, and are refused rather than guessed.
* Non-trading days, today before publication, and some bad inputs all return
  the same "no data" reply, so an empty reply only means "no table for that
  date". TPEx silently substitutes another date for some malformed requests,
  so the reported date is always checked against the requested one.
* Tables omit securities with no institutional trades; a missing ticker is
  missing data, never zero.
* Both exchanges state that foreign-dealer trades are already counted under
  dealers and are not added to the total, so the total is
  foreign (excluding foreign dealers) + investment trust + dealers.

Point-in-time: institutional data for a trading day is published after the
close. A historical run uses only trade dates before ``curr_date`` and never
requests ``curr_date``; a live run includes today's table only when the
exchange has published it under today's date.
"""

from __future__ import annotations

import csv
import io
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .bounded_http import bounded_request, decode_json_object
from .errors import NoMarketDataError, VendorError
from .taiwan_common import (
    norm_header,
    split_taiwan_ticker,
    taipei_now as _now,  # test seam; the shared Taiwan-time clock
)

_TOOL_SOURCE = "TWSE/TPEx institutional flows"
_TWSE_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"
_TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
_SOURCE_LABEL = {"sii": "TWSE", "otc": "TPEx"}
_BOARD_LABEL = {"sii": "TWSE listed (上市)", "otc": "TPEx listed (上櫃)"}
_SOURCE_DESCRIPTION = {
    "sii": f"TWSE 三大法人買賣超日報 T86 ({_TWSE_URL}, selectType=ALLBUT0999)",
    "otc": f"TPEx 三大法人買賣明細資訊 ({_TPEX_URL}, sect=EW, CSV)",
}

# Gap between the end of one request and the start of the next, per exchange,
# across the whole process. These are the spacings verified not to trip the
# exchanges' rate limiting during research; faster values are untested.
_REQUEST_SPACING_SECONDS = {"sii": 5.0, "otc": 6.0}
_SUPPORTED_FROM = {"sii": date(2017, 12, 18), "otc": date(2018, 1, 15)}

_DEFAULT_LOOK_BACK = 5
_MAX_LOOK_BACK = 20
_EXTRA_WEEKDAYS = 10          # holidays such as Lunar New Year span up to ~7 weekdays
_CACHE_MAX_TABLES = 64

NOT_IN_TABLE = "not in official table (no institutional trades recorded, or not listed)"
ABSENT = "absent in source"
WITHHELD_NOTICE = (
    "trade dates on or after {curr_date} are withheld because institutional data is "
    "published after the close and the decision time is unknown"
)

# Canonical fields, in output order.
GROUPS = (
    ("foreign", "Foreign (excl. foreign dealers)"),
    ("foreign_dealer", "Foreign dealers"),
    ("investment_trust", "Investment trust"),
    ("dealer", "Dealers"),
    ("dealer_proprietary", "Dealers proprietary"),
    ("dealer_hedge", "Dealers hedge"),
)
FIELDS = tuple(f"{g}_{s}" for g, _ in GROUPS for s in ("buy", "sell", "net")) + (
    "total_institutional_net",
)

_TWSE_LABELS = {
    "code": "證券代號",
    "name": "證券名稱",
    "foreign_buy": "外陸資買進股數(不含外資自營商)",
    "foreign_sell": "外陸資賣出股數(不含外資自營商)",
    "foreign_net": "外陸資買賣超股數(不含外資自營商)",
    "foreign_dealer_buy": "外資自營商買進股數",
    "foreign_dealer_sell": "外資自營商賣出股數",
    "foreign_dealer_net": "外資自營商買賣超股數",
    "investment_trust_buy": "投信買進股數",
    "investment_trust_sell": "投信賣出股數",
    "investment_trust_net": "投信買賣超股數",
    "dealer_net": "自營商買賣超股數",
    "dealer_proprietary_buy": "自營商買進股數(自行買賣)",
    "dealer_proprietary_sell": "自營商賣出股數(自行買賣)",
    "dealer_proprietary_net": "自營商買賣超股數(自行買賣)",
    "dealer_hedge_buy": "自營商買進股數(避險)",
    "dealer_hedge_sell": "自營商賣出股數(避險)",
    "dealer_hedge_net": "自營商買賣超股數(避險)",
    "total_institutional_net": "三大法人買賣超股數",
}
_TPEX_LABELS = {
    "code": "代號",
    "name": "名稱",
    "foreign_buy": "外資及陸資(不含外資自營商)-買進股數",
    "foreign_sell": "外資及陸資(不含外資自營商)-賣出股數",
    "foreign_net": "外資及陸資(不含外資自營商)-買賣超股數",
    "foreign_dealer_buy": "外資自營商-買進股數",
    "foreign_dealer_sell": "外資自營商-賣出股數",
    "foreign_dealer_net": "外資自營商-買賣超股數",
    "investment_trust_buy": "投信-買進股數",
    "investment_trust_sell": "投信-賣出股數",
    "investment_trust_net": "投信-買賣超股數",
    "dealer_buy": "自營商-買進股數",
    "dealer_sell": "自營商-賣出股數",
    "dealer_net": "自營商-買賣超股數",
    "dealer_proprietary_buy": "自營商(自行買賣)-買進股數",
    "dealer_proprietary_sell": "自營商(自行買賣)-賣出股數",
    "dealer_proprietary_net": "自營商(自行買賣)-買賣超股數",
    "dealer_hedge_buy": "自營商(避險)-買進股數",
    "dealer_hedge_sell": "自營商(避險)-賣出股數",
    "dealer_hedge_net": "自營商(避險)-買賣超股數",
    "total_institutional_net": "三大法人買賣超股數合計",
}
_LABELS = {"sii": _TWSE_LABELS, "otc": _TPEX_LABELS}

_TWSE_NO_TABLE_STATS = ("很抱歉，沒有符合條件的資料!",)
_TWSE_NO_TABLE_PREFIXES = ("查詢日期大於可查詢最大日期",)
_TWSE_UNIT = norm_header("單位：股")
_TPEX_TITLE_DATE_RE = re.compile(r"\s*(\d{2,3})年(\d{1,2})月(\d{1,2})日")
_TPEX_FILENAME_RE = re.compile(r'filename="?([^";]+)"?')


class TaiwanExchangeUnavailableError(VendorError):
    """TWSE or TPEx institutional data could not be retrieved or validated."""


def _fail(board: str, message: str) -> TaiwanExchangeUnavailableError:
    return TaiwanExchangeUnavailableError(f"{_SOURCE_LABEL[board]} {message}")


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _monotonic() -> float:
    return time.monotonic()


# --------------------------------------------------------------------------
# Parsed tables
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Table:
    board: str
    trade_date: date
    columns: dict[str, int]              # field key -> column index
    rows: dict[str, tuple[str, ...]]     # company code -> raw cells


def _map_labels(board: str, labels: list) -> dict[str, int]:
    """Exact normalized label -> column index; missing or duplicate labels fail."""
    positions: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if not isinstance(label, str):
            raise _fail(board, f"layout mismatch: column #{index} has no text label")
        positions.setdefault(norm_header(label), []).append(index)
    mapping = {}
    for key, label in _LABELS[board].items():
        found = positions.get(norm_header(label), [])
        if not found:
            raise _fail(board, f"layout mismatch: required column {label!r} not found")
        if len(found) > 1:
            raise _fail(board, f"layout mismatch: column {label!r} appears {len(found)} times")
        mapping[key] = found[0]
    return mapping


def _index_rows(board: str, data: list, width: int, code_index: int) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for row in data:
        if not isinstance(row, list | tuple) or len(row) != width:
            raise _fail(board, f"layout mismatch: row width does not match {width} columns")
        if not all(isinstance(cell, str) for cell in row):
            raise _fail(board, "layout mismatch: non-text cell in a data row")
        code = unicodedata.normalize("NFKC", row[code_index]).strip()
        if not code:
            raise _fail(board, "layout mismatch: data row without a security code")
        if code in rows:
            raise _fail(board, f"layout mismatch: security code {code!r} appears twice")
        rows[code] = tuple(row)
    return rows


def _fetch_twse(trade_date: date) -> _Table | None:
    ymd = trade_date.strftime("%Y%m%d")
    resp = bounded_request(
        "GET", _TWSE_URL, error_cls=TaiwanExchangeUnavailableError, source="TWSE",
        accept="application/json",
        params={"date": ymd, "selectType": "ALLBUT0999", "response": "json"},
    )
    env = decode_json_object(resp, error_cls=TaiwanExchangeUnavailableError, source="TWSE")
    stat = str(env.get("stat", "")).strip()
    if stat in _TWSE_NO_TABLE_STATS or stat.startswith(_TWSE_NO_TABLE_PREFIXES):
        return None
    if stat != "OK":
        raise _fail("sii", f"T86 returned stat {stat or '<missing>'!r}")
    reported = str(env.get("date", "")).strip()
    if reported != ymd:
        raise _fail("sii", f"T86 reported date {reported or '<missing>'!r} for a {ymd} request")
    if norm_header(str(env.get("hints", ""))) != _TWSE_UNIT:
        raise _fail("sii", f"T86 unit is {env.get('hints')!r}, expected '單位：股'")
    fields, data = env.get("fields"), env.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise _fail("sii", "layout mismatch: T86 fields or data missing")
    columns = _map_labels("sii", fields)
    rows = _index_rows("sii", data, len(fields), columns["code"])
    if not rows:
        raise _fail("sii", f"T86 returned OK with no rows for {ymd}")
    return _Table("sii", trade_date, columns, rows)


def _fetch_tpex(trade_date: date) -> _Table | None:
    roc_year = trade_date.year - 1911
    resp = bounded_request(
        "POST", _TPEX_URL, error_cls=TaiwanExchangeUnavailableError, source="TPEx",
        accept="text/csv",
        data={"type": "Daily", "sect": "EW", "date": trade_date.strftime("%Y/%m/%d"),
              "response": "csv"},
    )
    content_type = resp.headers.get("Content-Type", "").lower()

    if "json" in content_type:
        # TPEx answers a CSV request for a day without a table with a JSON body.
        env = decode_json_object(resp, error_cls=TaiwanExchangeUnavailableError, source="TPEx")
        tables = env.get("tables")
        if not isinstance(tables, list) or not tables:
            raise _fail("otc", "layout mismatch: JSON reply without tables")
        expected = f"{roc_year}/{trade_date.month:02d}/{trade_date.day:02d}"
        reported = {str(t.get("date", "")).strip() for t in tables if isinstance(t, dict)} - {""}
        if reported != {expected}:
            raise _fail("otc", f"reported date {sorted(reported)!r} for a {expected} request")
        if any(isinstance(t, dict) and t.get("data") for t in tables):
            raise _fail("otc", "returned table rows as JSON for a CSV request")
        return None

    if "csv" not in content_type:
        raise _fail("otc", f"unexpected content type {content_type or '<missing>'!r}")
    try:
        text = resp.body.decode("cp950")
    except UnicodeDecodeError as exc:
        raise _fail("otc", "CSV is not valid Big5 (cp950)") from exc
    lines = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if not lines:
        raise _fail("otc", "empty CSV")

    title = _TPEX_TITLE_DATE_RE.match(lines[0][0])
    try:
        title_date = (date(int(title.group(1)) + 1911, int(title.group(2)), int(title.group(3)))
                      if title else None)
    except ValueError:
        title_date = None
    if title_date != trade_date:
        raise _fail("otc", f"CSV title reports {lines[0][0][:12]!r} for a {trade_date} request")
    filename = _TPEX_FILENAME_RE.search(resp.headers.get("Content-Disposition", ""))
    expected_name = f"BIGD_{roc_year}{trade_date:%m%d}.csv"
    if filename and filename.group(1) != expected_name:
        raise _fail("otc", f"CSV file {filename.group(1)!r} does not match {expected_name!r}")

    header_at = next((i for i, row in enumerate(lines) if norm_header(row[0]) == "代號"), None)
    if header_at is None:
        raise _fail("otc", "layout mismatch: CSV header row not found")
    header = lines[header_at]
    # Every required TPEx label names its unit (股數), so the exact label match
    # is also the unit check.
    columns = _map_labels("otc", header)
    # Notes at the end of the file are single-cell rows.
    data = [row for row in lines[header_at + 1:] if len(row) > 1]
    rows = _index_rows("otc", data, len(header), columns["code"])
    if not rows:
        raise _fail("otc", f"CSV for {trade_date} has a header but no rows")
    return _Table("otc", trade_date, columns, rows)


# --------------------------------------------------------------------------
# Cache and request pacing
# --------------------------------------------------------------------------

_lock = threading.Lock()
_cache: OrderedDict[tuple[str, str], _Table] = OrderedDict()
_last_request_done: dict[str, float] = {}


def clear_cache() -> None:
    """Drop cached tables (tests, or a long-lived process)."""
    with _lock:
        _cache.clear()


def _load_table(board: str, trade_date: date, today: date) -> _Table | None:
    """Cached table for a completed past trading day, else a paced fetch.

    Only non-empty tables for dates before today are cached; today's table and
    "no table" replies are always fetched again.
    """
    key = (board, trade_date.isoformat())
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        last = _last_request_done.get(board)
    if last is not None:
        wait = _REQUEST_SPACING_SECONDS[board] - (_monotonic() - last)
        if wait > 0:
            _sleep(wait)
    try:
        table = (_fetch_twse if board == "sii" else _fetch_tpex)(trade_date)
    finally:
        with _lock:
            _last_request_done[board] = _monotonic()
    if table is not None and trade_date < today:
        with _lock:
            _cache[key] = table
            _cache.move_to_end(key)
            while len(_cache) > _CACHE_MAX_TABLES:
                _cache.popitem(last=False)
    return table


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------

def _int(board: str, cell: str, key: str) -> int:
    text = unicodedata.normalize("NFKC", cell).strip().replace(",", "")
    if not re.fullmatch(r"-?\d+", text):
        raise _fail(board, f"non-numeric {key} value {cell!r}")
    return int(text)


def _record(table: _Table, code: str) -> dict | None:
    """Parsed, integrity-checked values for ``code``, or None if it is not listed."""
    cells = table.rows.get(code)
    if cells is None:
        return None
    rec: dict = {"name": cells[table.columns["name"]].strip()}
    for key in FIELDS:
        index = table.columns.get(key)
        rec[key] = None if index is None else _int(table.board, cells[index], key)

    def check(ok: bool, what: str) -> None:
        if not ok:
            raise _fail(table.board, f"integrity check failed for {code} on {table.trade_date}: {what}")

    for group, _ in GROUPS:
        buy, sell, net = rec[f"{group}_buy"], rec[f"{group}_sell"], rec[f"{group}_net"]
        if buy is not None and sell is not None:
            check(buy - sell == net, f"{group} buy - sell != net")
    check(rec["dealer_proprietary_net"] + rec["dealer_hedge_net"] == rec["dealer_net"],
          "dealer proprietary + hedge != dealer net")
    if rec["dealer_buy"] is not None:
        check(rec["dealer_proprietary_buy"] + rec["dealer_hedge_buy"] == rec["dealer_buy"],
              "dealer proprietary + hedge buys != dealer buys")
        check(rec["dealer_proprietary_sell"] + rec["dealer_hedge_sell"] == rec["dealer_sell"],
              "dealer proprietary + hedge sells != dealer sells")
    check(rec["foreign_net"] + rec["investment_trust_net"] + rec["dealer_net"]
          == rec["total_institutional_net"],
          "foreign + investment trust + dealer != total")
    return rec


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def _fmt(value: int | None) -> str:
    return ABSENT if value is None else f"{value:,}"


def get_institutional_flows(
    ticker: str,
    curr_date: str,
    look_back_trading_days: int = _DEFAULT_LOOK_BACK,
) -> str:
    """Official daily institutional investor flows for a Taiwan-listed ticker.

    Returns up to ``look_back_trading_days`` (1-20) completed trading days,
    newest first, from TWSE (``.TW``) or TPEx (``.TWO``), point-in-time as of
    ``curr_date``.

    Raises :class:`NoMarketDataError` for non-Taiwan or non-numeric tickers
    (without any request) and when no supported table exists in the window;
    raises :class:`TaiwanExchangeUnavailableError` when a table cannot be
    retrieved or fails validation.
    """
    code, board = split_taiwan_ticker(ticker, _TOOL_SOURCE)
    try:
        as_of = datetime.strptime(curr_date, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"curr_date must be yyyy-mm-dd, got {curr_date!r}") from exc
    wanted = min(max(1, int(look_back_trading_days)), _MAX_LOOK_BACK)

    now = _now()
    today = now.date()
    live = as_of >= today
    floor = _SUPPORTED_FROM[board]
    source = _SOURCE_LABEL[board]

    found: list[_Table] = []
    skipped: list[date] = []
    today_status = None
    reached_floor = hit_cap = False
    weekdays_left = wanted + _EXTRA_WEEKDAYS
    day = today if live else as_of - timedelta(days=1)
    while len(found) < wanted:
        if day < floor:
            reached_floor = True
            break
        if day.weekday() >= 5:
            day -= timedelta(days=1)
            continue
        if weekdays_left == 0:
            hit_cap = True
            break
        weekdays_left -= 1
        table = _load_table(board, day, today)
        if table is not None:
            found.append(table)
        elif live and day == today:
            today_status = "not available"
        else:
            skipped.append(day)
        if live and day == today and table is not None:
            today_status = "included"
        day -= timedelta(days=1)

    if not found:
        if reached_floor:
            detail = (f"no supported {source} institutional-flow table before {curr_date}: dates "
                      f"before {floor} use an older layout that is not supported")
        else:
            detail = (f"no official {source} institutional-flow table found in the "
                      f"{wanted + _EXTRA_WEEKDAYS} weekdays examined up to {curr_date}")
        raise NoMarketDataError(ticker, canonical=code, detail=detail)

    records = [(t.trade_date, _record(t, code)) for t in found]

    header = [
        f"# Institutional Investor Flows (三大法人買賣超, official exchange data) for "
        f"{ticker.strip().upper()} ({_BOARD_LABEL[board]}, company code {code})",
        f"# Point-in-time as of: {curr_date}",
        f"# Source: {_SOURCE_DESCRIPTION[board]}",
        "# Unit: shares (股), as stated by the exchange.",
    ]
    if live:
        header.append(
            f"# Live run: today's ({today}) table is used only when the exchange has published "
            f"it under today's date; "
            + ("it is included." if today_status == "included" else
               "it is not available yet (not published, or not a trading day).")
        )
    else:
        header.append(
            "# Historical run: only completed trading days before the as-of date are used; "
            + WITHHELD_NOTICE.format(curr_date=curr_date)
            + ". Withheld is not evidence of any flow on those dates."
        )
    header += [
        "# Total = foreign (excluding foreign dealers) + investment trust + dealers. Per the "
        "exchanges' notes, foreign-dealer trades are already counted under dealers and are not "
        "added to the total.",
        "# A trade date or ticker missing from the official table is missing data, not zero. "
        "Flows are positioning evidence, not a buy or sell signal by themselves.",
    ]

    nets = [
        "## Net buy/sell (shares), newest first",
        "| Trade date | " + " | ".join(label for _, label in GROUPS) + " | Total |",
        "|---|" + "---:|" * (len(GROUPS) + 1),
    ]
    trades = [
        "## Buys and sells (shares)",
        "| Trade date | " + " | ".join(f"{label} buy | {label} sell" for _, label in GROUPS) + " |",
        "|---|" + "---:|" * (2 * len(GROUPS)),
    ]
    names = set()
    for trade_date, rec in records:
        if rec is None:
            nets.append(f"| {trade_date} | {NOT_IN_TABLE} |" + " n/a |" * len(GROUPS))
            trades.append(f"| {trade_date} | {NOT_IN_TABLE} |" + " n/a |" * (2 * len(GROUPS) - 1))
            continue
        names.add(rec["name"])
        nets.append(
            f"| {trade_date} | "
            + " | ".join(_fmt(rec[f"{g}_net"]) for g, _ in GROUPS)
            + f" | {_fmt(rec['total_institutional_net'])} |"
        )
        trades.append(
            f"| {trade_date} | "
            + " | ".join(f"{_fmt(rec[f'{g}_buy'])} | {_fmt(rec[f'{g}_sell'])}" for g, _ in GROUPS)
            + " |"
        )

    notes = []
    if names:
        notes.append(f"Company name in the exchange table: {', '.join(sorted(names))}.")
    if len(found) < wanted:
        notes.append(f"Only {len(found)} of the {wanted} requested trading days were found.")
    if skipped:
        notes.append(
            "Weekdays without an official table (not a trading day, or not published): "
            + ", ".join(str(d) for d in skipped) + "."
        )
    if reached_floor:
        notes.append(
            f"Dates before {floor} use an older {source} layout and are not supported; "
            "the look-back stopped there."
        )
    if hit_cap:
        notes.append(
            f"Stopped after examining {wanted + _EXTRA_WEEKDAYS} weekdays."
        )

    body = "\n".join(nets) + "\n\n" + "\n".join(trades)
    if notes:
        body += "\n\n" + "\n".join(f"- {n}" for n in notes)
    return "\n".join(header) + "\n\n" + body
