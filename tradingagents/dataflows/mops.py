"""MOPS (公開資訊觀測站) monthly revenue adapter for Taiwan-listed companies.

Taiwan-listed companies must file the previous month's operating revenue with
MOPS by the 10th of each month, and MOPS publishes one static per-market table
per month (``t21sc03``: 每月營業收入統計表) covering every company on that
board. This adapter reads those tables so the fundamentals analyst can see
official monthly revenue for ``.TW`` (TWSE, ``sii``) and ``.TWO`` (TPEx,
``otc``) tickers. It is supplementary operating evidence next to the yfinance
statements, not a replacement for them.

Source facts that shape the design (verified against the live site):

* The tables live on the legacy host ``mopsov.twse.com.tw`` as Big5 HTML at
  ``/nas/t21/{sii|otc}/t21sc03_{ROC year}_{month}_0.html`` (``_0`` = domestic
  companies). Plain GET, no cookies, no form fields, no API key.
* The host's certificate lacks a Subject Key Identifier, which Python 3.13+'s
  default strict X.509 mode in ``urllib`` rejects. The shared bounded transport
  (``bounded_http``, via ``mops_common``) uses ``requests`` with verification on.
* Amounts are stated by the source in ``千元`` (TWD thousands); MoM / YoY /
  cumulative-YoY percentages are published by MOPS, not derived here.
* A month's file appears as an empty template before anyone has filed and
  fills in as companies report. There is NO per-company announcement date,
  so for a historical run point-in-time safety is enforced with the statutory
  deadline (see :func:`_public_from`), and the withheld months are named in
  the output rather than silently dropped.
* MOPS regenerates past tables (a 2025 file carries a 2026 出表日期), so the
  values are as currently filed and may include corrections made after the
  original announcement. That residual limitation is stated in the output of
  every historical run; it cannot be removed with this source.

Fail-closed by construction: a non-200 status, oversized body, missing unit
label, or unexpected table layout raises a typed vendor error and never
returns partial numbers or raw HTML to the agent.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from datetime import date
from html.parser import HTMLParser

from .errors import NoMarketDataError
from .mops_common import (
    MopsUnavailableError,
    http_get as _http_get,  # test seam
)
from .taiwan_common import (
    norm_header,
    resolve_as_of,
    split_taiwan_ticker,
    taipei_now as _now,  # test seam; the shared Taiwan-time clock
)

logger = logging.getLogger(__name__)

_HOST = "https://mopsov.twse.com.tw"
_TABLE_PATH = "/nas/t21/{board}/t21sc03_{roc_year}_{month}_0.html"
_ENCODING = "cp950"                  # MOPS declares big5; cp950 is its superset

# The board code comes from taiwan_common.split_taiwan_ticker (sii / otc) and is
# also the MOPS path segment for this table.
_BOARD_LABEL = {"sii": "TWSE listed (上市)", "otc": "TPEx listed (上櫃)"}

# Statutory filing deadline: revenue for month M is due by the 10th of M+1
# (證券交易法 §36). The deadline moves to the next business day when the 10th
# falls on a weekend or holiday (Lunar New Year can push it most of a week),
# and the table carries no actual announcement dates, so a historical run
# treats month M as public only from the 16th of M+1.
_FILING_DEADLINE_DAY = 10
_HISTORICAL_PUBLIC_FROM_DAY = 16

# look_back_months is an LLM-supplied tool argument and each month is one
# whole-board table request, so it is capped (two years covers YoY context).
_MAX_LOOK_BACK_MONTHS = 24

# A month's table keeps filling in until its filing deadline has safely passed,
# so only months past that point are cached, and even those are re-fetched
# after the TTL so a long-lived process picks up later corrections.
_TABLE_CACHE_MAX = 64
_COMPLETE_TABLE_TTL_SECONDS = 12 * 60 * 60

# Columns are located by their header label, never by position. Keys are the
# fields the adapter emits; values are the leaf header labels exactly as the
# live t21sc03 tables print them (after norm_header), identical across every
# sii and otc industry table. Each required label must resolve to exactly one
# single-column header cell or the whole table is rejected.
_REQUIRED_COLUMNS = {
    "code":          "公司代號",
    "name":          "公司名稱",
    "revenue":       "當月營收",
    "prev_month":    "上月營收",
    "last_year":     "去年當月營收",
    "mom_pct":       "上月比較增減(%)",
    "yoy_pct":       "去年同月增減(%)",
    "ytd":           "當月累計營收",
    "ytd_last_year": "去年累計營收",
    "ytd_yoy_pct":   "前期比較增減(%)",
}
# Optional: MOPS comments 備註 out of the whole-market total table. When present
# it must still be unambiguous.
_OPTIONAL_COLUMNS = {
    "remark": "備註",
}
_UNIT_RE = re.compile(r"單位[：:]\s*([^\s<]+)")
# Every t21sc03 table states its amounts in TWD thousands. Any other unit label
# anywhere on the page fails closed rather than being reported under the wrong
# unit (or under the first month's unit in a multi-month report).
_EXPECTED_UNIT = "千元"
_ROC_DATE_RE = re.compile(r"出表日期[：:]\s*(\d{2,3})/(\d{2})/(\d{2})")


def _table_url(board: str, year: int, month: int) -> str:
    return _HOST + _TABLE_PATH.format(board=board, roc_year=year - 1911, month=month)


_cache_lock = threading.Lock()
_table_cache: OrderedDict[tuple[str, int, int], tuple[float, _ParsedTable]] = OrderedDict()


def _monotonic() -> float:
    return time.monotonic()


def _fetch_table(board: str, year: int, month: int) -> _ParsedTable:
    """Download and parse one board-month table.

    Tables for months whose filing deadline has safely passed (see
    :func:`_public_from`) are cached for ``_COMPLETE_TABLE_TTL_SECONDS``; a
    month that may still be receiving filings is always fetched again.
    """
    key = (board, year, month)
    with _cache_lock:
        entry = _table_cache.get(key)
        if entry is not None and _monotonic() - entry[0] < _COMPLETE_TABLE_TTL_SECONDS:
            _table_cache.move_to_end(key)
            return entry[1]
    url = _table_url(board, year, month)
    body = _http_get(url)
    try:
        table = _parse_table(body)
    except MopsUnavailableError as exc:
        raise MopsUnavailableError(f"{exc} ({url})") from None
    if _public_from(year, month) <= _now().date():
        with _cache_lock:
            _table_cache[key] = (_monotonic(), table)
            _table_cache.move_to_end(key)
            while len(_table_cache) > _TABLE_CACHE_MAX:
                _table_cache.popitem(last=False)
    return table


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

class _ParsedTable:
    __slots__ = ("unit", "report_date", "rows")

    def __init__(self, unit: str, report_date: str | None, rows: dict[str, dict[str, str]]):
        self.unit = unit
        self.report_date = report_date
        self.rows = rows  # company code -> {field key: raw cell text}


class _Cell:
    __slots__ = ("parts", "is_header", "colspan", "rowspan")

    def __init__(self, is_header: bool, colspan: int, rowspan: int):
        self.parts: list[str] = []
        self.is_header = is_header
        self.colspan = colspan
        self.rowspan = rowspan

    @property
    def text(self) -> str:
        return "".join(self.parts)


class _Table:
    __slots__ = ("rows", "row", "cell")

    def __init__(self):
        self.rows: list[list[_Cell]] = []
        self.row: list[_Cell] | None = None
        self.cell: _Cell | None = None

    def close_cell(self):
        if self.cell is not None and self.row is not None:
            self.row.append(self.cell)
        self.cell = None

    def close_row(self):
        self.close_cell()
        if self.row is not None:
            self.rows.append(self.row)
        self.row = None


def _span(attrs, name: str) -> int:
    value = dict(attrs).get(name)
    try:
        return max(1, int(value)) if value is not None else 1
    except (TypeError, ValueError):
        return 1


class _TableCollector(HTMLParser):
    """Collect every ``<table>`` as rows of cells, keeping nested tables apart.

    Text is attributed to the innermost open table only, so a wrapper cell that
    contains a data table does not swallow the data table's text. ``<br>`` adds
    nothing, which joins split labels such as ``公司<br>代號``.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[_Table] = []
        self._stack: list[_Table] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._stack.append(_Table())
        elif not self._stack:
            return
        elif tag == "tr":
            self._stack[-1].close_row()
            self._stack[-1].row = []
        elif tag in ("td", "th"):
            table = self._stack[-1]
            if table.row is None:  # cell without <tr>: layout we do not trust
                table.row = []
            table.close_cell()
            table.cell = _Cell(tag == "th", _span(attrs, "colspan"), _span(attrs, "rowspan"))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if not self._stack:
            return
        if tag in ("td", "th"):
            self._stack[-1].close_cell()
        elif tag == "tr":
            self._stack[-1].close_row()
        elif tag == "table":
            table = self._stack.pop()
            table.close_row()
            self.tables.append(table)

    def handle_data(self, data):
        if self._stack and self._stack[-1].cell is not None:
            self._stack[-1].cell.parts.append(data)

    def close(self):
        super().close()
        while self._stack:  # unterminated tables
            table = self._stack.pop()
            table.close_row()
            self.tables.append(table)


def _grid(rows: list[list[_Cell]]) -> list[dict[int, tuple[_Cell, bool]]]:
    """Expand rowspan/colspan: grid[r][c] = (cell, starts_in_row_r)."""
    grid: list[dict[int, tuple[_Cell, bool]]] = [{} for _ in rows]
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while c in grid[r]:
                c += 1
            for dr in range(cell.rowspan):
                if r + dr >= len(rows):
                    break
                for dc in range(cell.colspan):
                    grid[r + dr][c + dc] = (cell, dr == 0)
            c += cell.colspan
    return grid


def _column_map(header_row: dict[int, tuple[_Cell, bool]]) -> dict[str, int]:
    """Map each required/optional field to its single column index, or raise."""
    by_label: dict[str, list[tuple[_Cell, list[int]]]] = {}
    seen: dict[int, tuple[_Cell, list[int]]] = {}
    for col in sorted(header_row):
        cell, _ = header_row[col]
        entry = seen.get(id(cell))
        if entry is None:
            entry = (cell, [])
            seen[id(cell)] = entry
            by_label.setdefault(norm_header(cell.text), []).append(entry)
        entry[1].append(col)

    def resolve(label: str, required: bool) -> int | None:
        matches = by_label.get(label, [])
        if not matches:
            if required:
                raise MopsUnavailableError(
                    f"MOPS table layout mismatch: header labels not found: [{label!r}]"
                )
            return None
        if len(matches) > 1:
            raise MopsUnavailableError(
                f"MOPS table layout mismatch: header label {label!r} appears "
                f"{len(matches)} times"
            )
        _, cols = matches[0]
        if len(cols) != 1:
            raise MopsUnavailableError(
                f"MOPS table layout mismatch: header label {label!r} spans columns "
                f"{cols}; cannot map it to one column"
            )
        return cols[0]

    mapping: dict[str, int] = {}
    for key, label in _REQUIRED_COLUMNS.items():
        mapping[key] = resolve(label, required=True)
    for key, label in _OPTIONAL_COLUMNS.items():
        col = resolve(label, required=False)
        if col is not None:
            mapping[key] = col
    return mapping


_CODE_LABEL = _REQUIRED_COLUMNS["code"]
_REQUIRED_LABELS = frozenset(_REQUIRED_COLUMNS.values())


def _extract_rows(tables: list[_Table]) -> dict[str, dict[str, str]]:
    """Read company rows by header label from every table that has a header row.

    A header row is any row with a header cell carrying a required label; it
    must then resolve every required label (see _column_map) and defines the
    column mapping for the rows below it in the same table. Rows above it (group
    headings) and tables without one (page wrappers) are ignored. A company row
    is one whose 公司代號 cell holds a 4-6 digit code, and it must span exactly
    as many columns as its header: a short or long row would shift values under
    the wrong labels, so it rejects the table instead.
    """
    rows: dict[str, dict[str, str]] = {}
    header_found = False
    for table in tables:
        grid = _grid(table.rows)
        mapping: dict[str, int] | None = None
        header_width = 0
        for r in range(len(table.rows)):
            cells = grid[r]
            if any(
                norm_header(cell.text) in _REQUIRED_LABELS
                for cell, starts in cells.values()
                if starts and cell.is_header
            ):
                mapping = _column_map(cells)
                header_width = max(cells) + 1
                header_found = True
                continue
            if mapping is None:
                # A company-looking row (leading data cell with a 4-6 digit code)
                # that no header row maps: the header moved or vanished. Refuse
                # rather than skip it, which would report missing data instead.
                first = next((cells[c][0] for c in sorted(cells) if cells[c][1]), None)
                if (first is not None and not first.is_header and re.fullmatch(
                        r"\d{4,6}", unicodedata.normalize("NFKC", first.text).strip())):
                    raise MopsUnavailableError(
                        f"MOPS table layout mismatch: header labels not found: no "
                        f"{_CODE_LABEL!r} header row above company rows"
                    )
                continue
            code_entry = cells.get(mapping["code"])
            if code_entry is None or not code_entry[1]:
                continue
            code = unicodedata.normalize("NFKC", code_entry[0].text).strip()
            if not re.fullmatch(r"\d{4,6}", code):
                continue  # 合計 and other non-company rows
            row_width = max(cells) + 1
            if row_width != header_width:
                raise MopsUnavailableError(
                    f"MOPS table layout mismatch: row for company {code} spans "
                    f"{row_width} columns but its header spans {header_width}"
                )
            record: dict[str, str] = {}
            for key, col in mapping.items():
                entry = cells.get(col)
                if entry is None or not entry[1]:
                    if key in _OPTIONAL_COLUMNS:
                        continue
                    raise MopsUnavailableError(
                        f"MOPS table layout mismatch: row for company {code} has no "
                        f"cell under {_REQUIRED_COLUMNS[key]!r}"
                    )
                cell = entry[0]
                if cell.colspan != 1:
                    raise MopsUnavailableError(
                        f"MOPS table layout mismatch: row for company {code} spans "
                        f"columns under {(_REQUIRED_COLUMNS | _OPTIONAL_COLUMNS)[key]!r}"
                    )
                record[key] = cell.text.strip()
            if code in rows:
                raise MopsUnavailableError(
                    f"MOPS table layout mismatch: company {code} appears in more than one row"
                )
            rows[code] = record
    if not header_found:
        raise MopsUnavailableError(
            f"MOPS table layout mismatch: header labels not found: no {_CODE_LABEL!r} header row"
        )
    return rows


def _parse_table(body: bytes) -> _ParsedTable:
    text = body.decode(_ENCODING, errors="replace")
    units = {unicodedata.normalize("NFKC", u).strip() for u in _UNIT_RE.findall(text)}
    if not units:
        raise MopsUnavailableError("MOPS table layout mismatch: no 單位 label found")
    unexpected = sorted(units - {_EXPECTED_UNIT})
    if unexpected:
        raise MopsUnavailableError(
            f"MOPS table unit is {unexpected[0]!r}, expected {_EXPECTED_UNIT!r}; refusing to "
            "report amounts under a different unit"
        )
    unit = _EXPECTED_UNIT

    report_date = None
    m = _ROC_DATE_RE.search(text)
    if m:
        report_date = f"{int(m.group(1)) + 1911:04d}-{m.group(2)}-{m.group(3)}"

    collector = _TableCollector()
    collector.feed(text)
    collector.close()
    return _ParsedTable(unit=unit, report_date=report_date, rows=_extract_rows(collector.tables))


def _norm(cell: str) -> str:
    """Fold full-width digits/punctuation to ASCII and drop thousands separators."""
    s = unicodedata.normalize("NFKC", cell).strip()
    s = s.replace(",", "").replace("−", "-").replace("–", "-")
    return s


def _parse_amount(cell: str) -> int | None:
    s = _norm(cell)
    if s in ("", "-", "--", "N/A", "NA"):
        return None
    if not re.fullmatch(r"-?\d+", s):
        raise MopsUnavailableError(f"MOPS amount not numeric: {cell!r}")
    return int(s)


def _parse_percent(cell: str) -> float | None:
    s = _norm(cell).rstrip("%")
    if s in ("", "-", "--", "N/A", "NA"):
        return None
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        raise MopsUnavailableError(f"MOPS percentage not numeric: {cell!r}")
    return float(s)


# --------------------------------------------------------------------------
# Point-in-time
# --------------------------------------------------------------------------

def _month_add(year: int, month: int, delta: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def _public_from(year: int, month: int) -> date:
    """Earliest date on which month ``(year, month)`` revenue is assumed public
    for a historical run: the 16th of the following month (deadline + buffer)."""
    ny, nm = _month_add(year, month, 1)
    return date(ny, nm, _HISTORICAL_PUBLIC_FROM_DAY)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def _fmt_amount(v: int | None) -> str:
    return "n/a" if v is None else f"{v:,}"


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.2f}%"


def get_monthly_revenue(ticker: str, curr_date: str, look_back_months: int = 12) -> str:
    """Official MOPS monthly revenue for a Taiwan-listed ticker, point-in-time.

    Returns a markdown report of up to ``look_back_months`` (1-24) months
    ending with the last calendar month before ``curr_date`` (the current month
    can never be complete). For a historical run (``curr_date`` before today)
    months whose statutory deadline had not safely passed are withheld and
    named.

    Raises :class:`NoMarketDataError` for non-Taiwan tickers (without any
    request) and when no month has a row for the company; raises
    :class:`MopsUnavailableError` on any transport, size, or layout problem,
    and ``ValueError`` for a malformed or future ``curr_date``.
    """
    code, board = split_taiwan_ticker(ticker, "MOPS monthly revenue")
    today = _now().date()   # Taiwan date, shared with the other Taiwan adapters
    as_of, live = resolve_as_of(curr_date, today)
    historical = not live
    look_back_months = min(max(1, int(look_back_months)), _MAX_LOOK_BACK_MONTHS)

    latest_year, latest_month = _month_add(as_of.year, as_of.month, -1)
    candidates = [_month_add(latest_year, latest_month, -i) for i in range(look_back_months)]

    withheld: list[str] = []
    months: list[tuple[int, int]] = []
    for year, month in candidates:
        if historical and _public_from(year, month) > as_of:
            withheld.append(f"{year:04d}-{month:02d}")
        else:
            months.append((year, month))

    unit = None
    report_dates: set[str] = set()
    lines: list[str] = []
    not_reported: list[str] = []
    for year, month in months:  # newest first
        table = _fetch_table(board, year, month)
        unit = unit or table.unit
        if table.report_date:
            report_dates.add(table.report_date)
        label = f"{year:04d}-{month:02d}"
        row = table.rows.get(code)
        if row is None:
            not_reported.append(label)
            continue
        lines.append(
            f"| {label} | {_fmt_amount(_parse_amount(row['revenue']))} "
            f"| {_fmt_amount(_parse_amount(row['prev_month']))} "
            f"| {_fmt_amount(_parse_amount(row['last_year']))} "
            f"| {_fmt_pct(_parse_percent(row['mom_pct']))} "
            f"| {_fmt_pct(_parse_percent(row['yoy_pct']))} "
            f"| {_fmt_amount(_parse_amount(row['ytd']))} "
            f"| {_fmt_amount(_parse_amount(row['ytd_last_year']))} "
            f"| {_fmt_pct(_parse_percent(row['ytd_yoy_pct']))} "
            f"| {row['name']} | {_norm(row.get('remark', '')) or '-'} |"
        )

    if not lines:
        detail = "no MOPS monthly revenue rows for company " + code
        if withheld and not months:
            detail = (
                f"every requested month is withheld for point-in-time safety as of "
                f"{curr_date} (withheld: {', '.join(withheld)})"
            )
        elif not_reported:
            detail += f" in {', '.join(not_reported)}"
        raise NoMarketDataError(ticker, canonical=code, detail=detail)

    header = [
        f"# Monthly Revenue (MOPS official filing) for {ticker.upper()} "
        f"(MOPS company code {code}, {_BOARD_LABEL[board]})",
        f"# Point-in-time as of: {curr_date}",
        f"# Source: MOPS 每月營業收入統計表, {_HOST}{_TABLE_PATH.format(board=board, roc_year='<ROC year>', month='<month>')}",
        f"# Unit: {unit} as stated by MOPS (千元 = TWD thousands). MoM / YoY / cumulative "
        f"YoY percentages are as published by MOPS, not recomputed.",
    ]
    if historical:
        header.append(
            f"# Point-in-time rule: this source carries no per-company announcement date. "
            f"Historical availability is inferred from the statutory deadline (revenue for a "
            f"month is due by the {_FILING_DEADLINE_DAY}th of the following month) plus a holiday "
            f"buffer: a month is served only from the {_HISTORICAL_PUBLIC_FROM_DAY}th of the "
            f"following month. Withheld as not yet public on {curr_date}: "
            f"{', '.join(withheld) if withheld else 'none'}."
        )
        header.append(
            "# Restatement limitation: MOPS regenerates past tables, so figures are as "
            "currently filed and may include corrections made after the original "
            "announcement; they are not guaranteed to be the as-first-reported values."
        )
    else:
        header.append(
            "# Live run: months are served exactly as currently filed on MOPS; "
            "the current calendar month is excluded as incomplete."
        )
    if not_reported:
        header.append(
            f"# No MOPS row for company {code} in: {', '.join(not_reported)} "
            f"(not yet filed, or not listed then). This is missing data, not zero revenue."
        )
    if report_dates:
        header.append(f"# MOPS table generation date(s): {', '.join(sorted(report_dates))}")
    header.append(
        "# Monthly revenue is unaudited operating evidence; it is not a financial statement "
        "and says nothing about margins or EPS."
    )

    table_header = (
        f"| Month | Revenue ({unit}) | Prior month | Same month last year | MoM % | YoY % "
        f"| YTD cumulative | YTD last year | Cumulative YoY % | Company | Remark |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    )
    return "\n".join(header) + "\n\n" + table_header + "\n" + "\n".join(lines)


def clear_cache() -> None:
    """Drop cached MOPS tables (tests, or a long-lived process)."""
    with _cache_lock:
        _table_cache.clear()
