"""MOPS monthly revenue adapter (Taiwan Adapter step 4a).

Everything is network-mocked at ``mops._http_get``. The default fixture layout
copies the live ``t21sc03`` page structure: Big5 (cp950) bytes, nested wrapper
tables, an industry table with ``單位：千元``, a data table whose header is a
group row (``colspan`` groups plus ``備註`` with ``rowspan=2``) over leaf labels
split by ``<br>`` (``公司<br>代號``), company rows with thousands separators and
a capitalised ``<Td>`` cell, a ``合計`` row, and the whole-market total table
whose ``備註`` header is commented out. A ``flat`` layout (one header row, any
column order) exercises the header-label column mapping.
"""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest import mock

import pytest
import requests
from langchain_core.messages import AIMessage

from tradingagents.dataflows import bounded_http, interface, mops, taiwan_common
from tradingagents.dataflows.errors import NoMarketDataError

_TODAY = "2026-09-15"


def _taipei_noon(iso_date):
    """A Taiwan-time "now" on ``iso_date`` for the adapter's clock seam."""
    return datetime.fromisoformat(f"{iso_date} 12:00:00").replace(tzinfo=taiwan_common.TAIPEI)
_REAL_HTTP_GET = mops._http_get  # setUp replaces mops._http_get with a stub


class _FakeResponse:
    """Minimal ``requests.Response`` stand-in for exercising the real _http_get."""

    def __init__(self, status_code=200, body=b"", headers=None, chunk=64 * 1024):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._chunk = chunk
        self.bytes_served = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, size):
        for i in range(0, len(self._body), self._chunk):
            piece = self._body[i:i + self._chunk]
            self.bytes_served += len(piece)
            yield piece


# Leaf header labels in live order, as MOPS prints them (``<br>`` splits kept).
_LIVE_LABELS = ("公司代號", "公司名稱", "當月營收", "上月營收", "去年當月營收",
                "上月比較增減(%)", "去年同月增減(%)", "當月累計營收", "去年累計營收",
                "前期比較增減(%)", "備註")
_LIVE_LABEL_HTML = {
    "公司代號": "公司<br>代號",
    "上月比較增減(%)": "上月比較<br>增減(%)",
    "去年同月增減(%)": "去年同月<br>增減(%)",
    "前期比較增減(%)": "前期比較<br>增減(%)",
}


def _row(code, name, cur, prev, ly, mom, yoy, ytd, ytd_ly, ytd_yoy, remark="-"):
    """One company row, keyed by leaf header label (rendered per column order)."""
    return dict(zip(_LIVE_LABELS, (code, name, cur, prev, ly, mom, yoy, ytd, ytd_ly, ytd_yoy, remark), strict=True))


def _cells(row, labels):
    out = []
    for raw_label in labels:
        # Look values up by the normalised label so full-width header variants
        # still render the same row; unknown labels get "x".
        label = taiwan_common.norm_header(raw_label)
        value = row.get(label, "x")
        if label == "公司代號":
            out.append(f"<td align=center>{value}</td>")
        elif label in ("公司名稱", "備註"):
            out.append(f"<td align=left>{value}</td>")
        elif label == "去年同月增減(%)":
            out.append(f"<Td nowrap>{value}</Td>")  # the live page capitalises this one
        else:
            out.append(f"<td nowrap>{value}</td>")
    return "<tr align=right>" + "".join(out) + "</tr>"


def _live_data_table(rows, headers):
    group = ("<tr><th class=tt colspan=2>&nbsp;</th><th class=tt colspan=5 >營業收入</th>"
             "<th class=tt colspan=3 >累計營業收入</th><th rowspan=2 class=tt>備註</th></tr>")
    leaf = "<tr>" + "".join(
        f"<th class=tt>{_LIVE_LABEL_HTML.get(label, label)}</th>" for label in _LIVE_LABELS[:-1]
    ) + "</tr>"
    total = ("<tr align=right><th class=tt nowrap colspan=2 align=center>合計</th>"
             + "<td nowrap>1</td>" * 8 + "<td>&nbsp;</td></tr>")
    return ("<table  width=100% border=5 bordercolor='#FF6600' bgcolor='#FFFFFF'>"
            + group + (leaf if headers else "")
            + "".join(_cells(r, _LIVE_LABELS) for r in rows) + total + "</table>")


def _flat_data_table(rows, labels, header_attrs):
    head = "<tr>" + "".join(
        f"<th class=tt {header_attrs.get(label, '')}>{label}</th>" for label in labels
    ) + "</tr>"
    return "<table border=5>" + head + "".join(_cells(r, labels) for r in rows) + "</table>"


def _table(rows, title="上市公司115年8月份(累計與當月)營業收入統計表", unit="千元",
           report_date="115/09/15", headers=True, columns=None, header_attrs=None):
    """Big5 page bytes. ``columns=None`` renders the live layout; a label tuple
    renders a flat single-header table in that order (labels may repeat, and
    unknown labels get the cell text ``x``)."""
    if columns is None:
        data = _live_data_table(rows, headers)
    else:
        data = _flat_data_table(rows, columns, header_attrs or {})
    unit_th = f"<th class=tt align=right >單位：{unit}</th>" if unit else ""
    grand_total = (  # whole-market total: no company rows, 備註 commented out
        "<table border=0 width=100%><tr><td colspan=2><table width=100% border=5>"
        "<tr><th class=tt colspan=2>&nbsp;</th><th class=tt colspan=5 >營業收入</th>"
        "<th class=tt colspan=3 >累計營業收入</th><!-- <th rowspan=2 class=tt>備註</th> --></tr>"
        + "<tr>" + "".join(f"<th class=tt>{_LIVE_LABEL_HTML.get(x, x)}</th>" for x in _LIVE_LABELS[:-1])
        + "</tr><tr align=right><th class=tt nowrap colspan=2 align=center>合計</th>"
        + "<td nowrap>9</td>" * 8 + "</tr></table></td></tr></table>"
    )
    html = (
        "<html><head><meta http-equiv='Content-Type' content='text/html;charset=big5'></head>"
        f"<body><center><font size='5'><b>{title}</b></font>"
        f"<div class=tt>出表日期：{report_date}</div>"
        "<table border=0 width=100%><tr><td><br><table border=0 width=100%>"
        f"<tr><th class=tt align=left >產業別：半導體業</th>{unit_th}</tr>"
        f"<tr><td colspan=2>{data}</td></tr></table></td></tr></table><br>"
        f"{grand_total}</center></body></html>"
    )
    return html.encode("cp950", errors="strict")


# One realistic set of rows (values loosely modelled on the live 2026-08 file).
_TSMC = _row("2330", "台積電", "514,805,337", "467,580,548", "335,771,691",
             "10.09", "53.32", "3,386,869,575", "2,431,982,931", "39.26",
             "因先進製程需求增加所致。")
_NEG = _row("1101", "台泥", "13,515,534", "13,744,103", "12,214,776",
            "-1.66", "10.65", "98,726,969", "96,131,621", "2.69")
_BLANK = _row("9999", "新掛牌", "1,234", "--", "", "--", "", "1,234", "", "--")
_NEG_AMOUNT = _row("8888", "退貨月", "-500", "1,000", "800", "-150.00", "-162.50",
                   "500", "800", "-37.50")
_GLOBAL_ROW = _row("6488", "環球晶", "5,000,000", "4,800,000", "5,500,000",
                   "4.17", "-9.09", "40,000,000", "44,000,000", "-9.09")


class _Server:
    """Stateful stand-in for ``_http_get``: url -> bytes or exception."""

    def __init__(self):
        self.pages: dict[str, bytes | Exception] = {}
        self.requests: list[str] = []

    def add(self, board, year, month, body):
        self.pages[mops._table_url(board, year, month)] = body

    def __call__(self, url, timeout=bounded_http.DEFAULT_TIMEOUT_SECONDS):
        self.requests.append(url)
        page = self.pages.get(url)
        if page is None:
            raise mops.MopsUnavailableError(f"MOPS fetch failed (HTTP 404) for {url}")
        if isinstance(page, Exception):
            raise page
        return page


@pytest.mark.unit
class _Base(unittest.TestCase):
    def setUp(self):
        mops.clear_cache()
        self.server = _Server()
        self._patches = [
            mock.patch.object(mops, "_http_get", self.server),
            mock.patch.object(mops, "_now", lambda: _taipei_noon(_TODAY)),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        mops.clear_cache()


# --- ticker handling ---------------------------------------------------------

class TickerTests(_Base):
    def test_tw_and_two_map_to_company_code_and_board(self):
        self.assertEqual(taiwan_common.split_taiwan_ticker("2330.TW", "MOPS monthly revenue"), ("2330", "sii"))
        self.assertEqual(taiwan_common.split_taiwan_ticker("2330.tw", "MOPS monthly revenue"), ("2330", "sii"))
        self.assertEqual(taiwan_common.split_taiwan_ticker("6488.TWO", "MOPS monthly revenue"), ("6488", "otc"))
        self.assertEqual(taiwan_common.split_taiwan_ticker("5274.two", "MOPS monthly revenue"), ("5274", "otc"))

    def test_non_taiwan_tickers_are_refused_without_a_request(self):
        for ticker in ("AAPL", "7203.T", "0700.HK", "SHOP.TO", "BRK.B", ""):
            with self.subTest(ticker=ticker):
                with self.assertRaises(NoMarketDataError) as ctx:
                    mops.get_monthly_revenue(ticker, "2026-09-15")
                self.assertIn("not queried", str(ctx.exception))
        self.assertEqual(self.server.requests, [])

    def test_requests_go_to_the_right_board_directory(self):
        self.server.add("sii", 2026, 8, _table([_TSMC]))
        mops.get_monthly_revenue("2330.TW", "2026-09-15", look_back_months=1)
        self.server.add("otc", 2026, 8, _table([_GLOBAL_ROW]))
        mops.get_monthly_revenue("6488.TWO", "2026-09-15", look_back_months=1)
        self.assertEqual(self.server.requests, [
            "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_115_8_0.html",
            "https://mopsov.twse.com.tw/nas/t21/otc/t21sc03_115_8_0.html",
        ])


# --- parser ------------------------------------------------------------------

class ParserTests(_Base):
    def test_positive_growth_row_numbers_land_in_the_right_columns(self):
        self.server.add("sii", 2026, 8, _table([_NEG, _TSMC]))
        out = mops.get_monthly_revenue("2330.TW", "2026-09-15", look_back_months=1)
        self.assertIn(
            "| 2026-08 | 514,805,337 | 467,580,548 | 335,771,691 | +10.09% | +53.32% "
            "| 3,386,869,575 | 2,431,982,931 | +39.26% | 台積電 | 因先進製程需求增加所致。 |",
            out,
        )
        self.assertIn("Unit: 千元 as stated by MOPS", out)
        self.assertIn("MOPS table generation date(s): 2026-09-15", out)
        self.assertNotIn("<td", out)  # never raw HTML

    def test_negative_yoy_and_mom_keep_their_sign(self):
        self.server.add("sii", 2026, 8, _table([_NEG]))
        out = mops.get_monthly_revenue("1101.TW", "2026-09-15", look_back_months=1)
        self.assertIn("| 2026-08 | 13,515,534 | 13,744,103 | 12,214,776 | -1.66% | +10.65% ", out)

    def test_blank_and_dash_cells_become_na_not_zero(self):
        self.server.add("sii", 2026, 8, _table([_BLANK]))
        out = mops.get_monthly_revenue("9999.TW", "2026-09-15", look_back_months=1)
        self.assertIn("| 2026-08 | 1,234 | n/a | n/a | n/a | n/a | 1,234 | n/a | n/a | 新掛牌 | - |", out)
        self.assertNotIn("| 0 |", out)

    def test_negative_amount_and_fullwidth_input_parse(self):
        self.server.add("sii", 2026, 8, _table([_NEG_AMOUNT]))
        out = mops.get_monthly_revenue("8888.TW", "2026-09-15", look_back_months=1)
        self.assertIn("| 2026-08 | -500 | 1,000 | 800 | -150.00% | -162.50% | 500 | 800 | -37.50% |", out)
        self.assertEqual(mops._parse_amount("１，２３４"), 1234)
        self.assertEqual(mops._parse_amount("－５"), -5)
        self.assertEqual(mops._parse_percent("－１２.５"), -12.5)
        self.assertIsNone(mops._parse_percent("--"))
        self.assertIsNone(mops._parse_amount(""))

    def test_non_numeric_cell_fails_closed(self):
        with self.assertRaises(mops.MopsUnavailableError):
            mops._parse_amount("12O")  # letter O
        with self.assertRaises(mops.MopsUnavailableError):
            mops._parse_percent("n.a.")

    def test_percentages_are_as_published_not_recomputed(self):
        # Source MoM says 10.09 even though (514805337/467580548 - 1) = 10.10%.
        self.server.add("sii", 2026, 8, _table([_TSMC]))
        out = mops.get_monthly_revenue("2330.TW", "2026-09-15", look_back_months=1)
        self.assertIn("+10.09%", out)
        self.assertNotIn("+10.10%", out)


# --- header-label column mapping -------------------------------------------------

_REORDERED = ("去年當月營收", "公司名稱", "公司代號", "前期比較增減(%)", "備註",
              "當月累計營收", "上月比較增減(%)", "去年累計營收", "去年同月增減(%)",
              "上月營收", "當月營收")
_REQUIRED_LABELS = _LIVE_LABELS[:-1]


class HeaderMappingTests(_Base):
    _ROWS = [_NEG, _TSMC, _BLANK, _NEG_AMOUNT]

    def _parse(self, **kw):
        return mops._parse_table(_table(self._ROWS, **kw)).rows

    def _fails(self, msg_part, **kw):
        with self.assertRaises(mops.MopsUnavailableError) as ctx:
            mops._parse_table(_table(self._ROWS, **kw))
        self.assertIn(msg_part, str(ctx.exception))
        return str(ctx.exception)

    def test_live_layout_maps_every_field_including_rowspan_remark(self):
        rows = self._parse()
        self.assertEqual(set(rows), {"1101", "2330", "9999", "8888"})  # 合計 rows skipped
        self.assertEqual(rows["2330"], {
            "code": "2330", "name": "台積電", "revenue": "514,805,337",
            "prev_month": "467,580,548", "last_year": "335,771,691", "mom_pct": "10.09",
            "yoy_pct": "53.32", "ytd": "3,386,869,575", "ytd_last_year": "2,431,982,931",
            "ytd_yoy_pct": "39.26", "remark": "因先進製程需求增加所致。",
        })

    def test_reordered_columns_parse_identically(self):
        self.assertEqual(self._parse(columns=_REORDERED), self._parse())
        # ...and the rendered tool output is byte-for-byte the same.
        outputs = []
        for columns in (None, _REORDERED):
            mops.clear_cache()
            self.server.add("sii", 2026, 8, _table(self._ROWS, columns=columns))
            outputs.append(mops.get_monthly_revenue("2330.TW", _TODAY, look_back_months=1))
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn("| 2026-08 | 514,805,337 | 467,580,548 | 335,771,691 | +10.09% | +53.32% ",
                      outputs[1])

    def test_unknown_extra_columns_in_the_middle_are_ignored(self):
        columns = ("公司代號", "公司名稱", "市場別", "當月營收", "上月營收", "營收(百萬元)",
                   "去年當月營收", "上月比較增減(%)", "去年同月增減(%)", "產業代號",
                   "當月累計營收", "去年累計營收", "前期比較增減(%)", "備註")
        self.assertEqual(self._parse(columns=columns), self._parse())

    def test_missing_required_header_fails_closed(self):
        for label in _REQUIRED_LABELS:
            with self.subTest(label=label):
                columns = tuple(x for x in _LIVE_LABELS if x != label)
                msg = self._fails("header labels not found", columns=columns)
                self.assertIn(label, msg)

    def test_duplicate_required_header_fails_closed(self):
        for label in ("當月營收", "去年同月增減(%)", "公司名稱"):
            with self.subTest(label=label):
                columns = _LIVE_LABELS + (label,)
                self._fails(f"{label!r} appears 2 times", columns=columns)

    def test_duplicate_optional_remark_header_fails_closed(self):
        self._fails("'備註' appears 2 times", columns=_LIVE_LABELS + ("備註",))

    def test_required_header_spanning_two_columns_fails_closed(self):
        columns = ("公司代號", "公司名稱", "當月營收", "上月營收", "去年當月營收",
                   "上月比較增減(%)", "去年同月增減(%)", "當月累計營收", "去年累計營收",
                   "前期比較增減(%)")
        self._fails("spans columns", columns=columns, header_attrs={"當月營收": "colspan=2"})

    def test_remark_column_is_optional(self):
        columns = _REQUIRED_LABELS  # no 備註, like the live whole-market total table
        rows = self._parse(columns=columns)
        self.assertNotIn("remark", rows["2330"])
        mops.clear_cache()
        self.server.add("sii", 2026, 8, _table([_TSMC], columns=columns))
        out = mops.get_monthly_revenue("2330.TW", _TODAY, look_back_months=1)
        self.assertIn("| +39.26% | 台積電 | - |", out)

    def test_no_leaf_header_row_fails_closed(self):
        self._fails("no '公司代號' header row", headers=False)

    def test_company_row_with_a_missing_cell_fails_closed(self):
        # Dropping one cell would shift 備註 under 前期比較增減(%) and leave the
        # optional 備註 empty; the width check refuses instead of mis-assigning.
        for columns in (None, _LIVE_LABELS):
            with self.subTest(layout="live" if columns is None else "flat"):
                page = _table([_TSMC], columns=columns).replace(
                    "<td nowrap>39.26</td>".encode("cp950"), b"", 1)
                with self.assertRaises(mops.MopsUnavailableError) as ctx:
                    mops._parse_table(page)
                self.assertIn("row for company 2330 spans 10 columns but its header spans 11",
                              str(ctx.exception))

    def test_company_row_with_an_extra_cell_fails_closed(self):
        page = _table([_TSMC]).replace(
            "<td nowrap>39.26</td>".encode("cp950"), b"<td nowrap>39.26</td><td>extra</td>", 1)
        with self.assertRaises(mops.MopsUnavailableError) as ctx:
            mops._parse_table(page)
        self.assertIn("spans 12 columns but its header spans 11", str(ctx.exception))

    def test_header_normalization_is_exact_not_fuzzy(self):
        # full-width punctuation / spaces and <br> splits normalise to the live label...
        self.assertEqual(taiwan_common.norm_header("前期比較<br>增減（％）".replace("<br>", "")), "前期比較增減(%)")
        self.assertEqual(taiwan_common.norm_header(" 公司\u3000代號\n"), "公司代號")
        self.assertEqual(taiwan_common.norm_header("上月比較\xa0增減(%)"), "上月比較增減(%)")
        fullwidth = tuple(
            {"公司代號": "公司　代號", "上月比較增減(%)": "上月比較增減（％）"}.get(x, x)
            for x in _LIVE_LABELS
        )
        self.assertEqual(self._parse(columns=fullwidth), self._parse(columns=_LIVE_LABELS))
        # ...but a merely similar label is not accepted as the required one.
        near_miss = tuple("當月營收(千元)" if x == "當月營收" else x for x in _LIVE_LABELS)
        self._fails("'當月營收'", columns=near_miss)


# --- point-in-time -------------------------------------------------------------

class PointInTimeTests(_Base):
    def _serve_three_months(self):
        for month, cur in ((7, "260,000,000"), (8, "280,000,000"), (9, "300,000,000")):
            self.server.add("sii", 2026, month, _table([
                _row("2330", "台積電", cur, "250,000,000", "200,000,000", "4.00", "30.00",
                     "2,000,000,000", "1,500,000,000", "33.33")
            ]))

    def test_historical_run_never_returns_later_months(self):
        # MOPS today holds 07, 08 and 09; a run as of 2026-08-31 may only see
        # July (filed by 08-10). August is still in progress on 08-31, and
        # September lies in the future, so neither is requested or returned.
        self._serve_three_months()
        out = mops.get_monthly_revenue("2330.TW", "2026-08-31", look_back_months=1)
        self.assertIn("| 2026-07 | 260,000,000 |", out)
        self.assertNotIn("2026-08 |", out)
        self.assertNotIn("280,000,000", out)
        self.assertNotIn("300,000,000", out)
        self.assertIn("carries no per-company announcement date", out)
        self.assertIn("Restatement limitation", out)
        self.assertIn("not guaranteed to be the as-first-reported values", out)
        self.assertEqual(self.server.requests, [mops._table_url("sii", 2026, 7)])

    def test_historical_run_withholds_month_before_its_deadline_buffer(self):
        # As of 2026-08-12 July revenue was due (08-10) but inside the holiday
        # buffer, so it is withheld and named; June is served.
        self._serve_three_months()
        self.server.add("sii", 2026, 6, _table([
            _row("2330", "台積電", "240,000,000", "230,000,000", "190,000,000", "4.35", "26.32",
                 "1,700,000,000", "1,300,000,000", "30.77")
        ]))
        out = mops.get_monthly_revenue("2330.TW", "2026-08-12", look_back_months=2)
        self.assertIn("| 2026-06 | 240,000,000 |", out)
        self.assertNotIn("260,000,000", out)
        self.assertIn("Withheld as not yet public on 2026-08-12: 2026-07", out)
        self.assertEqual(self.server.requests, [mops._table_url("sii", 2026, 6)])

    def test_historical_run_before_buffer_day_still_withholds(self):
        self._serve_three_months()
        out = mops.get_monthly_revenue("2330.TW", "2026-09-12", look_back_months=2)
        self.assertNotIn("280,000,000", out)   # 08 revenue only served from 09-16
        self.assertIn("| 2026-07 | 260,000,000 |", out)
        self.assertIn("Withheld as not yet public on 2026-09-12: 2026-08", out)

    def test_historical_run_after_buffer_day_serves_the_month(self):
        # Pretend today is later so 2026-09-16 counts as historical.
        with mock.patch.object(mops, "_now", lambda: _taipei_noon("2026-12-01")):
            self._serve_three_months()
            out = mops.get_monthly_revenue("2330.TW", "2026-09-16", look_back_months=2)
        self.assertIn("| 2026-08 | 280,000,000 |", out)
        self.assertIn("| 2026-07 | 260,000,000 |", out)
        self.assertIn("Withheld as not yet public on 2026-09-16: none", out)

    def test_every_month_withheld_is_no_data_not_empty_table(self):
        self._serve_three_months()
        with self.assertRaises(NoMarketDataError) as ctx:
            mops.get_monthly_revenue("2330.TW", "2026-09-12", look_back_months=1)
        self.assertIn("withheld for point-in-time safety", str(ctx.exception))
        self.assertEqual(self.server.requests, [])

    def test_live_run_serves_filed_months_and_flags_unfiled(self):
        # Live (curr_date == today): August filed, July filed; no deadline rule.
        self.server.add("sii", 2026, 8, _table([_TSMC]))
        self.server.add("sii", 2026, 7, _table([]))   # table exists, 2330 not in it
        out = mops.get_monthly_revenue("2330.TW", _TODAY, look_back_months=2)
        self.assertIn("| 2026-08 | 514,805,337 |", out)
        self.assertIn("No MOPS row for company 2330 in: 2026-07", out)
        self.assertIn("missing data, not zero revenue", out)
        self.assertIn("Live run", out)
        self.assertNotIn("Restatement limitation", out)


# --- network / layout failures ---------------------------------------------------

class FailureTests(_Base):
    def _expect_unavailable(self, exc_or_body, msg_part):
        self.server.pages[mops._table_url("sii", 2026, 8)] = exc_or_body
        with self.assertRaises(mops.MopsUnavailableError) as ctx:
            mops.get_monthly_revenue("2330.TW", "2026-09-15", look_back_months=1)
        self.assertIn(msg_part, str(ctx.exception))

    def test_timeout_is_vendor_failure(self):
        self._expect_unavailable(mops.MopsUnavailableError("MOPS fetch failed (TimeoutError)"),
                                 "TimeoutError")

    def test_http_error_is_vendor_failure(self):
        self._expect_unavailable(mops.MopsUnavailableError("MOPS fetch failed (HTTP 503)"), "503")

    def test_missing_month_file_is_vendor_failure(self):
        # Nothing registered -> the stand-in raises the same 404 path _http_get uses.
        with self.assertRaises(mops.MopsUnavailableError) as ctx:
            mops.get_monthly_revenue("2330.TW", "2026-09-15", look_back_months=1)
        self.assertIn("404", str(ctx.exception))

    def test_layout_mismatch_missing_headers_fails_closed(self):
        self._expect_unavailable(_table([_TSMC], headers=False), "header labels not found")

    def test_layout_mismatch_missing_unit_fails_closed(self):
        self._expect_unavailable(_table([_TSMC], unit=""), "no 單位 label")

    def test_empty_table_is_no_data_not_zero(self):
        self.server.add("sii", 2026, 8, _table([]))
        with self.assertRaises(NoMarketDataError) as ctx:
            mops.get_monthly_revenue("2330.TW", "2026-09-15", look_back_months=1)
        self.assertIn("no MOPS monthly revenue rows for company 2330", str(ctx.exception))

    def _real_get(self, **fake):
        resp = _FakeResponse(**fake)
        with mock.patch.object(bounded_http.requests, "get", return_value=resp), \
                self.assertRaises(mops.MopsUnavailableError) as ctx:
            _REAL_HTTP_GET("https://mopsov.twse.com.tw/x")
        return str(ctx.exception), resp

    def test_oversized_streamed_body_is_abandoned_at_the_cap(self):
        big = b"x" * (bounded_http.DEFAULT_MAX_BODY_BYTES * 3)
        msg, resp = self._real_get(body=big)
        self.assertIn("exceeded", msg)
        # Stopped shortly after the cap instead of downloading all 12 MB.
        self.assertLessEqual(resp.bytes_served, bounded_http.DEFAULT_MAX_BODY_BYTES + bounded_http.CHUNK_BYTES)

    def test_oversized_declared_length_is_refused_without_reading(self):
        msg, resp = self._real_get(
            body=b"<html></html>", headers={"Content-Length": str(bounded_http.DEFAULT_MAX_BODY_BYTES + 1)}
        )
        self.assertIn("exceeded", msg)
        self.assertEqual(resp.bytes_served, 0)

    def test_non_200_status_is_vendor_failure(self):
        for status in (202, 404, 500, 503):
            with self.subTest(status=status):
                msg, _ = self._real_get(status_code=status, body=b"<html>error</html>")
                self.assertIn(f"HTTP {status}", msg)

    def test_transport_errors_are_vendor_failures(self):
        cases = (
            (requests.Timeout("slow"), "Timeout"),
            (requests.ConnectionError("reset"), "ConnectionError"),
            (requests.exceptions.SSLError("bad cert"), "SSLError"),
        )
        for exc, name in cases:
            with self.subTest(exc=name), \
                    mock.patch.object(bounded_http.requests, "get", side_effect=exc), \
                    self.assertRaises(mops.MopsUnavailableError) as ctx:
                _REAL_HTTP_GET("https://mopsov.twse.com.tw/x")
            self.assertIn(name, str(ctx.exception))

    def test_request_is_bounded_and_identified(self):
        resp = _FakeResponse(body=b"ok")
        with mock.patch.object(bounded_http.requests, "get", return_value=resp) as get:
            self.assertEqual(_REAL_HTTP_GET("https://mopsov.twse.com.tw/x"), b"ok")
        kwargs = get.call_args.kwargs
        self.assertEqual(kwargs["timeout"], bounded_http.DEFAULT_TIMEOUT_SECONDS)
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["headers"]["User-Agent"], bounded_http.DEFAULT_USER_AGENT)
        self.assertNotIn("verify", kwargs)  # TLS verification stays at the default (on)


# --- router integration ----------------------------------------------------------

class RouterTests(_Base):
    def test_registered_as_optional_mops_only_category(self):
        self.assertEqual(interface.get_category_for_method("get_monthly_revenue"), "taiwan_market_data")
        self.assertIn("taiwan_market_data", interface.OPTIONAL_CATEGORIES)
        self.assertEqual(list(interface.VENDOR_METHODS["get_monthly_revenue"]), ["mops"])
        self.assertIn("mops", interface.VENDOR_LIST)

    def test_vendor_failure_degrades_to_unavailable_sentinel(self):
        out = interface.route_to_vendor("get_monthly_revenue", "2330.TW", "2026-09-15", 1)
        self.assertTrue(out.startswith("DATA_UNAVAILABLE:"), out)
        self.assertIn("do not fabricate", out)
        self.assertNotIn("| 0 |", out)

    def test_non_taiwan_ticker_yields_no_data_sentinel_without_request(self):
        out = interface.route_to_vendor("get_monthly_revenue", "AAPL", "2026-09-15", 12)
        self.assertTrue(out.startswith("NO_DATA_AVAILABLE:"), out)
        self.assertIn("not queried", out)
        self.assertEqual(self.server.requests, [])

    def test_success_passes_through(self):
        self.server.add("sii", 2026, 8, _table([_TSMC]))
        out = interface.route_to_vendor("get_monthly_revenue", "2330.TW", "2026-09-15", 1)
        self.assertIn("| 2026-08 | 514,805,337 |", out)


# --- fundamentals analyst wiring ---------------------------------------------------

def _capturing_llm(captured):
    def bind_tools(tools):
        captured["tools"] = [t.name for t in tools]
        def run(prompt_value):
            captured["prompt"] = prompt_value.to_string()
            return AIMessage(content="report")
        return run
    llm = mock.MagicMock()
    llm.bind_tools.side_effect = bind_tools
    return llm


def _run_analyst(ticker):
    from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
    captured = {}
    out = create_fundamentals_analyst(_capturing_llm(captured))({
        "company_of_interest": ticker, "trade_date": "2026-09-15",
        "asset_type": "stock", "messages": [],
    })
    assert out["fundamentals_report"] == "report"
    return captured


@pytest.mark.unit
class FundamentalsAnalystWiringTests(unittest.TestCase):
    def test_taiwan_ticker_gets_monthly_revenue_tool_and_guidance(self):
        for ticker in ("2330.TW", "6488.TWO"):
            with self.subTest(ticker=ticker):
                c = _run_analyst(ticker)
                self.assertEqual(c["tools"], ["get_fundamentals", "get_balance_sheet",
                                              "get_cashflow", "get_income_statement",
                                              "get_monthly_revenue"])
                self.assertIn("get_monthly_revenue", c["prompt"])
                self.assertIn("not as a substitute for audited financial statements", c["prompt"])
                self.assertIn("do not infer EPS growth", c["prompt"])
                self.assertIn("single month's change", c["prompt"])

    def test_default_market_is_unchanged(self):
        for ticker in ("AAPL", "NVDA", "7203.T"):
            with self.subTest(ticker=ticker):
                c = _run_analyst(ticker)
                self.assertEqual(c["tools"], ["get_fundamentals", "get_balance_sheet",
                                              "get_cashflow", "get_income_statement"])
                self.assertNotIn("monthly revenue", c["prompt"].lower())
                self.assertNotIn("MOPS", c["prompt"])

    def test_fundamentals_tool_node_can_execute_the_tool(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        nodes = TradingAgentsGraph._create_tool_nodes(None)
        self.assertIn("get_monthly_revenue", nodes["fundamentals"].tools_by_name)
        # Existing statement tools still there: monthly revenue is additive.
        for name in ("get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"):
            self.assertIn(name, nodes["fundamentals"].tools_by_name)

    def test_yfinance_fundamentals_routing_untouched_for_taiwan_and_us(self):
        # The new category adds a vendor; it must not change where the statement
        # tools route for any market.
        for method in ("get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"):
            self.assertEqual(interface.get_category_for_method(method), "fundamental_data")
            self.assertIn("yfinance", interface.VENDOR_METHODS[method])
            self.assertNotIn("mops", interface.VENDOR_METHODS[method])
        from tradingagents.default_config import DEFAULT_CONFIG
        self.assertEqual(DEFAULT_CONFIG["data_vendors"]["fundamental_data"], "yfinance")
        self.assertEqual(DEFAULT_CONFIG["data_vendors"]["taiwan_market_data"], "mops")


# --- hardening: look-back cap, future dates, cache of still-filling months -----------------

class HardeningTests(_Base):
    def test_look_back_months_is_capped(self):
        with mock.patch.object(mops, "_http_get", lambda url: self.server.requests.append(url)
                               or _table([_TSMC])):
            mops.get_monthly_revenue("2330.TW", _TODAY, look_back_months=500)
        self.assertEqual(len(self.server.requests), mops._MAX_LOOK_BACK_MONTHS)
        self.assertEqual(mops._MAX_LOOK_BACK_MONTHS, 24)

    def test_future_curr_date_is_refused_without_requests(self):
        with self.assertRaises(ValueError) as ctx:
            mops.get_monthly_revenue("2330.TW", "2026-10-01", look_back_months=1)
        self.assertIn("future dates are not supported", str(ctx.exception))
        self.assertEqual(self.server.requests, [])
        out = interface.route_to_vendor("get_monthly_revenue", "2330.TW", "2026-10-01", 1)
        self.assertTrue(out.startswith("DATA_UNAVAILABLE:"), out)
        self.assertIn("future dates are not supported", out)

    def test_month_still_receiving_filings_is_never_cached(self):
        # On 09-03 August is still filling in: 2330 has not filed yet.
        self.server.add("sii", 2026, 8, _table([_NEG]))
        with mock.patch.object(mops, "_now", lambda: _taipei_noon("2026-09-03")), \
                self.assertRaises(NoMarketDataError):
            mops.get_monthly_revenue("2330.TW", "2026-09-03", look_back_months=1)
        # By 09-10 it has; the same process must see the new filing.
        self.server.add("sii", 2026, 8, _table([_NEG, _TSMC]))
        with mock.patch.object(mops, "_now", lambda: _taipei_noon("2026-09-10")):
            out = mops.get_monthly_revenue("2330.TW", "2026-09-10", look_back_months=1)
        self.assertIn("| 2026-08 | 514,805,337 |", out)
        self.assertEqual(len(self.server.requests), 2)

    def test_complete_month_is_cached_until_ttl(self):
        clock = [0.0]
        self.server.add("sii", 2026, 7, _table([_TSMC]))
        with mock.patch.object(mops, "_monotonic", lambda: clock[0]):
            for _ in range(2):   # July is past its 08-16 safety date on 09-15
                mops.get_monthly_revenue("2330.TW", "2026-08-31", look_back_months=1)
            self.assertEqual(len(self.server.requests), 1)
            clock[0] += mops._COMPLETE_TABLE_TTL_SECONDS + 1
            mops.get_monthly_revenue("2330.TW", "2026-08-31", look_back_months=1)
        self.assertEqual(len(self.server.requests), 2)

    def test_cache_is_bounded(self):
        with mock.patch.object(mops, "_TABLE_CACHE_MAX", 2), \
                mock.patch.object(mops, "_http_get", lambda url: _table([_TSMC])):
            mops.get_monthly_revenue("2330.TW", "2026-08-31", look_back_months=4)
        self.assertEqual(len(mops._table_cache), 2)
