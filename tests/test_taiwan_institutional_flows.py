"""TWSE/TPEx institutional investor flows (Taiwan Adapter step 4c).

Everything is mocked at ``flows.bounded_request`` (real JSON decoding and CSV
parsing still run), with the clock at ``flows._now`` and pacing at
``flows._monotonic`` / ``flows._sleep``. Fixtures copy the live formats: TWSE
T86 JSON (``stat``, ``date``, ``hints``, 19 ``fields``) and the TPEx dailyTrade
Big5 CSV (title line with the ROC date, group-qualified 24-column header,
single-cell notes, ``BIGD_<ROC date>.csv`` filename), plus the JSON body TPEx
returns for a CSV request on a day without a table.
"""

from __future__ import annotations

import json
import threading
import time
import unittest
from datetime import date, datetime
from unittest import mock

import pytest
from langchain_core.messages import AIMessage
from requests.structures import CaseInsensitiveDict

from tradingagents.dataflows import interface, taiwan_common, taiwan_institutional_flows as flows
from tradingagents.dataflows.bounded_http import BoundedResponse
from tradingagents.dataflows.errors import NoMarketDataError

TWSE_FIELDS = [
    "證券代號", "證券名稱",
    "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
    "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
    "投信買進股數", "投信賣出股數", "投信買賣超股數",
    "自營商買賣超股數",
    "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)",
    "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)",
    "三大法人買賣超股數",
]
TPEX_HEADER = [
    "代號", "名稱",
    "外資及陸資(不含外資自營商)-買進股數", "外資及陸資(不含外資自營商)-賣出股數", "外資及陸資(不含外資自營商)-買賣超股數",
    "外資自營商-買進股數", "外資自營商-賣出股數", "外資自營商-買賣超股數",
    "外資及陸資-買進股數", "外資及陸資-賣出股數", "外資及陸資-買賣超股數",
    "投信-買進股數", "投信-賣出股數", "投信-買賣超股數",
    "自營商(自行買賣)-買進股數", "自營商(自行買賣)-賣出股數", "自營商(自行買賣)-買賣超股數",
    "自營商(避險)-買進股數", "自營商(避險)-賣出股數", "自營商(避險)-買賣超股數",
    "自營商-買進股數", "自營商-賣出股數", "自營商-買賣超股數",
    "三大法人買賣超股數合計",
]
# Live rows from the research cross-check (2026-09-14).
TSMC_0914 = ["2330", "台積電          ", "5,128,006", "16,319,028", "-11,191,022", "0", "0", "0",
             "552,435", "20,000", "532,435", "196,288", "58,060", "91,430", "-33,370",
             "530,887", "301,229", "229,658", "-10,462,299"]
GWC_0914 = ["6488", "環球晶", "1,530,883", "1,839,712", "-308,829", "0", "0", "0",
            "1,530,883", "1,839,712", "-308,829", "4,110", "600", "3,510", "113,241", "34,000",
            "79,241", "149,833", "148,473", "1,360", "263,074", "182,473", "80,601", "-224,718"]

NOW_HISTORICAL = "2026-09-20 12:00:00"


def _taipei(iso):
    return datetime.fromisoformat(iso).replace(tzinfo=taiwan_common.TAIPEI)


def _values(foreign=(10, 4), fdealer=(0, 0), trust=(5, 1), prop=(3, 2), hedge=(7, 9)):
    v = {}
    for key, (buy, sell) in (("foreign", foreign), ("foreign_dealer", fdealer),
                             ("investment_trust", trust), ("dealer_proprietary", prop),
                             ("dealer_hedge", hedge)):
        v[f"{key}_buy"], v[f"{key}_sell"], v[f"{key}_net"] = buy, sell, buy - sell
    v["dealer_buy"] = v["dealer_proprietary_buy"] + v["dealer_hedge_buy"]
    v["dealer_sell"] = v["dealer_proprietary_sell"] + v["dealer_hedge_sell"]
    v["dealer_net"] = v["dealer_proprietary_net"] + v["dealer_hedge_net"]
    v["total_institutional_net"] = v["foreign_net"] + v["investment_trust_net"] + v["dealer_net"]
    return v


def _n(x):
    return f"{x:,}"


def twse_row(code, v, name="某公司"):
    keys = ["foreign_buy", "foreign_sell", "foreign_net", "foreign_dealer_buy", "foreign_dealer_sell",
            "foreign_dealer_net", "investment_trust_buy", "investment_trust_sell",
            "investment_trust_net", "dealer_net", "dealer_proprietary_buy", "dealer_proprietary_sell",
            "dealer_proprietary_net", "dealer_hedge_buy", "dealer_hedge_sell", "dealer_hedge_net",
            "total_institutional_net"]
    return [code, name.ljust(16)] + [_n(v[k]) for k in keys]


def _json_resp(url, obj, content_type="application/json;charset=UTF-8"):
    return BoundedResponse(url=url, status=200, headers=CaseInsensitiveDict({"Content-Type": content_type}),
                           body=json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def twse_resp(day, rows, *, fields=None, stat="OK", reported=None, hints="單位：股"):
    ymd = day.strftime("%Y%m%d")
    env = {"stat": stat, "date": reported or ymd, "title": "三大法人買賣超日報", "hints": hints,
           "fields": fields or TWSE_FIELDS, "data": rows, "notes": [], "total": len(rows)}
    return _json_resp(flows._TWSE_URL, env)


def twse_no_table(message="很抱歉，沒有符合條件的資料!"):
    return _json_resp(flows._TWSE_URL, {"stat": message})


def _roc(day, sep="/"):
    return f"{day.year - 1911}{sep}{day.month:02d}{sep}{day.day:02d}"


def tpex_csv(day, rows, *, header=None, title_day=None, filename_day="same",
             content_type="application/csv;charset=MS950"):
    title_day = title_day or day
    title = (f"{title_day.year - 1911}年{title_day.month:02d}月{title_day.day:02d}日 "
             "三大法人日交易資訊(含普通股、鉅額、零股、綜合帳戶之投信買賣成交量)依股票代碼排序")
    lines = [title, ",".join(header or TPEX_HEADER)]
    lines += [",".join(f'"{c}"' for c in row) for row in rows]
    lines += ['"說明:"', '"因外資自營商買賣金額已計入自營商買賣金額，故不納入三大法人買賣金額之合計數計算。"']
    headers = {"Content-Type": content_type}
    fday = day if filename_day == "same" else filename_day
    if fday is not None:
        headers["Content-Disposition"] = f'attachment; filename="BIGD_{_roc(fday, "")}.csv"'
    return BoundedResponse(url=flows._TPEX_URL, status=200, headers=CaseInsensitiveDict(headers),
                           body="\n".join(lines).encode("cp950"))


def tpex_no_table(day, reported=None, data=()):
    roc = reported or _roc(day)
    return _json_resp(flows._TPEX_URL, {"csvName": f"BIGD_{_roc(day, '')}.csv", "columnNum": 25,
                                        "tables": [{"title": "", "date": roc, "fields": TPEX_HEADER,
                                                    "data": list(data)}]})


class FakeExchanges:
    """Stand-in for ``bounded_request``: serves per-date replies and records calls."""

    def __init__(self):
        self.twse: dict[date, object] = {}
        self.tpex: dict[date, object] = {}
        self.calls: list[tuple[str, str, dict, str]] = []

    def __call__(self, method, url, *, error_cls, source, accept, params=None, data=None, **kw):
        if url == flows._TWSE_URL:
            self.calls.append((source, method, dict(params), accept))
            day = datetime.strptime(params["date"], "%Y%m%d").date()
            reply = self.twse.get(day, twse_no_table())
        elif url == flows._TPEX_URL:
            self.calls.append((source, method, dict(data), accept))
            day = datetime.strptime(data["date"], "%Y/%m/%d").date()
            reply = self.tpex.get(day, tpex_no_table(day))
        else:
            raise AssertionError(url)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def dates(self, source=None):
        out = []
        for src, _, payload, _ in self.calls:
            if source and src != source:
                continue
            raw = payload["date"]
            out.append(datetime.strptime(raw, "%Y%m%d" if "/" not in raw else "%Y/%m/%d").date())
        return out


def D(text):
    return date.fromisoformat(text)


@pytest.mark.unit
class _Base(unittest.TestCase):
    NOW = NOW_HISTORICAL

    def setUp(self):
        flows.clear_cache()
        flows._last_request_done.clear()
        self.ex = FakeExchanges()
        self.clock = [0.0]
        self.slept = []

        def sleep(seconds):
            self.slept.append(seconds)
            self.clock[0] += seconds

        self._patches = [
            mock.patch.object(flows, "bounded_request", self.ex),
            mock.patch.object(flows, "_now", lambda: _taipei(self.NOW)),
            mock.patch.object(flows, "_monotonic", lambda: self.clock[0]),
            mock.patch.object(flows, "_sleep", sleep),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        flows.clear_cache()
        flows._last_request_done.clear()

    def run_tool(self, ticker="2330.TW", curr_date="2026-09-15", n=1):
        return flows.get_institutional_flows(ticker, curr_date, n)


# --- routing and request contract -----------------------------------------------------

class RoutingTests(_Base):
    def test_tw_uses_twse_t86_json(self):
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [TSMC_0914])
        self.run_tool("2330.TW")
        self.assertEqual(self.ex.calls, [
            ("TWSE", "GET", {"date": "20260914", "selectType": "ALLBUT0999", "response": "json"},
             "application/json"),
        ])

    def test_two_uses_tpex_daily_trade_csv(self):
        self.ex.tpex[D("2026-09-14")] = tpex_csv(D("2026-09-14"), [GWC_0914])
        self.run_tool("6488.two")
        self.assertEqual(self.ex.calls, [
            ("TPEx", "POST", {"type": "Daily", "sect": "EW", "date": "2026/09/14", "response": "csv"},
             "text/csv"),
        ])

    def test_non_taiwan_and_non_numeric_make_no_request(self):
        for ticker in ("AAPL", "7203.T", "0700.HK", "ABCD.TW", ""):
            with self.subTest(ticker=ticker), self.assertRaises(NoMarketDataError) as ctx:
                self.run_tool(ticker)
            self.assertIn("not queried", str(ctx.exception))
        self.assertEqual(self.ex.calls, [])

    def test_look_back_is_clamped(self):
        for d in range(1, 31):           # 20 weekdays in 2026-08-01..30; 08-31 has no table
            day = date(2026, 8, d)
            if day.weekday() < 5:
                self.ex.twse[day] = twse_resp(day, [twse_row("2330", _values())])
        out = flows.get_institutional_flows("2330.TW", "2026-09-01", 0)
        self.assertEqual(out.count("| 2026-08-"), 2)        # one nets row + one trades row
        flows.clear_cache()
        out = flows.get_institutional_flows("2330.TW", "2026-09-01", 99)
        self.assertEqual(out.count("| 2026-08-"), 2 * 20)


# --- values and output -----------------------------------------------------------------------

class OutputTests(_Base):
    def test_twse_values_and_absent_dealer_totals(self):
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [TSMC_0914])
        out = self.run_tool("2330.TW")
        self.assertIn("| 2026-09-14 | -11,191,022 | 0 | 532,435 | 196,288 | -33,370 | 229,658 | -10,462,299 |", out)
        self.assertIn("| 2026-09-14 | 5,128,006 | 16,319,028 | 0 | 0 | 552,435 | 20,000 | "
                      "absent in source | absent in source | 58,060 | 91,430 | 530,887 | 301,229 |", out)
        self.assertIn("Unit: shares", out)
        self.assertIn("Company name in the exchange table: 台積電.", out)
        self.assertIn("foreign-dealer trades are already counted under dealers", out)
        self.assertIn("TWSE listed", out)

    def test_tpex_values_including_dealer_totals(self):
        self.ex.tpex[D("2026-09-14")] = tpex_csv(D("2026-09-14"), [GWC_0914])
        out = self.run_tool("6488.TWO")
        self.assertIn("| 2026-09-14 | -308,829 | 0 | 3,510 | 80,601 | 79,241 | 1,360 | -224,718 |", out)
        self.assertIn("| 263,074 | 182,473 |", out)
        self.assertNotIn("absent in source |", out.split("## Buys and sells")[1])
        self.assertIn("TPEx listed", out)

    def test_ticker_missing_from_table_is_not_zero(self):
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [twse_row("2303", _values())])
        out = self.run_tool("2330.TW")
        self.assertIn(f"| 2026-09-14 | {flows.NOT_IN_TABLE} | n/a |", out)
        net_row = next(line for line in out.splitlines() if line.startswith("| 2026-09-14"))
        self.assertNotIn("| 0 |", net_row)


# --- labels, units, reported dates -----------------------------------------------------------

class ValidationTests(_Base):
    def _fails(self, part, ticker="2330.TW"):
        with self.assertRaises(flows.TaiwanExchangeUnavailableError) as ctx:
            self.run_tool(ticker)
        self.assertIn(part, str(ctx.exception))

    def test_twse_reordered_and_extra_columns_are_mapped_by_label(self):
        order = [0, 18, 1, 5, 6, 7, 2, 3, 4, 11, 8, 9, 10, 15, 16, 17, 12, 13, 14]
        fields = [TWSE_FIELDS[i] for i in order] + ["備註"]
        row = [TSMC_0914[i] for i in order] + ["x"]
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [row], fields=fields)
        out = self.run_tool("2330.TW")
        self.assertIn("| 2026-09-14 | -11,191,022 | 0 | 532,435 | 196,288 | -33,370 | 229,658 | -10,462,299 |", out)

    def test_tpex_reordered_csv_columns_are_mapped_by_label(self):
        order = list(range(len(TPEX_HEADER)))
        order[2:5], order[20:23] = order[20:23], order[2:5]
        header = [TPEX_HEADER[i] for i in order]
        row = [GWC_0914[i] for i in order]
        self.ex.tpex[D("2026-09-14")] = tpex_csv(D("2026-09-14"), [row], header=header)
        out = self.run_tool("6488.TWO")
        self.assertIn("| 2026-09-14 | -308,829 | 0 | 3,510 | 80,601 | 79,241 | 1,360 | -224,718 |", out)

    def test_missing_required_columns_fail_closed(self):
        for label in ("外陸資買賣超股數(不含外資自營商)", "自營商買賣超股數(避險)", "三大法人買賣超股數", "證券名稱"):
            with self.subTest(label=label):
                i = TWSE_FIELDS.index(label)
                fields = TWSE_FIELDS[:i] + TWSE_FIELDS[i + 1:]
                row = TSMC_0914[:i] + TSMC_0914[i + 1:]
                self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [row], fields=fields)
                self._fails(f"required column {label!r} not found")
        i = TPEX_HEADER.index("自營商-買進股數")
        self.ex.tpex[D("2026-09-14")] = tpex_csv(
            D("2026-09-14"), [GWC_0914[:i] + GWC_0914[i + 1:]], header=TPEX_HEADER[:i] + TPEX_HEADER[i + 1:])
        self._fails("required column '自營商-買進股數' not found", ticker="6488.TWO")

    def test_duplicate_column_fails_closed(self):
        self.ex.twse[D("2026-09-14")] = twse_resp(
            D("2026-09-14"), [TSMC_0914 + ["1"]], fields=TWSE_FIELDS + ["投信買賣超股數"])
        self._fails("'投信買賣超股數' appears 2 times")

    def test_twse_wrong_unit_fails_closed(self):
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [TSMC_0914], hints="單位：千股")
        self._fails("unit is '單位：千股'")

    def test_twse_reported_date_mismatch_fails_closed(self):
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [TSMC_0914], reported="20260911")
        self._fails("reported date '20260911' for a 20260914 request")

    def test_tpex_silent_substitution_fails_closed(self):
        # A malformed request can return the latest day's table: title and file say 09-14.
        self.ex.tpex[D("2026-09-14")] = tpex_csv(D("2026-09-14"), [GWC_0914], title_day=D("2026-09-11"))
        self._fails("CSV title reports", ticker="6488.TWO")

    def test_tpex_filename_mismatch_fails_closed(self):
        self.ex.tpex[D("2026-09-14")] = tpex_csv(D("2026-09-14"), [GWC_0914], filename_day=D("2026-09-11"))
        self._fails("does not match 'BIGD_1150914.csv'", ticker="6488.TWO")

    def test_tpex_rewritten_empty_day_fails_closed(self):
        # 2026/02/31-style rewrite: the empty JSON reply reports another date.
        self.ex.tpex[D("2026-09-14")] = tpex_no_table(D("2026-09-14"), reported="115/09/13")
        self._fails("reported date ['115/09/13'] for a 115/09/14 request", ticker="6488.TWO")

    def test_tpex_json_with_rows_or_unexpected_content_type_fails_closed(self):
        self.ex.tpex[D("2026-09-14")] = tpex_no_table(D("2026-09-14"), data=[GWC_0914])
        self._fails("returned table rows as JSON", ticker="6488.TWO")
        self.ex.tpex[D("2026-09-14")] = tpex_csv(D("2026-09-14"), [GWC_0914], content_type="text/html")
        self._fails("unexpected content type 'text/html'", ticker="6488.TWO")

    def test_unexpected_twse_stat_fails_closed(self):
        self.ex.twse[D("2026-09-14")] = twse_no_table("查詢日期小於101年05月02日，請重新查詢!")
        self._fails("T86 returned stat")

    def test_row_integrity_failures(self):
        cases = {
            "total": (18, "-10,462,298", "foreign + investment trust + dealer != total"),
            "dealer split": (11, "196,289", "dealer proprietary + hedge != dealer net"),
            "buy-sell": (4, "-11,191,021", "foreign buy - sell != net"),
            "non-numeric": (8, "--", "non-numeric investment_trust_buy value '--'"),
        }
        for name, (index, value, part) in cases.items():
            with self.subTest(case=name):
                flows.clear_cache()
                row = list(TSMC_0914)
                row[index] = value
                self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [row])
                self._fails(part)
        row = list(GWC_0914)
        row[20], row[21] = "263,075", "182,474"   # dealer buy and sell both +1: net still agrees
        self.ex.tpex[D("2026-09-14")] = tpex_csv(D("2026-09-14"), [row])
        self._fails("dealer proprietary + hedge buys != dealer buys", ticker="6488.TWO")

    def test_duplicate_code_and_row_width_fail_closed(self):
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [TSMC_0914, TSMC_0914])
        self._fails("security code '2330' appears twice")
        flows.clear_cache()
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [TSMC_0914[:-1]])
        self._fails("row width does not match 19 columns")


# --- no-table replies, weekends, the walk ------------------------------------------------------

class WalkTests(_Base):
    def test_no_table_replies_are_skipped_and_listed(self):
        self.ex.twse[D("2026-09-14")] = twse_no_table()
        self.ex.twse[D("2026-09-11")] = twse_no_table("查詢日期大於可查詢最大日期，請重新查詢!")
        self.ex.twse[D("2026-09-10")] = twse_resp(D("2026-09-10"), [TSMC_0914])
        out = self.run_tool("2330.TW")
        self.assertIn("| 2026-09-10 |", out)
        self.assertIn("Weekdays without an official table (not a trading day, or not published): "
                      "2026-09-14, 2026-09-11.", out)

    def test_weekends_are_never_requested(self):
        for day in (D("2026-09-11"), D("2026-09-10")):
            self.ex.twse[day] = twse_resp(day, [twse_row("2330", _values())])
        self.run_tool("2330.TW", "2026-09-14", n=2)     # Monday: walk starts Sunday
        self.assertEqual(self.ex.dates(), [D("2026-09-11"), D("2026-09-10")])

    def test_lunar_new_year_gap_is_walked(self):
        self.ex.twse[D("2026-02-23")] = twse_resp(D("2026-02-23"), [twse_row("2330", _values())])
        self.ex.twse[D("2026-02-11")] = twse_resp(D("2026-02-11"), [twse_row("2330", _values())])
        out = self.run_tool("2330.TW", "2026-02-24", n=2)
        self.assertIn("| 2026-02-23 |", out)
        self.assertIn("| 2026-02-11 |", out)
        self.assertIn("2026-02-20, 2026-02-19, 2026-02-18, 2026-02-17, 2026-02-16, 2026-02-13, 2026-02-12", out)

    def test_weekday_cap_then_no_data(self):
        with self.assertRaises(NoMarketDataError) as ctx:
            self.run_tool("2330.TW", "2026-09-15", n=1)
        self.assertEqual(len(self.ex.calls), 1 + flows._EXTRA_WEEKDAYS)
        self.assertIn("no official TWSE institutional-flow table found", str(ctx.exception))

    def test_partial_window_is_reported(self):
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [TSMC_0914])
        out = self.run_tool("2330.TW", "2026-09-15", n=3)
        self.assertIn("Only 1 of the 3 requested trading days were found.", out)
        self.assertIn("Stopped after examining 13 weekdays.", out)

    def test_supported_from_floor(self):
        for day in (D("2017-12-19"), D("2017-12-18")):
            self.ex.twse[day] = twse_resp(day, [twse_row("2330", _values())])
        out = self.run_tool("2330.TW", "2017-12-20", n=5)
        self.assertIn("Dates before 2017-12-18 use an older TWSE layout and are not supported", out)
        self.assertEqual(self.ex.dates(), [D("2017-12-19"), D("2017-12-18")])
        self.ex.calls.clear()
        with self.assertRaises(NoMarketDataError) as ctx:
            self.run_tool("2330.TW", "2017-12-18", n=5)
        self.assertIn("older layout that is not supported", str(ctx.exception))
        self.assertEqual(self.ex.calls, [])
        with self.assertRaises(NoMarketDataError):
            self.run_tool("6488.TWO", "2018-01-15", n=1)
        self.assertEqual(self.ex.calls, [])


# --- point-in-time ------------------------------------------------------------------------------

class PointInTimeTests(_Base):
    def _serve(self, include_same_day=True):
        self.ex.twse.clear()
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [twse_row("2330", _values(trust=(14, 0)))])
        if include_same_day:
            self.ex.twse[D("2026-09-15")] = twse_resp(D("2026-09-15"), [twse_row("2330", _values(trust=(15, 0)))])
            self.ex.twse[D("2026-09-16")] = twse_resp(D("2026-09-16"), [twse_row("2330", _values(trust=(16, 0)))])

    def test_historical_never_requests_curr_date_or_later(self):
        self._serve()
        out = self.run_tool("2330.TW", "2026-09-15", n=1)
        self.assertEqual(self.ex.dates(), [D("2026-09-14")])
        self.assertIn("| 2026-09-14 |", out)
        self.assertNotIn("2026-09-15 |", out)
        self.assertIn(flows.WITHHELD_NOTICE.format(curr_date="2026-09-15"), out)

    def test_historical_output_does_not_depend_on_same_day_data(self):
        self._serve(include_same_day=True)
        with_same_day = self.run_tool("2330.TW", "2026-09-15", n=1)
        flows.clear_cache()
        self._serve(include_same_day=False)
        without_same_day = self.run_tool("2330.TW", "2026-09-15", n=1)
        self.assertEqual(with_same_day, without_same_day)

    def test_live_includes_today_only_when_published_under_today(self):
        self.NOW = "2026-09-15 16:00:00"
        self._serve()
        out = self.run_tool("2330.TW", "2026-09-15", n=2)
        self.assertEqual(self.ex.dates(), [D("2026-09-15"), D("2026-09-14")])
        self.assertIn("| 2026-09-15 |", out)
        self.assertIn("it is included.", out)
        self.assertNotIn("2026-09-16", out)

    def test_live_before_publication_falls_back_without_listing_today_as_skipped(self):
        self.NOW = "2026-09-15 11:45:00"
        self._serve(include_same_day=False)
        out = self.run_tool("2330.TW", "2026-09-15", n=1)
        self.assertEqual(self.ex.dates(), [D("2026-09-15"), D("2026-09-14")])
        self.assertIn("it is not available yet (not published, or not a trading day).", out)
        self.assertNotIn("Weekdays without an official table", out)

    def test_live_today_with_wrong_reported_date_fails_closed(self):
        self.NOW = "2026-09-15 16:00:00"
        self.ex.twse[D("2026-09-15")] = twse_resp(D("2026-09-15"), [TSMC_0914], reported="20260914")
        with self.assertRaises(flows.TaiwanExchangeUnavailableError):
            self.run_tool("2330.TW", "2026-09-15", n=1)


# --- cache and pacing ------------------------------------------------------------------------------

class CacheAndPacingTests(_Base):
    def _serve_week(self):
        for day in (D("2026-09-14"), D("2026-09-11"), D("2026-09-10")):
            self.ex.twse[day] = twse_resp(day, [twse_row("2330", _values()), twse_row("2303", _values())])

    def test_second_call_and_second_ticker_hit_the_cache(self):
        self._serve_week()
        first = self.run_tool("2330.TW", "2026-09-15", n=3)
        requests_after_first = len(self.ex.calls)
        slept_after_first = len(self.slept)
        self.assertEqual(self.run_tool("2330.TW", "2026-09-15", n=3), first)
        self.run_tool("2303.TW", "2026-09-15", n=3)
        self.assertEqual(len(self.ex.calls), requests_after_first)
        self.assertEqual(len(self.slept), slept_after_first)

    def test_no_table_replies_and_failures_are_not_cached(self):
        self.ex.twse[D("2026-09-11")] = twse_resp(D("2026-09-11"), [TSMC_0914])
        self.run_tool("2330.TW", "2026-09-15", n=1)            # 09-14 empty, 09-11 table
        self.run_tool("2330.TW", "2026-09-15", n=1)
        self.assertEqual(self.ex.dates().count(D("2026-09-14")), 2)
        self.assertEqual(self.ex.dates().count(D("2026-09-11")), 1)
        flows.clear_cache()
        self.ex.twse[D("2026-09-10")] = flows.TaiwanExchangeUnavailableError("TWSE fetch failed (Timeout)")
        for _ in range(2):
            with self.assertRaises(flows.TaiwanExchangeUnavailableError):
                self.run_tool("2330.TW", "2026-09-11", n=1)
        self.assertEqual(self.ex.dates().count(D("2026-09-10")), 2)

    def test_todays_table_is_never_cached(self):
        self.NOW = "2026-09-15 16:00:00"
        self.ex.twse[D("2026-09-15")] = twse_resp(D("2026-09-15"), [TSMC_0914])
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [TSMC_0914])
        self.run_tool("2330.TW", "2026-09-15", n=2)
        self.run_tool("2330.TW", "2026-09-15", n=2)
        self.assertEqual(self.ex.dates().count(D("2026-09-15")), 2)
        self.assertEqual(self.ex.dates().count(D("2026-09-14")), 1)

    def test_clear_cache_and_board_separation(self):
        self._serve_week()
        self.ex.tpex[D("2026-09-14")] = tpex_csv(D("2026-09-14"), [GWC_0914])
        self.run_tool("2330.TW", "2026-09-15", n=1)
        self.run_tool("6488.TWO", "2026-09-15", n=1)
        self.assertEqual(len(self.ex.calls), 2)
        flows.clear_cache()
        self.run_tool("2330.TW", "2026-09-15", n=1)
        self.assertEqual(len(self.ex.calls), 3)

    def test_cache_is_bounded(self):
        self._serve_week()
        with mock.patch.object(flows, "_CACHE_MAX_TABLES", 2):
            self.run_tool("2330.TW", "2026-09-15", n=3)
            self.assertEqual(list(flows._cache), [("sii", "2026-09-11"), ("sii", "2026-09-10")])

    def test_requests_are_spaced_per_exchange_across_calls(self):
        self.assertEqual(flows._REQUEST_SPACING_SECONDS, {"sii": 5.0, "otc": 6.0})
        self._serve_week()
        self.ex.tpex[D("2026-09-14")] = tpex_csv(D("2026-09-14"), [GWC_0914])
        self.ex.tpex[D("2026-09-11")] = tpex_csv(D("2026-09-11"), [GWC_0914])
        self.run_tool("2330.TW", "2026-09-15", n=2)        # two TWSE requests: one 5 s gap
        self.assertEqual(self.slept, [5.0])
        self.run_tool("6488.TWO", "2026-09-15", n=2)       # TPEx is paced separately: one 6 s gap
        self.assertEqual(self.slept, [5.0, 6.0])
        self.run_tool("2330.TW", "2026-09-15", n=3)        # cached 09-14/09-11, then a new 09-10
        self.assertEqual(self.slept, [5.0, 6.0])           # enough clock time has passed already
        flows.clear_cache()
        flows._last_request_done["sii"] = self.clock[0] - 2.0
        self.run_tool("2330.TW", "2026-09-15", n=1)        # previous TWSE request ended 2 s ago
        self.assertEqual(self.slept[-1], 3.0)


# --- router ---------------------------------------------------------------------------------------

class RouterTests(_Base):
    def test_category_vendor_resolves(self):
        self.assertEqual(interface.get_category_for_method("get_institutional_flows"), "taiwan_flow_data")
        self.assertEqual(interface.get_vendor("taiwan_flow_data", "get_institutional_flows"), "twse_tpex")
        self.assertIn("taiwan_flow_data", interface.OPTIONAL_CATEGORIES)
        self.assertEqual(list(interface.VENDOR_METHODS["get_institutional_flows"]), ["twse_tpex"])

    def test_router_passes_results_and_degrades_failures(self):
        self.ex.twse[D("2026-09-14")] = twse_resp(D("2026-09-14"), [TSMC_0914])
        out = interface.route_to_vendor("get_institutional_flows", "2330.TW", "2026-09-15", 1)
        self.assertIn("| 2026-09-14 | -11,191,022 |", out)
        flows.clear_cache()
        self.ex.twse[D("2026-09-14")] = flows.TaiwanExchangeUnavailableError("TWSE fetch failed (HTTP 503)")
        out = interface.route_to_vendor("get_institutional_flows", "2330.TW", "2026-09-15", 1)
        self.assertTrue(out.startswith("DATA_UNAVAILABLE:"), out)
        out = interface.route_to_vendor("get_institutional_flows", "AAPL", "2026-09-15", 5)
        self.assertTrue(out.startswith("NO_DATA_AVAILABLE:"), out)


# --- Market Analyst wiring --------------------------------------------------------------------------

def _run_market_analyst(ticker):
    from tradingagents.agents.analysts.market_analyst import create_market_analyst
    captured = {}

    def bind_tools(tools):
        captured["tools"] = [t.name for t in tools]

        def run(prompt_value):
            captured["prompt"] = prompt_value.to_string()
            return AIMessage(content="report")
        return run

    llm = mock.MagicMock()
    llm.bind_tools.side_effect = bind_tools
    out = create_market_analyst(llm)({"company_of_interest": ticker, "trade_date": "2026-09-15",
                                      "asset_type": "stock", "messages": []})
    assert out["market_report"] == "report"
    return captured


_DEFAULT_MARKET_TOOLS = ["get_stock_data", "get_indicators", "get_verified_market_snapshot"]


@pytest.mark.unit
class MarketAnalystWiringTests(unittest.TestCase):
    def test_taiwan_binds_flows_tool_and_guidance(self):
        for ticker in ("2330.TW", "6488.TWO"):
            with self.subTest(ticker=ticker):
                c = _run_market_analyst(ticker)
                self.assertEqual(c["tools"], _DEFAULT_MARKET_TOOLS + ["get_institutional_flows"])
                self.assertIn("official institutional flow tool", c["prompt"])
                self.assertIn("missing data, not zero", c["prompt"])
                self.assertIn("not as an automatic buy or sell signal", c["prompt"])

    def test_default_market_unchanged(self):
        for ticker in ("AAPL", "NVDA", "7203.T"):
            with self.subTest(ticker=ticker):
                c = _run_market_analyst(ticker)
                self.assertEqual(c["tools"], _DEFAULT_MARKET_TOOLS)
                self.assertNotIn("institutional flow", c["prompt"])

    def test_market_tool_node_can_execute_it(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        tools = TradingAgentsGraph._create_tool_nodes(None)["market"].tools_by_name
        for name in _DEFAULT_MARKET_TOOLS + ["get_institutional_flows"]:
            self.assertIn(name, tools)


# --- hardening: future dates and concurrent runs ------------------------------------------------

class FutureDateTests(_Base):
    def test_future_curr_date_is_refused_without_requests(self):
        self.NOW = "2026-09-15 12:00:00"
        with self.assertRaises(ValueError) as ctx:
            self.run_tool("2330.TW", "2026-10-01", n=1)
        self.assertIn("future dates are not supported", str(ctx.exception))
        self.assertEqual(self.ex.calls, [])


@pytest.mark.unit
class ConcurrencyTests(unittest.TestCase):
    """Real threads and a real clock with short spacing: the per-exchange lock
    must serialize requests to one exchange without blocking the other."""

    SPACING = 0.08
    FETCH_SECONDS = 0.10
    TODAY = date(2026, 9, 20)

    def setUp(self):
        flows.clear_cache()
        flows._last_request_done.clear()
        self.events = []                       # (board, start, end)
        self.events_lock = threading.Lock()
        self.inflight = {"sii": 0, "otc": 0, "all": 0}
        self.max_inflight = {"sii": 0, "otc": 0, "all": 0}

        def fake_fetch(board):
            def fetch(day):
                with self.events_lock:
                    for key in (board, "all"):
                        self.inflight[key] += 1
                        self.max_inflight[key] = max(self.max_inflight[key], self.inflight[key])
                start = time.monotonic()
                time.sleep(self.FETCH_SECONDS)
                end = time.monotonic()
                with self.events_lock:
                    for key in (board, "all"):
                        self.inflight[key] -= 1
                    self.events.append((board, start, end))
                return flows._Table(board, day, {"code": 0}, {"2330": ("2330",)})
            return fetch

        self._patches = [
            mock.patch.object(flows, "_fetch_twse", fake_fetch("sii")),
            mock.patch.object(flows, "_fetch_tpex", fake_fetch("otc")),
            mock.patch.object(flows, "_REQUEST_SPACING_SECONDS", {"sii": self.SPACING, "otc": self.SPACING}),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        flows.clear_cache()
        flows._last_request_done.clear()

    def _run_threads(self, jobs):
        barrier = threading.Barrier(len(jobs))
        results = [None] * len(jobs)

        def worker(i, board, day):
            barrier.wait()
            results[i] = flows._load_table(board, day, self.TODAY)

        threads = [threading.Thread(target=worker, args=(i, b, d)) for i, (b, d) in enumerate(jobs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        return results

    def test_one_exchange_is_serialized_and_spaced(self):
        jobs = [("sii", date(2026, 9, d)) for d in (7, 8, 9, 10)]
        self._run_threads(jobs)
        self.assertEqual(self.max_inflight["sii"], 1)
        spans = sorted((s, e) for b, s, e in self.events if b == "sii")
        self.assertEqual(len(spans), 4)
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
            self.assertGreaterEqual(next_start - prev_end, self.SPACING - 0.01)

    def test_exchanges_do_not_block_each_other(self):
        self._run_threads([("sii", date(2026, 9, 10)), ("otc", date(2026, 9, 10))])
        self.assertEqual(self.max_inflight["all"], 2)

    def test_concurrent_misses_for_one_table_fetch_it_once(self):
        results = self._run_threads([("sii", date(2026, 9, 10))] * 3)
        self.assertEqual(len(self.events), 1)
        self.assertTrue(all(r is results[0] and r is not None for r in results))
