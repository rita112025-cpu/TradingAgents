"""MOPS material announcements adapter (Taiwan Adapter step 4b-1).

Everything is mocked: the MOPS API at ``mops_announcements._post_json`` (or at
``requests.post`` for the HTTP layer), the clock at ``_now`` and the politeness
gap at ``_sleep``. Fixture envelopes copy the live ``t05st01`` /
``t05st01_detail`` shapes: ``{"code", "message", "result"}`` with ``titles``
labels, ROC dates (``115/09/14``), ``HH:MM:SS`` times, and a detail reference
object per row.
"""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest import mock

import pytest
import requests
from langchain_core.messages import AIMessage

from tradingagents.dataflows import bounded_http, interface, mops_announcements as ma, taiwan_common
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.mops_common import MopsUnavailableError

_REAL_POST_JSON = ma._post_json

_LIST_TITLES = ["公司代號", "公司名稱", "發言日期", "發言時間", "主旨", "詳細資料"]
_DETAIL_TITLES = ["序號", "發言日期", "發言時間", "發言人", "發言人職稱", "發言人電話",
                  "主旨", "符合條款", "事實發生日", "說明"]
_MARKET = {"sii": "上市公司", "otc": "上櫃公司"}
_NAME = {"2330": "台積電", "6488": "環球晶"}


def _taipei(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=taiwan_common.TAIPEI)


def _roc(iso_date: str) -> str:
    y, m, d = iso_date.split("-")
    return f"{int(y) - 1911}/{m}/{d}"


def _titles(labels):
    return [{"sub": [], "main": label} for label in labels]


class _Ann:
    """One announcement in the fake MOPS database."""

    def __init__(self, when, subject, code="2330", board="sii", enter=None, serial="1",
                 clause="第51款", fact=None, text="說明內容"):
        self.when = when                    # "2026-09-14 10:00:00"
        self.date, self.time = when.split(" ")
        self.subject, self.code, self.board = subject, code, board
        self.enter = enter or self.date.replace("-", "")
        self.enter = f"{int(self.enter[:4]) - 1911}{self.enter[4:]}"  # ROC yyyMMdd
        self.serial, self.clause, self.text = serial, clause, text
        self.fact = fact or self.date

    def list_row(self, labels=_LIST_TITLES):
        values = {
            "公司代號": self.code, "公司名稱": _NAME.get(self.code, "某公司"),
            "發言日期": _roc(self.date), "發言時間": self.time, "主旨": self.subject,
            "詳細資料": {"parameters": {"companyId": self.code, "marketKind": self.board,
                                    "enterDate": self.enter, "serialNumber": self.serial},
                       "apiName": "t05st01_detail"},
        }
        return [values.get(label, "extra") for label in labels]

    def detail_result(self):
        row = {"序號": self.serial, "發言日期": _roc(self.date), "發言時間": self.time,
               "發言人": "發言人  ", "發言人職稱": "財務長", "發言人電話": "03-000",
               "主旨": self.subject, "符合條款": self.clause, "事實發生日": _roc(self.fact),
               "說明": self.text}
        return {"marketName": _MARKET[self.board], "companyId": self.code,
                "titles": _titles(_DETAIL_TITLES), "data": [[row[t] for t in _DETAIL_TITLES]],
                "footer": [], "header": {}, "companyAbbreviation": _NAME.get(self.code, "")}


def _ok(result):
    return {"code": 200, "message": "查詢成功", "result": result, "datetime": "115/09/15 11:00:00"}


_NO_MATCH = {"code": 406, "message": "查無相符資料", "result": None}
_BAD_PARAMS = {"code": 500, "message": "傳入參數異常", "result": None}


class _FakeMops:
    """Stand-in for ``_post_json``: serves list months and details, records calls.

    The list handler deliberately ignores firstDay/lastDay and returns the whole
    month, so client-side point-in-time filtering is what the tests exercise.
    """

    def __init__(self, announcements=(), code="2330", board="sii"):
        self.announcements = list(announcements)
        self.code, self.board = code, board
        self.market_name = _MARKET[board]
        self.list_titles = _LIST_TITLES
        self.list_override = None             # envelope or exception for every list call
        self.detail_override = {}             # (enter, serial) -> envelope or exception
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, api, body, timeout=None):
        self.calls.append((api, dict(body)))
        if api == "t05st01":
            if isinstance(self.list_override, Exception):
                raise self.list_override
            if self.list_override is not None:
                return self.list_override
            year, month = int(body["year"]) + 1911, int(body["month"])
            rows = [a for a in self.announcements
                    if a.code == body["companyId"]
                    and (int(a.date[:4]), int(a.date[5:7])) == (year, month)]
            if not rows:
                return _NO_MATCH
            return _ok({"marketName": self.market_name, "companyId": body["companyId"],
                        "titles": _titles(self.list_titles),
                        "data": [a.list_row(self.list_titles) for a in rows],
                        "companyAbbreviation": _NAME.get(self.code, "")})
        if api == "t05st01_detail":
            key = (body["enterDate"], body["serialNumber"])
            override = self.detail_override.get(key)
            if isinstance(override, Exception):
                raise override
            if override is not None:
                return override
            for a in self.announcements:
                if (a.enter, a.serial) == key:
                    return _ok(a.detail_result())
            return _NO_MATCH
        raise AssertionError(f"unexpected api {api}")

    def list_calls(self):
        return [b for api, b in self.calls if api == "t05st01"]

    def detail_calls(self):
        return [b for api, b in self.calls if api == "t05st01_detail"]


@pytest.mark.unit
class _Base(unittest.TestCase):
    NOW = "2026-09-20 12:00:00"

    def setUp(self):
        self.mops = _FakeMops()
        self._patches = [
            mock.patch.object(ma, "_post_json", self.mops),
            mock.patch.object(ma, "_now", lambda: _taipei(self.NOW)),
            mock.patch.object(ma, "_sleep", lambda s: None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def serve(self, *anns, code="2330", board="sii"):
        self.mops.announcements = list(anns)
        self.mops.code, self.mops.board = code, board
        self.mops.market_name = _MARKET[board]

    def run_tool(self, ticker="2330.TW", curr_date="2026-09-15", **kw):
        return ma.get_material_announcements(ticker, curr_date, **kw)


# --- Taiwan routing ---------------------------------------------------------------

class RoutingTests(_Base):
    def test_tw_two_and_lowercase_tickers_query_mops(self):
        self.serve(_Ann("2026-09-10 10:00:00", "TW 公告"))
        for ticker in ("2330.TW", "2330.tw"):
            with self.subTest(ticker=ticker):
                self.mops.calls.clear()
                out = self.run_tool(ticker)
                self.assertIn("TW 公告", out)
                self.assertTrue(all(b["companyId"] == "2330" for b in self.mops.list_calls()))
        self.serve(_Ann("2026-09-10 10:00:00", "TWO 公告", code="6488", board="otc"),
                   code="6488", board="otc")
        for ticker in ("6488.TWO", "6488.two"):
            with self.subTest(ticker=ticker):
                self.mops.calls.clear()
                out = self.run_tool(ticker)
                self.assertIn("TWO 公告", out)
                self.assertIn("上櫃公司", out)

    def test_non_taiwan_tickers_make_zero_requests(self):
        for ticker in ("AAPL", "7203.T", "0700.HK", "SHOP.TO", ""):
            with self.subTest(ticker=ticker), self.assertRaises(NoMarketDataError) as ctx:
                self.run_tool(ticker)
            self.assertIn("not queried", str(ctx.exception))
        self.assertEqual(self.mops.calls, [])

    def test_list_request_contract_always_sends_day_bounds(self):
        self.serve()
        self.run_tool("2330.TW", "2026-09-15", look_back_days=30)
        # Window 2026-08-16 .. 2026-09-14 -> one request per month, ROC year,
        # unpadded month, explicit firstDay/lastDay.
        self.assertEqual(self.mops.list_calls(), [
            {"companyId": "2330", "year": "115", "month": "8", "firstDay": "16", "lastDay": "31"},
            {"companyId": "2330", "year": "115", "month": "9", "firstDay": "1", "lastDay": "14"},
        ])

    def test_look_back_and_detail_limit_are_clamped(self):
        self.serve()
        self.run_tool("2330.TW", "2026-09-15", look_back_days=3650)
        self.assertLessEqual(len(self.mops.list_calls()), 4)   # <= 90 days of months


# --- board validation ----------------------------------------------------------------

class BoardValidationTests(_Base):
    def test_matching_boards_pass(self):
        self.serve(_Ann("2026-09-10 10:00:00", "上市公告"))
        self.assertIn("上市公司", self.run_tool("2330.TW"))
        self.serve(_Ann("2026-09-10 10:00:00", "上櫃公告", code="6488", board="otc"),
                   code="6488", board="otc")
        self.assertIn("上櫃公司", self.run_tool("6488.TWO"))

    def test_tw_ticker_reported_as_otc_fails_closed(self):
        self.serve(_Ann("2026-09-10 10:00:00", "x"))
        self.mops.market_name = "上櫃公司"
        with self.assertRaises(ma.MopsBoardMismatchError) as ctx:
            self.run_tool("2330.TW")
        self.assertIn("上櫃公司", str(ctx.exception))
        self.assertEqual(self.mops.detail_calls(), [])

    def test_two_ticker_reported_as_listed_fails_closed(self):
        self.serve(_Ann("2026-09-10 10:00:00", "x", code="6488", board="otc"),
                   code="6488", board="otc")
        self.mops.market_name = "上市公司"
        with self.assertRaises(ma.MopsBoardMismatchError):
            self.run_tool("6488.TWO")

    def test_detail_reference_naming_other_board_fails_closed(self):
        self.serve(_Ann("2026-09-10 10:00:00", "x", board="otc"))   # row says otc
        self.mops.market_name = "上市公司"
        with self.assertRaises(ma.MopsBoardMismatchError):
            self.run_tool("2330.TW")
        self.assertEqual(self.mops.detail_calls(), [])

    def test_detail_response_on_other_board_fails_closed(self):
        ann = _Ann("2026-09-10 10:00:00", "x")
        self.serve(ann)
        bad = ann.detail_result() | {"marketName": "上櫃公司"}
        self.mops.detail_override[(ann.enter, ann.serial)] = _ok(bad)
        with self.assertRaises(ma.MopsBoardMismatchError):
            self.run_tool("2330.TW")

    def test_board_mismatch_is_a_vendor_failure_type(self):
        self.assertTrue(issubclass(ma.MopsBoardMismatchError, MopsUnavailableError))


# --- point-in-time -------------------------------------------------------------------

_PIT = (
    _Ann("2026-09-14 10:00:00", "A-0914-1000", serial="1"),
    _Ann("2026-09-15 10:00:00", "B-0915-1000", serial="1"),
    _Ann("2026-09-15 18:00:00", "C-0915-1800", serial="2"),
    _Ann("2026-09-16 09:00:00", "D-0916-0900", serial="1"),
)


class PointInTimeTests(_Base):
    def test_historical_curr_date_withholds_the_whole_same_day(self):
        self.serve(*_PIT)
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("A-0914-1000", out)
        for hidden in ("B-0915-1000", "C-0915-1800", "D-0916-0900"):
            self.assertNotIn(hidden, out)
        self.assertIn(ma.WITHHELD_SAME_DAY_NOTICE, out)
        self.assertIn("not evidence that 2026-09-15 had no announcements", out)
        # curr_date itself is never requested from MOPS.
        self.assertEqual(self.mops.list_calls()[-1]["lastDay"], "14")

    def test_historical_next_day_sees_the_previous_day(self):
        self.serve(*_PIT)
        out = self.run_tool("2330.TW", "2026-09-16")
        for shown in ("A-0914-1000", "B-0915-1000", "C-0915-1800"):
            self.assertIn(shown, out)
        self.assertNotIn("D-0916-0900", out)

    def test_withheld_output_does_not_leak_same_day_existence(self):
        self.serve(*_PIT)
        with_same_day = self.run_tool("2330.TW", "2026-09-15")
        self.serve(_PIT[0], _PIT[3])
        without_same_day = self.run_tool("2330.TW", "2026-09-15")
        self.assertEqual(with_same_day, without_same_day)

    def test_live_run_includes_only_what_is_published_by_now(self):
        self.NOW = "2026-09-15 12:00:00"
        self.serve(*_PIT)
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("A-0914-1000", out)
        self.assertIn("B-0915-1000", out)       # published 10:00, before 12:00
        self.assertNotIn("C-0915-1800", out)    # later today
        self.assertNotIn("D-0916-0900", out)    # tomorrow
        self.assertIn("Live run", out)
        self.assertIn("2026-09-15 12:00:00 Taiwan time", out)
        self.assertNotIn(ma.WITHHELD_SAME_DAY_NOTICE, out)

    def test_live_clock_moves_the_cutoff(self):
        self.NOW = "2026-09-15 18:30:00"
        self.serve(*_PIT)
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("C-0915-1800", out)
        self.assertNotIn("D-0916-0900", out)

    def test_filter_uses_publication_timestamp_not_enter_date(self):
        # Published on curr_date but enterDate the day before -> still withheld.
        late = _Ann("2026-09-15 18:00:00", "LATE-enter-0914", enter="20260914", serial="7")
        # Published the day before but enterDate on curr_date -> still shown.
        early = _Ann("2026-09-14 09:00:00", "EARLY-enter-0915", enter="20260915", serial="8")
        self.serve(late, early)
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("EARLY-enter-0915", out)
        self.assertNotIn("LATE-enter-0914", out)

    def test_window_lower_bound(self):
        self.serve(_Ann("2026-08-15 10:00:00", "TOO-OLD"), _Ann("2026-08-16 10:00:00", "IN-WINDOW"))
        out = self.run_tool("2330.TW", "2026-09-15", look_back_days=30)
        self.assertIn("IN-WINDOW", out)
        self.assertNotIn("TOO-OLD", out)


# --- detail cap and partial failure -----------------------------------------------------

def _eight():
    return [_Ann(f"2026-09-{d:02d} 1{d % 10}:00:00", f"S{d:02d}", serial="1", text=f"說明{d:02d}")
            for d in range(1, 9)]


class DetailTests(_Base):
    def test_detail_requested_only_for_newest_five(self):
        anns = _eight()
        self.serve(*anns)
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertEqual(len(self.mops.detail_calls()), 5)
        requested = {b["enterDate"] for b in self.mops.detail_calls()}
        self.assertEqual(requested, {a.enter for a in anns[3:]})       # 09-04 .. 09-08
        for a in anns:
            self.assertIn(a.subject, out)                               # all 8 listed
        self.assertEqual(out.count("| fetched |"), 5)
        self.assertEqual(out.count("not fetched (beyond detail cap)"), 3)
        for d in ("01", "02", "03"):
            self.assertNotIn(f"說明{d}", out)                           # no detail text for old
        for d in ("04", "08"):
            self.assertIn(f"說明{d}", out)
        self.assertIn("Detail was requested for the newest 5 (cap 5); 5 fetched, 0 unavailable", out)

    def test_detail_limit_above_cap_is_clamped(self):
        self.serve(*_eight())
        self.run_tool("2330.TW", "2026-09-15", detail_limit=50)
        self.assertEqual(len(self.mops.detail_calls()), 5)

    def test_one_detail_timeout_keeps_the_rest(self):
        anns = _eight()
        self.serve(*anns)
        victim = anns[6]   # 09-07, inside the newest five
        self.mops.detail_override[(victim.enter, victim.serial)] = MopsUnavailableError(
            "MOPS fetch failed (ReadTimeout) for https://mops.twse.com.tw/mops/api/t05st01_detail")
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertEqual(out.count("| fetched |"), 4)
        self.assertIn("detail unavailable: MOPS fetch failed (ReadTimeout)", out)
        self.assertIn(victim.subject, out)                 # list row kept
        self.assertNotIn(victim.text, out)                 # no invented detail
        self.assertIn("4 fetched, 1 unavailable", out)

    def test_detail_no_match_invalid_params_and_timestamp_mismatch(self):
        a, b, c = (_Ann("2026-09-10 10:00:00", "NOMATCH", serial="1"),
                   _Ann("2026-09-11 10:00:00", "BADPARAM", serial="1"),
                   _Ann("2026-09-12 10:00:00", "MISMATCH", serial="1", text="不該出現"))
        self.serve(a, b, c)
        self.mops.detail_override[(a.enter, a.serial)] = _NO_MATCH
        self.mops.detail_override[(b.enter, b.serial)] = _BAD_PARAMS
        shifted = c.detail_result()
        shifted["data"][0][2] = "11:11:11"
        self.mops.detail_override[(c.enter, c.serial)] = _ok(shifted)
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("detail unavailable: MOPS returned no matching detail", out)
        self.assertIn("detail unavailable: MOPS t05st01_detail returned code 500: 傳入參數異常", out)
        self.assertIn("detail unavailable: detail timestamp does not match the list", out)
        self.assertNotIn("不該出現", out)

    def test_row_naming_another_api_is_never_followed(self):
        ann = _Ann("2026-09-10 10:00:00", "EVIL")
        self.serve(ann)
        original = ann.list_row
        ann.list_row = lambda labels=_LIST_TITLES: [
            {"parameters": {}, "apiName": "some_other_api"} if v == original(labels)[5] else v
            for v in original(labels)
        ]
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertEqual(self.mops.detail_calls(), [])
        self.assertIn("detail unavailable: no usable detail reference", out)

    def test_detail_fields_and_event_date_are_separate_from_timestamp(self):
        ann = _Ann("2026-09-11 18:03:01", "董事會決議", clause="第51款", fact="2026-09-10",
                   text="一、核准財報\n二、配息")
        self.serve(ann)
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("| 2026-09-11 | 18:03:01 | 2330 | 台積電 | 上市公司 | 董事會決議 | fetched |", out)
        self.assertIn("- 符合條款: 第51款", out)
        self.assertIn("- 事實發生日 (event date): 2026-09-10", out)
        self.assertIn("  > 一、核准財報", out)
        self.assertIn("  > 二、配息", out)

    def test_long_detail_text_is_clipped_with_a_marker(self):
        self.serve(_Ann("2026-09-11 10:00:00", "長", text="字" * 2500))
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("[truncated by adapter: 500 more characters]", out)

    def test_table_cells_escape_pipes_and_newlines(self):
        self.serve(_Ann("2026-09-11 10:00:00", "主旨|含\n換行"))
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("| 主旨\\|含 換行 |", out)


# --- the four states -------------------------------------------------------------------

class StateTests(_Base):
    def test_no_announcements_is_a_result_not_a_failure(self):
        self.serve()
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("## Result: no material announcements found", out)
        self.assertIn("board could not be confirmed", out)
        self.assertIn(ma.WITHHELD_SAME_DAY_NOTICE, out)   # still stated
        self.assertNotIn("unavailable", out.lower().replace("detail unavailable", ""))

    def test_list_unavailable_raises_and_router_degrades(self):
        self.mops.list_override = MopsUnavailableError("MOPS fetch failed (ConnectTimeout)")
        with self.assertRaises(MopsUnavailableError):
            self.run_tool("2330.TW", "2026-09-15")
        out = interface.route_to_vendor("get_material_announcements", "2330.TW", "2026-09-15", 30)
        self.assertTrue(out.startswith("DATA_UNAVAILABLE:"), out)
        self.assertIn("do not fabricate", out)

    def test_non_taiwan_via_router_is_no_data_without_request(self):
        out = interface.route_to_vendor("get_material_announcements", "AAPL", "2026-09-15", 30)
        self.assertTrue(out.startswith("NO_DATA_AVAILABLE:"), out)
        self.assertEqual(self.mops.calls, [])

    def test_router_registration(self):
        self.assertEqual(interface.get_category_for_method("get_material_announcements"),
                         "taiwan_market_data")
        self.assertIn("taiwan_market_data", interface.OPTIONAL_CATEGORIES)
        self.assertEqual(list(interface.VENDOR_METHODS["get_material_announcements"]), ["mops"])


# --- list fail-closed ---------------------------------------------------------------------

class ListFailClosedTests(_Base):
    def _fails(self, msg_part):
        with self.assertRaises(MopsUnavailableError) as ctx:
            self.run_tool("2330.TW", "2026-09-15")
        self.assertIn(msg_part, str(ctx.exception))

    def _result(self, **changes):
        ann = _Ann("2026-09-10 10:00:00", "x")
        base = {"marketName": "上市公司", "companyId": "2330",
                "titles": _titles(_LIST_TITLES), "data": [ann.list_row()]}
        return _ok(base | changes)

    def test_invalid_parameter_response(self):
        self.mops.list_override = _BAD_PARAMS
        self._fails("returned code 500: 傳入參數異常")

    def test_unexpected_code(self):
        self.mops.list_override = {"code": 302, "message": "moved", "result": None}
        self._fails("returned code 302")

    def test_code_200_without_result(self):
        self.mops.list_override = {"code": 200, "message": "查詢成功", "result": None}
        self._fails("without a result object")

    def test_missing_titles(self):
        env = self._result()
        del env["result"]["titles"]
        self.mops.list_override = env
        self._fails("missing titles")

    def test_missing_required_title(self):
        for label in ("公司代號", "公司名稱", "發言日期", "發言時間", "主旨"):
            with self.subTest(label=label):
                labels = [t for t in _LIST_TITLES if t != label]
                ann = _Ann("2026-09-10 10:00:00", "x")
                self.mops.list_override = self._result(
                    titles=_titles(labels), data=[ann.list_row(labels)])
                self._fails(f"required title {label!r} not found")

    def test_duplicate_required_title(self):
        labels = _LIST_TITLES + ["主旨"]
        ann = _Ann("2026-09-10 10:00:00", "x")
        self.mops.list_override = self._result(titles=_titles(labels), data=[ann.list_row(labels)])
        self._fails("'主旨' appears 2 times")

    def test_unknown_extra_and_reordered_titles_are_allowed(self):
        labels = ["主旨", "市場別", "發言時間", "公司代號", "詳細資料", "公司名稱", "發言日期"]
        self.serve(_Ann("2026-09-10 10:00:00", "REORDERED"))
        self.mops.list_titles = labels
        out = self.run_tool("2330.TW", "2026-09-15")
        self.assertIn("| 2026-09-10 | 10:00:00 | 2330 | 台積電 | 上市公司 | REORDERED | fetched |", out)

    def test_title_with_sub_columns(self):
        titles = _titles(_LIST_TITLES)
        titles[4]["sub"] = ["a", "b"]
        self.mops.list_override = self._result(titles=titles)
        self._fails("has sub-columns")

    def test_row_width_mismatch(self):
        env = self._result()
        env["result"]["data"][0] = env["result"]["data"][0][:-1]
        self.mops.list_override = env
        self._fails("row width does not match")

    def test_unparseable_timestamp(self):
        env = self._result()
        env["result"]["data"][0][2] = "115-09-10"
        self.mops.list_override = env
        self._fails("unparseable timestamp")

    def test_data_not_a_list_and_company_mismatch(self):
        self.mops.list_override = self._result(data={"rows": []})
        self._fails("data is not a list")
        self.mops.list_override = self._result(companyId="2303")
        self._fails("companyId '2303' does not match '2330'")

    def test_row_for_another_company(self):
        env = self._result()
        env["result"]["data"][0][0] = "2303"
        self.mops.list_override = env
        self._fails("row for company '2303'")


# --- HTTP layer (real _post_json, mocked requests) --------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, body=b"", headers=None, chunk=64 * 1024):
        self.status_code, self.headers, self._body, self._chunk = status_code, headers or {}, body, chunk
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


@pytest.mark.unit
class HttpTests(unittest.TestCase):
    def _fails(self, msg_part, **kw):
        with mock.patch.object(bounded_http.requests, "post", **kw), \
                self.assertRaises(MopsUnavailableError) as ctx:
            _REAL_POST_JSON("t05st01", {"companyId": "2330"})
        self.assertIn(msg_part, str(ctx.exception))

    def test_transport_errors(self):
        for exc, name in ((requests.Timeout("slow"), "Timeout"),
                          (requests.exceptions.SSLError("cert"), "SSLError"),
                          (requests.ConnectionError("reset"), "ConnectionError")):
            with self.subTest(name=name):
                self._fails(name, side_effect=exc)

    def test_http_error_status(self):
        for status in (403, 500, 503):
            with self.subTest(status=status):
                self._fails(f"HTTP {status}", return_value=_FakeResponse(status, b"{}"))

    def test_malformed_and_non_object_json(self):
        self._fails("malformed JSON", return_value=_FakeResponse(body=b"<html>busy</html>"))
        self._fails("unexpected JSON shape", return_value=_FakeResponse(body=b"[1, 2]"))

    def test_oversized_body(self):
        resp = _FakeResponse(body=b"x" * (bounded_http.DEFAULT_MAX_BODY_BYTES * 2))
        self._fails("exceeded", return_value=resp)
        self.assertLessEqual(resp.bytes_served, bounded_http.DEFAULT_MAX_BODY_BYTES + bounded_http.CHUNK_BYTES)
        self._fails("exceeded", return_value=_FakeResponse(
            body=b"{}", headers={"Content-Length": str(bounded_http.DEFAULT_MAX_BODY_BYTES + 1)}))

    def test_request_is_json_post_bounded_and_verified(self):
        resp = _FakeResponse(body=b'{"code": 200, "result": {}}')
        with mock.patch.object(bounded_http.requests, "post", return_value=resp) as post:
            self.assertEqual(_REAL_POST_JSON("t05st01", {"companyId": "2330"})["code"], 200)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://mops.twse.com.tw/mops/api/t05st01")
        self.assertEqual(kwargs["json"], {"companyId": "2330"})
        self.assertEqual(kwargs["timeout"], bounded_http.DEFAULT_TIMEOUT_SECONDS)
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["headers"]["User-Agent"], bounded_http.DEFAULT_USER_AGENT)
        self.assertNotIn("verify", kwargs)

    def test_requests_are_spaced(self):
        slept = []
        fake = _FakeMops([_Ann("2026-09-10 10:00:00", "x")])
        with mock.patch.object(ma, "_post_json", fake), \
                mock.patch.object(ma, "_now", lambda: _taipei("2026-09-20 12:00:00")), \
                mock.patch.object(ma, "_sleep", slept.append):
            ma.get_material_announcements("2330.TW", "2026-09-15")
        self.assertEqual(len(slept), len(fake.calls) - 1)
        self.assertTrue(all(s == ma._REQUEST_SPACING_SECONDS for s in slept))


# --- News Analyst wiring -------------------------------------------------------------------------

def _run_news_analyst(ticker):
    from tradingagents.agents.analysts.news_analyst import create_news_analyst
    captured = {}

    def bind_tools(tools):
        captured["tools"] = [t.name for t in tools]

        def run(prompt_value):
            captured["prompt"] = prompt_value.to_string()
            return AIMessage(content="report")
        return run

    llm = mock.MagicMock()
    llm.bind_tools.side_effect = bind_tools
    out = create_news_analyst(llm)({
        "company_of_interest": ticker, "trade_date": "2026-09-15",
        "asset_type": "stock", "messages": [],
    })
    assert out["news_report"] == "report"
    return captured


_DEFAULT_NEWS_TOOLS = ["get_news", "get_global_news", "get_macro_indicators", "get_prediction_markets"]


@pytest.mark.unit
class NewsAnalystWiringTests(unittest.TestCase):
    def test_taiwan_ticker_gets_announcements_tool_and_guidance(self):
        for ticker in ("2330.TW", "6488.TWO"):
            with self.subTest(ticker=ticker):
                c = _run_news_analyst(ticker)
                self.assertEqual(c["tools"], _DEFAULT_NEWS_TOOLS + ["get_material_announcements"])
                self.assertIn("official material announcements", c["prompt"])
                self.assertIn("事實發生日", c["prompt"])
                self.assertIn("Do not treat withheld, unavailable, or missing announcement data", c["prompt"])
                self.assertIn("not automatically bullish or bearish", c["prompt"])

    def test_default_market_is_unchanged(self):
        for ticker in ("AAPL", "NVDA", "7203.T"):
            with self.subTest(ticker=ticker):
                c = _run_news_analyst(ticker)
                self.assertEqual(c["tools"], _DEFAULT_NEWS_TOOLS)
                self.assertNotIn("material announcements", c["prompt"])

    def test_news_tool_node_can_execute_it(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        nodes = TradingAgentsGraph._create_tool_nodes(None)
        self.assertIn("get_material_announcements", nodes["news"].tools_by_name)
        for name in ("get_news", "get_global_news", "get_insider_transactions",
                     "get_macro_indicators", "get_prediction_markets"):
            self.assertIn(name, nodes["news"].tools_by_name)
