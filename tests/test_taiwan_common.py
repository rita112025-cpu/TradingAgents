"""Taiwan-market helpers (``taiwan_common``): clock, board mapping, ticker
parsing and label normalization, independent of any data source."""

from __future__ import annotations

import inspect
import re
import unittest
from datetime import date, timedelta

import pytest

from tradingagents.dataflows import taiwan_common
from tradingagents.dataflows.errors import NoMarketDataError


@pytest.mark.unit
class ClockTests(unittest.TestCase):
    def test_taipei_now_is_utc_plus_eight(self):
        self.assertEqual(taiwan_common.taipei_now().utcoffset(), timedelta(hours=8))


@pytest.mark.unit
class TickerParsingTests(unittest.TestCase):
    def test_board_and_code(self):
        cases = {"2330.TW": ("2330", "sii"), "2330.tw": ("2330", "sii"),
                 "6488.TWO": ("6488", "otc"), " 5274.two ": ("5274", "otc")}
        for ticker, expected in cases.items():
            with self.subTest(ticker=ticker):
                self.assertEqual(taiwan_common.split_taiwan_ticker(ticker, "X"), expected)

    def test_refusals_name_the_source_and_are_unqueried(self):
        for ticker in ("AAPL", "7203.T", "0700.HK", "", "ABCD.TW"):
            with self.subTest(ticker=ticker), self.assertRaises(NoMarketDataError) as ctx:
                taiwan_common.split_taiwan_ticker(ticker, "Some source")
            self.assertIn("not queried", str(ctx.exception))
        with self.assertRaises(NoMarketDataError) as ctx:
            taiwan_common.split_taiwan_ticker("AAPL", "TWSE institutional flows")
        self.assertIn("TWSE institutional flows is only available", str(ctx.exception))
        with self.assertRaises(NoMarketDataError) as ctx:
            taiwan_common.split_taiwan_ticker("ABCD.TW", "X")
        self.assertIn("not a numeric Taiwan company code", str(ctx.exception))

    def test_board_codes(self):
        self.assertEqual(taiwan_common.BOARD_BY_SUFFIX, {".TW": "sii", ".TWO": "otc"})


@pytest.mark.unit
class HeaderNormalizationTests(unittest.TestCase):
    def test_exact_after_nfkc_and_whitespace_removal(self):
        self.assertEqual(taiwan_common.norm_header(" 公司　代號\n"), "公司代號")
        self.assertEqual(taiwan_common.norm_header("前期比較增減（％）"), "前期比較增減(%)")
        self.assertNotEqual(taiwan_common.norm_header("當月營收(千元)"), "當月營收")


@pytest.mark.unit
def test_module_does_not_depend_on_a_data_source():
    # Market helpers must stay source-agnostic: within the package they may only
    # use the error types and the market profile, never a data-source adapter.
    source = inspect.getsource(taiwan_common)
    internal = sorted(set(re.findall(r"^from \.(\w+) import", source, flags=re.MULTILINE)))
    assert internal == ["errors", "market_profiles"], internal


@pytest.mark.unit
class ResolveAsOfTests(unittest.TestCase):
    TODAY = date(2026, 9, 15)

    def test_past_is_historical_and_today_is_live(self):
        self.assertEqual(taiwan_common.resolve_as_of("2026-09-14", self.TODAY), (date(2026, 9, 14), False))
        self.assertEqual(taiwan_common.resolve_as_of("2026-09-15", self.TODAY), (date(2026, 9, 15), True))

    def test_future_date_is_refused(self):
        for curr_date in ("2026-09-16", "2026-10-01", "2030-01-01"):
            with self.subTest(curr_date=curr_date), self.assertRaises(ValueError) as ctx:
                taiwan_common.resolve_as_of(curr_date, self.TODAY)
            self.assertIn("future dates are not supported", str(ctx.exception))

    def test_malformed_date_is_refused(self):
        for curr_date in ("2026/09/14", "20260914", "", None):
            with self.subTest(curr_date=curr_date), self.assertRaises(ValueError) as ctx:
                taiwan_common.resolve_as_of(curr_date, self.TODAY)
            self.assertIn("curr_date must be yyyy-mm-dd", str(ctx.exception))
