"""MOPS adapter bindings: both MOPS adapters use the MOPS error type and the
MOPS wrappers over ``bounded_http``, and take the Taiwan clock, ticker parser
and label normalizer from ``taiwan_common`` rather than from ``mops_common``."""

from __future__ import annotations

import unittest
from unittest import mock

import pytest

from tradingagents.dataflows import (
    bounded_http,
    mops,
    mops_announcements,
    mops_common,
    taiwan_common,
)
from tradingagents.dataflows.errors import NoMarketDataError


@pytest.mark.unit
class SharedBindingTests(unittest.TestCase):
    def test_both_adapters_use_the_shared_clock(self):
        self.assertIs(mops._now, taiwan_common.taipei_now)
        self.assertIs(mops_announcements._now, taiwan_common.taipei_now)

    def test_both_adapters_use_the_shared_error_type(self):
        self.assertIs(mops.MopsUnavailableError, mops_common.MopsUnavailableError)
        self.assertIs(mops_announcements.MopsUnavailableError, mops_common.MopsUnavailableError)
        self.assertTrue(issubclass(mops_announcements.MopsBoardMismatchError,
                                   mops_common.MopsUnavailableError))

    def test_both_adapters_use_the_shared_transport(self):
        self.assertIs(mops._http_get, mops_common.http_get)
        with mock.patch.object(mops_announcements, "http_post_json",
                               return_value={"code": 200}) as post:
            mops_announcements._post_json("t05st01", {"companyId": "2330"})
        post.assert_called_once_with(
            "https://mops.twse.com.tw/mops/api/t05st01", {"companyId": "2330"},
            timeout=bounded_http.DEFAULT_TIMEOUT_SECONDS,
        )

    def test_mops_common_does_not_re_export_generic_helpers(self):
        for name in ("taipei_now", "TAIPEI", "split_taiwan_ticker", "norm_header",
                     "BOARD_BY_SUFFIX", "USER_AGENT", "TIMEOUT_SECONDS", "MAX_BODY_BYTES",
                     "CHUNK_BYTES", "requests"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(mops_common, name))

    def test_board_market_names_cover_every_board(self):
        self.assertEqual(set(taiwan_common.BOARD_BY_SUFFIX.values()),
                         set(mops_common.BOARD_MARKET_NAME))

    def test_no_adapter_keeps_a_private_copy(self):
        for module in (mops, mops_announcements):
            for name in ("_split_ticker", "_norm_header", "_UA", "_TIMEOUT", "_MAX_BODY_BYTES",
                         "_CHUNK_BYTES", "_BOARD_BY_SUFFIX", "_too_large", "_TAIPEI",
                         "_BOARD_MARKET_NAME"):
                with self.subTest(module=module.__name__, name=name):
                    self.assertFalse(hasattr(module, name))


@pytest.mark.unit
class ClockTests(unittest.TestCase):
    def test_monthly_revenue_today_follows_the_taiwan_clock(self):
        # 2026-09-15 00:30 in Taiwan is still 2026-09-14 in UTC. With the Taiwan
        # clock, curr_date 2026-09-14 is a historical run: August revenue (public
        # only from 09-16) is withheld and only July is requested. A UTC or
        # system-date clock would call it live and request August too.
        taiwan_after_midnight = taiwan_common.taipei_now().replace(
            year=2026, month=9, day=15, hour=0, minute=30, second=0, microsecond=0)
        with mock.patch.object(mops, "_now", lambda: taiwan_after_midnight), \
                mock.patch.object(mops, "_fetch_table") as fetch:
            fetch.return_value = mops._ParsedTable(unit="千元", report_date=None, rows={})
            with self.assertRaises(NoMarketDataError):
                mops.get_monthly_revenue("2330.TW", "2026-09-14", look_back_months=2)
        fetch.assert_called_once_with("sii", 2026, 7)
