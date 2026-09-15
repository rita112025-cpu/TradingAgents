"""Taiwan-market helpers shared by every Taiwan data adapter.

Nothing here is tied to a data source: the Taiwan clock, the Yahoo-suffix
board mapping, ticker parsing through the market profile, and the CJK label
normalization used for exact header matching.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta, timezone

from .errors import NoMarketDataError
from .market_profiles import MARKET_TAIWAN, resolve_market

# Taiwan has used UTC+8 without daylight saving since 1979; a fixed offset
# avoids depending on a tz database (absent on Windows without tzdata).
TAIPEI = timezone(timedelta(hours=8), "Asia/Taipei")

# Yahoo suffix -> board code (``sii`` TWSE listed, ``otc`` TPEx listed).
# Taiwan detection itself is the market profile's job; this only maps a
# known-Taiwan suffix to its board.
BOARD_BY_SUFFIX = {".TW": "sii", ".TWO": "otc"}


def taipei_now() -> datetime:
    """Current Taiwan time, the single "now" for every Taiwan adapter.

    Adapters bind it as their own ``_now`` so tests patch one seam per module.
    """
    return datetime.now(TAIPEI)


def resolve_as_of(curr_date: str, today: date) -> tuple[date, bool]:
    """Parse ``curr_date`` and classify the run against Taiwan ``today``.

    Returns ``(as_of, live)``: live when ``as_of`` is today, historical when it
    is earlier. A malformed date or a date after today raises ``ValueError``:
    data "as of" a day that has not arrived does not exist, and treating it as
    live would label today's data with a later date.
    """
    try:
        as_of = datetime.strptime(curr_date, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"curr_date must be yyyy-mm-dd, got {curr_date!r}") from exc
    if as_of > today:
        raise ValueError(
            f"curr_date {curr_date} is after today ({today}, Taiwan time); "
            "future dates are not supported"
        )
    return as_of, as_of == today


def norm_header(label: str) -> str:
    """Minimal header normalization: NFKC (full-width -> half-width, NBSP ->
    space) and removal of all whitespace. The labels are CJK, where whitespace
    only comes from formatting. No fuzzy matching: labels compare exactly.
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", label))


def split_taiwan_ticker(ticker: str, source: str) -> tuple[str, str]:
    """``2330.TW`` -> (``"2330"``, ``"sii"``); ``6488.TWO`` -> (``"6488"``, ``"otc"``).

    Raises :class:`NoMarketDataError` naming ``source`` for anything that is
    not a Taiwan-listed ticker with a numeric company code, so callers can
    refuse before making any request.
    """
    symbol = (ticker or "").strip().upper()
    if resolve_market(symbol) != MARKET_TAIWAN:
        raise NoMarketDataError(
            ticker, detail=f"{source} is only available for Taiwan-listed tickers "
                           "(.TW / .TWO); not queried",
        )
    for suffix, board in BOARD_BY_SUFFIX.items():
        if symbol.endswith(suffix):
            code = symbol[: -len(suffix)]
            if re.fullmatch(r"\d{4,6}", code):
                return code, board
            raise NoMarketDataError(
                ticker, detail=f"{code!r} is not a numeric Taiwan company code; not queried",
            )
    raise NoMarketDataError(
        ticker, detail="Taiwan ticker without a board suffix (.TW/.TWO); not queried",
    )
