from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    return route_to_vendor("get_stock_data", symbol, start_date, end_date)


@tool
def get_institutional_flows(
    ticker: Annotated[str, "Taiwan-listed ticker symbol with exchange suffix, e.g. 2330.TW or 6488.TWO"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    look_back_trading_days: Annotated[int, "number of completed trading days to return (max 20)"] = 5,
) -> str:
    """
    Retrieve official daily institutional investor flows (三大法人買賣超: foreign
    investors, investment trusts, dealers) for a Taiwan-listed company from
    TWSE (.TW) or TPEx (.TWO), in shares, point-in-time as of curr_date.
    Args:
        ticker (str): Taiwan-listed ticker symbol, e.g. 2330.TW
        curr_date (str): Current date you are trading at, yyyy-mm-dd
        look_back_trading_days (int): Completed trading days to return (default 5, max 20)
    Returns:
        str: Per-day net and buy/sell tables by investor group
    """
    return route_to_vendor("get_institutional_flows", ticker, curr_date, look_back_trading_days)
