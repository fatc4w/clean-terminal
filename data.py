"""Data fetching layer: yfinance + FRED, all results cached per-session."""
from __future__ import annotations
from datetime import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
from fredapi import Fred


_DEFAULT_FRED_KEY = "7abe0fefb07c9a7fd7a310473e0de4eb"


def _fred_key() -> str:
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        return _DEFAULT_FRED_KEY


@st.cache_resource(show_spinner=False)
def _fred_client() -> Fred:
    return Fred(api_key=_fred_key())


# Headline indices for the returns table — now includes RoW markets too.
MARKETS = {
    "SPX":           "^GSPC",
    "NDQ":           "^NDX",
    "KOSPI":         "^KS11",
    "Nikkei 225":    "^N225",
    "CSI 300":       "000300.SS",
    "STOXX 600":     "^STOXX",
    "STI":           "^STI",
    "TAIEX":         "^TWII",
    "NIFTY 50":      "^NSEI",
    "IBOVESPA":      "^BVSP",
    "TASI":          "^TASI.SR",
    "JSE All Share": "^J203.JO",
    "BMV IPC":       "^MXX",
}

CORR_ASSETS = {
    "WTI Crude":        {"source": "yf",   "ticker": "CL=F"},
    "SPX":              {"source": "yf",   "ticker": "^GSPC"},
    "ACWI":             {"source": "yf",   "ticker": "ACWI"},
    "Gold":             {"source": "yf",   "ticker": "GC=F"},
    "10Y UST (Price)":  {"source": "yf",   "ticker": "ZN=F"},
    "DXY (Broad)":      {"source": "fred", "ticker": "DTWEXBGS"},
}

# RoW indices for the indexed-to-100 chart (SPX overlaid as reference)
ROW_MARKETS = {
    "SPX (US)":      "^GSPC",
    "TAIEX":         "^TWII",
    "CSI 300":       "000300.SS",
    "KOSPI":         "^KS11",
    "NIFTY 50":      "^NSEI",
    "IBOVESPA":      "^BVSP",
    "TASI":          "^TASI.SR",
    "JSE All Share": "^J203.JO",
    "BMV IPC":       "^MXX",
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_yf_series(ticker: str, period: str = "max", start: str | None = None) -> pd.Series:
    try:
        if start is not None:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=False, threads=False)
        else:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=False, threads=False)
    except Exception:
        return pd.Series(dtype=float, name=ticker)

    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)

    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        col = "Adj Close" if "Adj Close" in level0 else "Close"
        s = df[col].iloc[:, 0]
    else:
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = df[col]
    s = s.dropna()
    s.name = ticker
    return s


@st.cache_data(ttl=3600, show_spinner=False)
def get_yf_multi(tickers: list[str], period: str = "max", start: str | None = None) -> pd.DataFrame:
    series = {}
    for t in tickers:
        s = get_yf_series(t, period=period, start=start)
        if len(s) > 0:
            series[t] = s
    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1)
    df.columns = list(series.keys())
    return df.sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def get_fred_series(series_id: str, start: str | None = None) -> pd.Series:
    try:
        s = _fred_client().get_series(series_id, observation_start=start)
    except Exception:
        return pd.Series(dtype=float, name=series_id)
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_returns_table() -> pd.DataFrame:
    """1W / 1M / 3M / YTD percent returns for the headline equity indices."""
    tickers = list(MARKETS.values())
    df = get_yf_multi(tickers, period="2y")
    if df.empty:
        return pd.DataFrame(columns=["1W", "1M", "3M", "YTD"])

    inv = {v: k for k, v in MARKETS.items()}
    df = df.rename(columns=inv)
    ordered = [k for k in MARKETS.keys() if k in df.columns]
    df = df[ordered].ffill().dropna(how="all")

    last_date = df.index[-1]
    last = df.iloc[-1]

    def pct(days: int) -> pd.Series:
        target = last_date - pd.Timedelta(days=days)
        hist = df.loc[df.index <= target]
        base = hist.iloc[-1] if len(hist) else df.iloc[0]
        return (last / base - 1.0) * 100.0

    ytd_start = pd.Timestamp(year=last_date.year, month=1, day=1)
    hist_ytd = df.loc[df.index < ytd_start]
    ytd_base = hist_ytd.iloc[-1] if len(hist_ytd) else df.iloc[0]
    ytd = (last / ytd_base - 1.0) * 100.0

    return pd.DataFrame({
        "1W":  pct(7),
        "1M":  pct(30),
        "3M":  pct(90),
        "YTD": ytd,
    })


@st.cache_data(ttl=3600, show_spinner=False)
def get_corr_assets_prices(years: int = 10) -> pd.DataFrame:
    start = (datetime.now() - pd.DateOffset(years=years)).date().isoformat()
    cols = {}
    for name, cfg in CORR_ASSETS.items():
        if cfg["source"] == "yf":
            s = get_yf_series(cfg["ticker"], start=start)
        else:
            s = get_fred_series(cfg["ticker"], start=start)
        cols[name] = s
    df = pd.concat(cols, axis=1).sort_index()
    df.columns = list(cols.keys())
    bidx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(bidx).ffill()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_row_prices(years: int = 10) -> pd.DataFrame:
    start = (datetime.now() - pd.DateOffset(years=years)).date().isoformat()
    tickers = list(ROW_MARKETS.values())
    df = get_yf_multi(tickers, start=start)
    inv = {v: k for k, v in ROW_MARKETS.items()}
    df = df.rename(columns=inv)
    ordered = [k for k in ROW_MARKETS.keys() if k in df.columns]
    df = df[ordered]
    bidx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(bidx).ffill()
    return df
