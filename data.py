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


# Headline indices for the returns table (local-currency price returns —
# this is the convention for the headline returns panel).
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

# RoW indices for the indexed-to-100 chart. SPX overlaid as reference.
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

# FX rates used to convert each RoW local-currency price into USD.
# Convention: tickers are USDxxx=X (= units of local currency per 1 USD),
# so USD_price = local_price / fx_rate.
ROW_FX = {
    "SPX (US)":      None,         # already USD
    "TAIEX":         "TWD=X",      # USD/TWD
    "CSI 300":       "CNY=X",      # USD/CNY
    "KOSPI":         "KRW=X",      # USD/KRW
    "NIFTY 50":      "INR=X",      # USD/INR
    "IBOVESPA":      "BRL=X",      # USD/BRL
    "TASI":          "SAR=X",      # USD/SAR
    "JSE All Share": "ZAR=X",      # USD/ZAR
    "BMV IPC":       "MXN=X",      # USD/MXN
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
    """1W / 1M / 3M / YTD price returns (LOCAL currency)."""
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


def _to_usd(local_price: pd.Series, fx_usd_per_local_inv: pd.Series) -> pd.Series:
    """Convert a local-currency price series to USD.

    `fx_usd_per_local_inv` is the USD/xxx series (units of local currency
    per 1 USD), i.e. the yfinance xxx=X quote. So USD_price = local / fx.
    FX is forward-filled onto the equity calendar to handle different
    trading hours / holidays cleanly.
    """
    if len(fx_usd_per_local_inv) == 0:
        return pd.Series(dtype=float)
    joined_idx = local_price.index.union(fx_usd_per_local_inv.index).sort_values()
    fx_aligned = (
        fx_usd_per_local_inv.reindex(joined_idx)
        .ffill()
        .reindex(local_price.index)
    )
    usd = local_price / fx_aligned
    return usd.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def get_row_prices(years: int = 10) -> pd.DataFrame:
    """RoW equity indices converted to USD terms, aligned on a business-day grid.

    SPX is left as-is (already USD). Every other index is divided by its
    matching USD/xxx FX series so the resulting line measures what a USD
    investor actually earned (price + FX). If an FX series fails to load,
    that index is dropped from the chart rather than shown in mixed units.
    """
    start = (datetime.now() - pd.DateOffset(years=years)).date().isoformat()

    out = {}
    for name, ticker in ROW_MARKETS.items():
        price = get_yf_series(ticker, start=start)
        if len(price) == 0:
            continue

        fx_ticker = ROW_FX.get(name)
        if fx_ticker is None:
            out[name] = price  # already USD
            continue

        fx = get_yf_series(fx_ticker, start=start)
        if len(fx) == 0:
            # FX missing -> skip rather than mix currencies
            continue

        usd_price = _to_usd(price, fx)
        if len(usd_price) > 0:
            out[name] = usd_price

    if not out:
        return pd.DataFrame()

    df = pd.concat(out, axis=1).sort_index()
    df.columns = list(out.keys())
    bidx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(bidx).ffill()
    return df
