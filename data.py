"""Data fetching: yfinance (equities) + FRED (macro) + Polygon.io (FX)."""
from __future__ import annotations
from datetime import datetime
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from fredapi import Fred


# ---------------------------------------------------------------------------
# API keys. Prefer Streamlit secrets, fall back to embedded defaults so the
# app keeps running even if .streamlit/secrets.toml isn't set up.
# ---------------------------------------------------------------------------
_DEFAULT_FRED_KEY    = "7abe0fefb07c9a7fd7a310473e0de4eb"
_DEFAULT_POLYGON_KEY = "bLpGMPIw63rhldGByrLngehm8BpLlCr5"


def _fred_key() -> str:
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        return _DEFAULT_FRED_KEY


def _polygon_key() -> str:
    try:
        return st.secrets["POLYGON_API_KEY"]
    except Exception:
        return _DEFAULT_POLYGON_KEY


@st.cache_resource(show_spinner=False)
def _fred_client() -> Fred:
    return Fred(api_key=_fred_key())


# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------
# Section 1 headline table (intentionally LOCAL currency).
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

# Correlation panel.
CORR_ASSETS = {
    "WTI Crude":        {"source": "yf",   "ticker": "CL=F"},
    "SPX":              {"source": "yf",   "ticker": "^GSPC"},
    "ACWI":             {"source": "yf",   "ticker": "ACWI"},
    "Gold":             {"source": "yf",   "ticker": "GC=F"},
    "10Y UST (Price)":  {"source": "yf",   "ticker": "ZN=F"},
    "DXY (Broad)":      {"source": "fred", "ticker": "DTWEXBGS"},
}

# Section 4: RoW indices that get USD-converted before indexing.
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

# FX pairs are USD-base (USDxxx = units of local currency per 1 USD), so
# the USD-denominated price = local_price / fx. SPX needs no conversion.
ROW_FX = {
    "SPX (US)":      None,
    "TAIEX":         "USDTWD",
    "CSI 300":       "USDCNY",
    "KOSPI":         "USDKRW",
    "NIFTY 50":      "USDINR",
    "IBOVESPA":      "USDBRL",
    "TASI":          "USDSAR",
    "JSE All Share": "USDZAR",
    "BMV IPC":       "USDMXN",
}


# ---------------------------------------------------------------------------
# yfinance helpers (equities + FX fallback)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Polygon FX
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_polygon_fx(pair: str, start: str, end: str | None = None) -> pd.Series:
    """Daily FX close from Polygon.io.

    pair: e.g. 'USDKRW'  (-> Polygon ticker 'C:USDKRW')
    Returns a Series indexed by date, values in units of the quote currency
    per 1 unit of the base currency. For 'USDKRW' that means KRW per USD.
    """
    if end is None:
        end = pd.Timestamp.today().date().isoformat()

    url = f"https://api.polygon.io/v2/aggs/ticker/C:{pair}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort":     "asc",
        "limit":    50000,
        "apiKey":   _polygon_key(),
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return pd.Series(dtype=float, name=pair)

    results = payload.get("results") or []
    if not results:
        return pd.Series(dtype=float, name=pair)

    idx = pd.to_datetime([row["t"] for row in results], unit="ms").normalize()
    vals = [row["c"] for row in results]
    s = pd.Series(vals, index=idx, name=pair).dropna()
    # Polygon FX runs continuously incl. weekends — that's fine; we ffill
    # onto the equity calendar downstream.
    return s


def _get_fx_with_fallback(pair: str, start: str) -> pd.Series:
    """Try Polygon first, then yfinance, so a single broken feed can't
    silently drop a whole index from the chart."""
    s = get_polygon_fx(pair, start=start)
    if len(s) > 0:
        return s
    # yfinance fallback. yfinance uses xxx=X (short form) for USD/xxx.
    yf_ticker = pair[3:] + "=X"   # 'USDKRW' -> 'KRW=X'
    s = get_yf_series(yf_ticker, start=start)
    if len(s) > 0:
        return s
    # last resort: long form
    return get_yf_series(pair + "=X", start=start)


# ---------------------------------------------------------------------------
# Section-specific aggregators
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_returns_table() -> pd.DataFrame:
    """1W / 1M / 3M / YTD price returns in LOCAL currency."""
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


def _to_usd(local_price: pd.Series, fx_usd_base: pd.Series) -> pd.Series:
    """Convert local-currency price -> USD using fx = USD/xxx (xxx per 1 USD).
    FX is reindexed onto the equity calendar and forward-filled so non-overlapping
    holidays don't punch holes in the output."""
    if len(fx_usd_base) == 0:
        return pd.Series(dtype=float)
    joined_idx = local_price.index.union(fx_usd_base.index).sort_values()
    fx_aligned = (
        fx_usd_base.reindex(joined_idx)
        .ffill()
        .reindex(local_price.index)
    )
    usd = local_price / fx_aligned
    return usd.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def get_row_prices(years: int = 10) -> pd.DataFrame:
    """RoW equity indices converted to USD terms, business-day aligned.

    FX is sourced from Polygon first, then yfinance as fallback, so a single
    feed outage can't drop an index from the chart. If absolutely no FX feed
    can be obtained for a given pair, the index is shown in its local
    currency so it never disappears from the chart — and labelled as such.
    """
    start = (datetime.now() - pd.DateOffset(years=years)).date().isoformat()

    out = {}
    note_local_ccy = []   # indices we had to leave in local currency

    for name, ticker in ROW_MARKETS.items():
        price = get_yf_series(ticker, start=start)
        if len(price) == 0:
            continue

        fx_pair = ROW_FX.get(name)
        if fx_pair is None:
            out[name] = price                # SPX already USD
            continue

        fx = _get_fx_with_fallback(fx_pair, start=start)

        if len(fx) == 0:
            # Last-resort: keep the index visible in local currency. This
            # is mildly misleading but better than silently dropping a
            # major market like KOSPI from the dashboard.
            out[f"{name} (local ccy)"] = price
            note_local_ccy.append(name)
            continue

        usd_price = _to_usd(price, fx)
        if len(usd_price) > 0:
            out[name] = usd_price
        else:
            out[f"{name} (local ccy)"] = price
            note_local_ccy.append(name)

    if not out:
        return pd.DataFrame()

    df = pd.concat(out, axis=1).sort_index()
    df.columns = list(out.keys())
    bidx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(bidx).ffill()
    df.attrs["local_ccy_fallback"] = note_local_ccy
    return df
