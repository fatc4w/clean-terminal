"""Macro Markets Dashboard."""
from __future__ import annotations
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import (
    MARKETS, CORR_ASSETS, ROW_MARKETS,
    get_market_returns_table,
    get_corr_assets_prices,
    get_fred_series,
    get_wei_data,
    get_row_prices,
)
from utils import (
    log_returns,
    ewma_corr_pairwise_series,
    latest_corr_matrix,
    index_to_100,
)


st.set_page_config(
    page_title="Macro Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px;}
        h1 {font-size: 1.9rem; margin-bottom: 0.1rem;}
        h2 {color: #FF6B35; border-bottom: 1px solid #2a2e36; padding-bottom: 0.4rem; margin-top: 2.4rem;}
        h3 {color: #C9CCD0; font-weight: 500;}
        [data-testid="stMetricValue"] {font-size: 1.15rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Macro Markets Dashboard")
st.caption(f"As of {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} • Data: yfinance + FRED + Polygon")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
WIN_STD  = ["1Y", "5Y", "Max"]
WIN_YTD  = ["YTD", "1Y", "5Y", "Max"]
WIN_FULL = ["1W", "1M", "3M", "YTD", "1Y", "5Y", "Max"]
PLOT_BG  = "#0E1117"


def _window_start(end, window, series_list=None):
    if window == "1W":  return end - pd.Timedelta(days=7)
    if window == "1M":  return end - pd.DateOffset(months=1)
    if window == "3M":  return end - pd.DateOffset(months=3)
    if window == "1Y":  return end - pd.DateOffset(years=1)
    if window == "5Y":  return end - pd.DateOffset(years=5)
    if window == "YTD": return pd.Timestamp(year=end.year, month=1, day=1)
    if series_list is not None:
        return min(s.index[0] for s in series_list if len(s) > 0)
    return None


def _line_fig(series_dict, height=380, y_fmt=None, y_title=None):
    fig = go.Figure()
    for name, s in series_dict.items():
        if s is None or len(s) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(width=2),
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        yaxis=dict(title=y_title) if y_title else dict(),
    )
    if y_fmt:
        fig.update_yaxes(tickformat=y_fmt)
    return fig


def _bar_fig(series: pd.Series, height=380, y_title=None, color_pos="#26a69a", color_neg="#ef5350"):
    """Bar chart with green/red colouring for positive/negative values."""
    colors = [color_pos if v >= 0 else color_neg for v in series.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=series.index, y=series.values,
        marker_color=colors,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#555", line_width=1)
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        yaxis=dict(title=y_title) if y_title else dict(),
        bargap=0.15,
    )
    return fig


_NBER_RECESSIONS = [
    ("2008-01-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


def _wei_fig(wei_series: pd.Series, overlays: dict | None = None, height: int = 420) -> go.Figure:
    """Line + fill chart for WEI with positive (blue) / negative (red) shading.

    overlays: optional dict of {label: pd.Series} plotted as additional lines.
    """
    s = wei_series.dropna()
    pos = s.where(s >= 0)
    neg = s.where(s < 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=pos.values,
        mode="none", fill="tozeroy",
        fillcolor="rgba(31,119,180,0.18)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=s.index, y=neg.values,
        mode="none", fill="tozeroy",
        fillcolor="rgba(214,39,40,0.18)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values,
        mode="lines", name="WEI",
        line=dict(color="#1f77b4", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>WEI: %{y:.2f}%<extra></extra>",
    ))
    overlay_colors = ["#FF6B35", "#26a69a", "#AB47BC", "#FFCA28"]
    for i, (name, series) in enumerate((overlays or {}).items()):
        if series is None or len(series) == 0:
            continue
        color = overlay_colors[i % len(overlay_colors)]
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name=name,
            line=dict(color=color, width=1.8, dash="dot"),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name}: %{{y:.2f}}%<extra></extra>",
        ))
    for r_start, r_end in _NBER_RECESSIONS:
        fig.add_vrect(
            x0=r_start, x1=r_end,
            fillcolor="grey", opacity=0.15,
            layer="below", line_width=0,
            annotation_text="Recession", annotation_position="top left",
            annotation_font_size=8, annotation_font_color="#888",
        )
    fig.add_hline(y=0, line_color="#555", line_width=0.8, line_dash="dash")
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        yaxis=dict(title="% (4Q GDP-scaled / YoY)", tickformat=".1f"),
    )
    return fig


def fred_chart_block(series_dict, caption, key_prefix, include_ytd=False,
                     y_fmt=None, y_title=None, height=380):
    if caption:
        st.caption(caption)
    options = WIN_YTD if include_ytd else WIN_STD
    window = st.segmented_control(
        "Window", options=options, default=options[0], key=f"{key_prefix}_win",
    )
    if window is None:
        window = options[0]

    valid = [s for s in series_dict.values() if s is not None and len(s) > 0]
    if not valid:
        st.warning("No data available")
        return
    end = max(s.index[-1] for s in valid)
    start = _window_start(end, window, series_list=valid)

    sliced = {}
    for n, s in series_dict.items():
        if s is None or len(s) == 0:
            continue
        sliced[n] = s.loc[s.index >= start] if start is not None else s
    fig = _line_fig(sliced, height=height, y_fmt=y_fmt, y_title=y_title)
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_chart")


def fred_bar_block(series: pd.Series, caption: str, key_prefix: str,
                   include_ytd=False, y_title=None, height=380):
    """Bar chart block with the same window-selector pattern."""
    if caption:
        st.caption(caption)
    options = WIN_YTD if include_ytd else WIN_STD
    window = st.segmented_control(
        "Window", options=options, default=options[0], key=f"{key_prefix}_win",
    )
    if window is None:
        window = options[0]

    if series is None or len(series) == 0:
        st.warning("No data available")
        return
    end = series.index[-1]
    start = _window_start(end, window, series_list=[series])
    sliced = series.loc[series.index >= start] if start is not None else series

    fig = _bar_fig(sliced, height=height, y_title=y_title)
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_chart")


# ---------------------------------------------------------------------------
# 1. FINANCIAL MARKETS
# ---------------------------------------------------------------------------
st.header("1. Financial Markets")

st.subheader("Global Equity Index Performance (local currency)")
with st.spinner("Fetching equity index data…"):
    returns_table = get_market_returns_table()


def _color_perf(v):
    if pd.isna(v):
        return "color: #666;"
    return f"color: {'#26a69a' if v >= 0 else '#ef5350'}; font-weight: 600;"


if returns_table.empty:
    st.warning("Could not load equity index returns.")
else:
    styled = (
        returns_table.style
        .format("{:+.2f}%", na_rep="–")
        .map(_color_perf)
    )
    n_rows = len(returns_table)
    tbl_height = n_rows * 35 + 38
    st.dataframe(styled, width="stretch", height=tbl_height)


# ---- Correlation Matrix + Pair time-series via dropdowns ----
st.subheader("Cross-Asset Correlation Matrix")
st.caption("EWMA correlation (λ = 0.94) of daily log returns over the last ~10 years.")

with st.spinner("Computing EWMA correlations…"):
    corr_prices = get_corr_assets_prices(years=10)
    rets = log_returns(corr_prices).dropna(how="all").fillna(0.0)
    corr_series_dict = ewma_corr_pairwise_series(rets, lam=0.94)
    asset_list = list(corr_prices.columns)
    corr_matrix = latest_corr_matrix(corr_series_dict, asset_list)

text_labels = [[f"{v:.2f}" for v in row] for row in corr_matrix.values]
heat_fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=asset_list, y=asset_list,
    text=text_labels, texttemplate="%{text}",
    textfont=dict(size=13),
    colorscale=[[0.0, "#ef5350"], [0.5, "#262931"], [1.0, "#26a69a"]],
    zmid=0, zmin=-1, zmax=1,
    hovertemplate="%{y} vs %{x}<br>ρ = %{z:.3f}<extra></extra>",
    colorbar=dict(title="ρ", thickness=14, len=0.8),
))
heat_fig.update_layout(
    height=520, template="plotly_dark",
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(autorange="reversed"),
    plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
)
st.plotly_chart(heat_fig, width="stretch", key="corr_heatmap")

st.markdown("**Pair Correlation Time Series**")
st.caption("Pick any two assets to view the EWMA correlation history between them.")

pcol1, pcol2, pcol3 = st.columns([2, 2, 2])
with pcol1:
    default_a = "SPX" if "SPX" in asset_list else asset_list[0]
    asset_a = st.selectbox(
        "Asset A", asset_list,
        index=asset_list.index(default_a),
        key="pair_a",
    )
with pcol2:
    others = [a for a in asset_list if a != asset_a]
    default_b = "Gold" if "Gold" in others else others[0]
    asset_b = st.selectbox(
        "Asset B", others,
        index=others.index(default_b),
        key="pair_b",
    )
with pcol3:
    pair_win = st.segmented_control(
        "Window", options=["1Y", "5Y", "Max"], default="5Y", key="pair_win",
    )
    if pair_win is None:
        pair_win = "5Y"

_s1 = corr_series_dict.get((asset_a, asset_b))
_s2 = corr_series_dict.get((asset_b, asset_a))
pair_series = _s1 if _s1 is not None else _s2

if pair_series is not None and len(pair_series) > 0:
    end = pair_series.index[-1]
    if pair_win == "1Y":
        p_start = end - pd.DateOffset(years=1)
    elif pair_win == "5Y":
        p_start = end - pd.DateOffset(years=5)
    else:
        p_start = pair_series.index[0]
    sf = pair_series.loc[pair_series.index >= p_start]

    pair_fig = go.Figure()
    pair_fig.add_trace(go.Scatter(
        x=sf.index, y=sf.values, mode="lines",
        line=dict(color="#FF6B35", width=2),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.10)",
        hovertemplate="%{x|%Y-%m-%d}<br>ρ = %{y:.3f}<extra></extra>",
    ))
    pair_fig.add_hline(y=0, line_dash="dash", line_color="#666", line_width=1)
    pair_fig.update_layout(
        height=400, template="plotly_dark",
        yaxis=dict(title=f"ρ  {asset_a} / {asset_b}", range=[-1, 1]),
        xaxis=dict(title=""),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
    )
    st.plotly_chart(pair_fig, width="stretch", key="pair_corr_chart")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current", f"{sf.iloc[-1]:.3f}")
    c2.metric(f"{pair_win} Mean", f"{sf.mean():.3f}")
    c3.metric(f"{pair_win} Min", f"{sf.min():.3f}")
    c4.metric(f"{pair_win} Max", f"{sf.max():.3f}")
else:
    st.warning("No correlation data available for the selected pair.")


# ---------------------------------------------------------------------------
# 2. ECONOMIC CONDITIONS
# ---------------------------------------------------------------------------
st.header("2. Economic Conditions")

with st.spinner("Loading FRED macro series…"):
    y2      = get_fred_series("DGS2",     start="2000-01-01")
    y5      = get_fred_series("DGS5",     start="2000-01-01")
    y10     = get_fred_series("DGS10",    start="2000-01-01")
    y30     = get_fred_series("DGS30",    start="2000-01-01")
    pce     = get_fred_series("PCEPILFE", start="1990-01-01")
    mich    = get_fred_series("MICH",     start="1990-01-01")
    payems  = get_fred_series("PAYEMS",   start="1990-01-01")
    gdp     = get_fred_series("GDP",      start="2005-01-01")

with st.spinner("Loading Dallas Fed WEI…"):
    wei = get_wei_data()

pce_yoy    = (pce.pct_change(periods=12).dropna() * 100) if len(pce) > 12 else pd.Series(dtype=float)
payems_mom = payems.diff(1).dropna() if len(payems) > 1 else pd.Series(dtype=float)
gdp_yoy    = (gdp.pct_change(periods=4).dropna() * 100) if len(gdp) > 4 else pd.Series(dtype=float)

# Row 1: Treasury yields | Core PCE YoY
c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown("**US Treasury Yields**")
    fred_chart_block(
        {"2Y": y2, "5Y": y5, "10Y": y10, "30Y": y30},
        caption="US Treasury Yields (%)",
        key_prefix="ust",
        y_fmt=".2f", y_title="Yield (%)",
        height=380,
    )
with c2:
    st.markdown("**Core PCE YoY**")
    fred_chart_block(
        {"Core PCE YoY (%)": pce_yoy},
        caption="PCEPILFE — Core PCE Price Index, Year-over-Year % Change",
        key_prefix="pce",
        y_fmt=".2f", y_title="YoY (%)",
        height=380,
    )

# Row 2: UMich inflation expectations | Nonfarm payrolls MoM
c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown("**Inflation Expectations**")
    fred_chart_block(
        {"UMich 1Y Inflation Expectation": mich},
        caption="MICH — Univ. of Michigan 1Y Inflation Expectation (%)",
        key_prefix="mich",
        y_fmt=".1f", y_title="%",
        height=380,
    )
with c2:
    st.markdown("**Nonfarm Payrolls — MoM Change**")
    fred_bar_block(
        payems_mom,
        caption="PAYEMS — All Employees, Total Nonfarm: Monthly Change (thousands)",
        key_prefix="payems",
        y_title="Jobs Added (thousands)",
        height=380,
    )

# Row 3: WEI + Nominal GDP YoY overlay (full width)
st.markdown("**Dallas Fed Weekly Economic Index (WEI) vs Nominal GDP Growth YoY**")
st.caption(
    "WEI (blue, weekly) scaled to 4-quarter GDP growth rate — positive = above-trend activity. "
    "GDP YoY (orange, dotted) = nominal GDP (FRED: GDP) year-over-year % change, quarterly. "
    "Grey bands = NBER recessions."
)
if wei is None or len(wei) == 0:
    st.warning("WEI data unavailable — Dallas Fed page may be unreachable.")
else:
    wei_window = st.segmented_control(
        "Window", options=WIN_STD, default=WIN_STD[0], key="wei_win",
    )
    if wei_window is None:
        wei_window = WIN_STD[0]

    wei_end = wei.index[-1]
    wei_start = _window_start(wei_end, wei_window, series_list=[wei])
    wei_sliced = wei.loc[wei.index >= wei_start] if wei_start is not None else wei
    gdp_yoy_sliced = (
        gdp_yoy.loc[gdp_yoy.index >= wei_start]
        if wei_start is not None and len(gdp_yoy) > 0
        else gdp_yoy
    )

    latest_val = wei_sliced.iloc[-1]
    latest_date = wei_sliced.index[-1]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Latest WEI", f"{latest_val:.2f}%",
               help=f"As of {latest_date.strftime('%b %d, %Y')}")
    mc2.metric(f"{wei_window} WEI Mean", f"{wei_sliced.mean():.2f}%")
    if len(gdp_yoy_sliced) > 0:
        mc3.metric("Latest GDP YoY", f"{gdp_yoy_sliced.iloc[-1]:.2f}%")
        mc4.metric(f"{wei_window} GDP YoY Mean", f"{gdp_yoy_sliced.mean():.2f}%")

    st.plotly_chart(
        _wei_fig(wei_sliced, overlays={"GDP YoY (nominal)": gdp_yoy_sliced}),
        width="stretch", key="wei_chart",
    )


# ---------------------------------------------------------------------------
# 3. FINANCIAL CONDITIONS
# ---------------------------------------------------------------------------
st.header("3. Financial Conditions")

with st.spinner("Loading money market data…"):
    sofr     = get_fred_series("SOFR",          start="2018-01-01")
    effr     = get_fred_series("EFFR",          start="2000-01-01")
    discount = get_fred_series("DPCREDIT",      start="2003-01-01")
    onrrp    = get_fred_series("RRPONTSYAWARD", start="2013-01-01")
    iorb     = get_fred_series("IORB",          start="2021-07-29")
    nfci     = get_fred_series("NFCI",          start="1971-01-01")

st.subheader("Money Market Rates")
fred_chart_block(
    {
        "SOFR": sofr,
        "EFFR": effr,
        "Discount Window (Primary Credit)": discount,
        "ON RRP Award Rate": onrrp,
        "IORB": iorb,
    },
    caption="Money Market Rates (%)",
    key_prefix="mmr",
    y_fmt=".2f", y_title="Rate (%)",
    height=420,
)

st.subheader("Money Market Spreads")
if any(len(s) for s in (sofr, effr, onrrp)):
    start_idx = min(s.index.min() for s in (sofr, effr, onrrp) if len(s) > 0)
    daily_idx = pd.date_range(start=start_idx, end=pd.Timestamp.today(), freq="D")
    sofr_d  = sofr.reindex(daily_idx).ffill()
    effr_d  = effr.reindex(daily_idx).ffill()
    onrrp_d = onrrp.reindex(daily_idx).ffill()
    spread_se = ((sofr_d - effr_d) * 100).dropna()
    spread_so = ((sofr_d - onrrp_d) * 100).dropna()
else:
    spread_se = pd.Series(dtype=float)
    spread_so = pd.Series(dtype=float)

fred_chart_block(
    {"SOFR − EFFR (bps)": spread_se, "SOFR − ON RRP (bps)": spread_so},
    caption="Money Market Spreads (basis points)",
    key_prefix="mms",
    y_fmt=".0f", y_title="Spread (bps)",
    height=380,
)

st.subheader("Chicago Fed National Financial Conditions Index")
st.caption("NFCI (positive = tighter than average, negative = looser). "
           "Drag the slider to set your window — defaults to trailing 5 years.")

nfci_valid = nfci.dropna()
if len(nfci_valid) == 0:
    st.warning("No NFCI data available.")
else:
    min_d = nfci_valid.index.min().date()
    max_d = nfci_valid.index.max().date()
    default_start = (nfci_valid.index.max() - pd.DateOffset(years=5)).date()
    if default_start < min_d:
        default_start = min_d
    sel = st.slider(
        "Window",
        min_value=min_d,
        max_value=max_d,
        value=(default_start, max_d),
        format="YYYY-MM-DD",
        key="nfci_slider",
    )
    s_start, s_end = sel
    sliced_nfci = nfci_valid.loc[str(s_start):str(s_end)]
    nfci_fig = _line_fig({"NFCI": sliced_nfci}, height=400, y_fmt=".2f", y_title="Index")
    st.plotly_chart(nfci_fig, width="stretch", key="nfci_chart")


# ---------------------------------------------------------------------------
# 4. US vs ROW  (USD-denominated)
# ---------------------------------------------------------------------------
st.header("4. US vs Rest of World")
st.subheader("Equity Market Performance — USD terms (Indexed to 100)")
st.caption(
    "All RoW indices converted from local currency to USD (Polygon FX, yfinance fallback) "
    "before indexing. Every line measures what a USD-based investor earned (price + FX)."
)

with st.spinner("Loading global equity index data (with FX conversion)…"):
    row_prices = get_row_prices(years=10)

row_window = st.segmented_control(
    "Window", options=WIN_FULL, default="YTD", key="row_win",
)
if row_window is None:
    row_window = "YTD"

if row_prices.empty:
    st.warning("Could not load global equity prices.")
else:
    end = row_prices.index[-1]
    valid_cols = [row_prices[c].dropna() for c in row_prices.columns if len(row_prices[c].dropna()) > 0]
    start = _window_start(end, row_window, series_list=valid_cols)

    sliced = row_prices.loc[row_prices.index >= start].copy().ffill().dropna(how="all")
    indexed = index_to_100(sliced)

    palette = [
        "#42A5F5", "#AB47BC", "#FFCA28", "#26C6DA",
        "#EF5350", "#66BB6A", "#FF7043", "#EC407A",
    ]
    non_spx = [c for c in indexed.columns if c != "SPX (US)"]

    fig = go.Figure()
    for i, name in enumerate(non_spx):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=indexed.index, y=indexed[name],
            name=name, mode="lines",
            line=dict(color=color, width=1.7),
            opacity=0.45,
            hovertemplate=f"<b>{name}</b> (USD)<br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
        ))
    if "SPX (US)" in indexed.columns:
        fig.add_trace(go.Scatter(
            x=indexed.index, y=indexed["SPX (US)"],
            name="SPX (US)", mode="lines",
            line=dict(color="#FAFAFA", width=3.2),
            opacity=1.0,
            hovertemplate="<b>SPX (US)</b><br>%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        height=540, template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
            itemclick="toggleothers", itemdoubleclick="toggle",
        ),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        yaxis=dict(title="USD Indexed Value (start = 100)"),
    )
    st.caption(
        f"USD-indexed to 100 from start of {row_window} window. SPX bold/opaque as US reference. "
        f"Single-click a legend entry to isolate one index; double-click to restore all."
    )
    st.plotly_chart(fig, width="stretch", key="row_chart")

st.subheader("Policy Rates, Overnight Rates & Inflation — Major DM ex-US")
st.caption(
    "Each chart pairs a region's policy rate, its risk-free overnight benchmark, and headline "
    "inflation. Note: Japan's TONA is calculated from uncollateralised overnight call transactions "
    "(same underlying data as the call rate). Australia's AONIA is, by RBA definition, the Cash Rate."
)


def _yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    """YoY % change. periods=12 for monthly, 4 for quarterly."""
    if s is None or len(s) <= periods:
        return pd.Series(dtype=float)
    return (s.pct_change(periods=periods).dropna() * 100.0)


with st.spinner("Loading global rates and inflation…"):
    # --- Euro area: ECB rates + €STR + national CPI indices (-> compute YoY) ---
    ecb_dfr   = get_fred_series("ECBDFR",                start="1999-01-01")  # Deposit Facility
    ecb_mro   = get_fred_series("ECBMRRFR",              start="1999-01-01")  # Main Refi
    estr      = get_fred_series("ECBESTRVOLWGTTRMDMNRT", start="2019-10-01")  # €STR
    de_cpi    = get_fred_series("DEUCPIALLMINMEI",       start="1995-01-01")  # Germany CPI idx
    fr_cpi    = get_fred_series("FRACPIALLMINMEI",       start="1995-01-01")  # France CPI idx
    it_cpi    = get_fred_series("ITACPIALLMINMEI",       start="1995-01-01")  # Italy CPI idx
    es_cpi    = get_fred_series("ESPCPIALLMINMEI",       start="1995-01-01")  # Spain CPI idx
    de_yoy    = _yoy(de_cpi, 12)
    fr_yoy    = _yoy(fr_cpi, 12)
    it_yoy    = _yoy(it_cpi, 12)
    es_yoy    = _yoy(es_cpi, 12)

    # --- UK ---
    uk_bank   = get_fred_series("BOERUKM",               start="1990-01-01")  # BoE Bank Rate
    sonia     = get_fred_series("IUDSOIA",               start="1997-01-01")  # SONIA
    uk_cpi_idx = get_fred_series("GBRCPIALLMINMEI",      start="1990-01-01")  # UK CPI idx
    uk_yoy    = _yoy(uk_cpi_idx, 12)

    # --- Japan ---
    jp_call   = get_fred_series("IRSTCI01JPM156N",       start="1990-01-01")  # Call/TONA proxy
    jp_cpi_idx = get_fred_series("JPNCPIALLMINMEI",      start="1990-01-01")  # Japan CPI idx
    jp_yoy    = _yoy(jp_cpi_idx, 12)

    # --- Australia (CPI is quarterly) ---
    au_cash   = get_fred_series("IRSTCB01AUM156N",       start="1990-01-01")  # RBA Cash Rate / AONIA
    au_cpi_idx = get_fred_series("AUSCPIALLQINMEI",      start="1990-01-01")  # AU CPI idx (Q)
    au_yoy    = _yoy(au_cpi_idx, 4)

r1c1, r1c2 = st.columns(2, gap="medium")
with r1c1:
    st.markdown("**Euro Area — ECB, €STR & National HICP YoY**")
    fred_chart_block(
        {
            "ECB Deposit Facility": ecb_dfr,
            "ECB Main Refi Rate":   ecb_mro,
            "€STR":                 estr,
            "Germany CPI YoY":      de_yoy,
            "France CPI YoY":       fr_yoy,
            "Italy CPI YoY":        it_yoy,
            "Spain CPI YoY":        es_yoy,
        },
        caption="ECB policy rates, €STR overnight benchmark, and CPI YoY for DE/FR/IT/ES (%, OECD MEI)",
        key_prefix="eu_block",
        include_ytd=True,
        y_fmt=".2f", y_title="%",
        height=420,
    )
with r1c2:
    st.markdown("**United Kingdom — Bank Rate, SONIA & CPI YoY**")
    fred_chart_block(
        {
            "BoE Bank Rate":  uk_bank,
            "SONIA":          sonia,
            "UK CPI YoY":     uk_yoy,
        },
        caption="Bank of England Bank Rate, SONIA overnight benchmark, and UK headline CPI YoY (%)",
        key_prefix="uk_block",
        include_ytd=True,
        y_fmt=".2f", y_title="%",
        height=420,
    )

r2c1, r2c2 = st.columns(2, gap="medium")
with r2c1:
    st.markdown("**Japan — Uncollateralised Call Rate / TONA & CPI YoY**")
    fred_chart_block(
        {
            "Uncoll. O/N Call Rate / TONA": jp_call,
            "Japan CPI YoY":                jp_yoy,
        },
        caption=(
            "BoJ uncollateralised overnight call rate (TONA is computed from the same trades), "
            "alongside headline CPI YoY (%). The BoJ policy rate is set as the target for this rate."
        ),
        key_prefix="jp_block",
        include_ytd=True,
        y_fmt=".2f", y_title="%",
        height=420,
    )
with r2c2:
    st.markdown("**Australia — RBA Cash Rate / AONIA & CPI YoY**")
    fred_chart_block(
        {
            "RBA Cash Rate / AONIA": au_cash,
            "Australia CPI YoY (Q)": au_yoy,
        },
        caption=(
            "RBA Cash Rate Target — AONIA is the realised overnight rate the RBA publishes "
            "as the Cash Rate. CPI is quarterly (ABS)."
        ),
        key_prefix="au_block",
        include_ytd=True,
        y_fmt=".2f", y_title="%",
        height=420,
    )

st.markdown("---")
st.caption(
    "Methodology: index returns from yfinance (Adj Close where available). "
    "Section 1 returns table is local currency. Section 4 RoW indices converted to USD via "
    "Polygon FX (yfinance fallback). Correlations: daily log returns, ~10y, EWMA λ = 0.94. "
    "Core PCE = YoY % Δ of PCEPILFE. DXY = FRED DTWEXBGS. 10Y UST price proxy = ZN=F. "
    "DM ex-US rates & inflation sourced from FRED (ECB, BoE, BoJ, RBA, Eurostat HICP, OECD MEI CPI)."
)
