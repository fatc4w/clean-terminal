"""Macro Markets Dashboard
A single-page Streamlit app, n x 1 layout (one panel per row, with the
Economic Conditions section laid out 2-up).

Run locally:
    streamlit run app.py
"""
from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import (
    MARKETS, CORR_ASSETS, ROW_MARKETS,
    get_market_returns_table,
    get_corr_assets_prices,
    get_fred_series,
    get_row_prices,
)
from utils import (
    log_returns,
    ewma_corr_pairwise_series,
    latest_corr_matrix,
    index_to_100,
)


# ---------------------------------------------------------------------------
# Page setup + light styling
# ---------------------------------------------------------------------------
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
st.caption(f"As of {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} • Data: yfinance + FRED")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
WIN_STD = ["1Y", "5Y", "Max"]
WIN_YTD = ["YTD", "1Y", "5Y", "Max"]
PLOT_BG = "#0E1117"


def _window_start(end: pd.Timestamp, window: str, series_list=None) -> pd.Timestamp | None:
    if window == "1Y":
        return end - pd.DateOffset(years=1)
    if window == "5Y":
        return end - pd.DateOffset(years=5)
    if window == "YTD":
        return pd.Timestamp(year=end.year, month=1, day=1)
    # Max
    if series_list is not None:
        return min(s.index[0] for s in series_list if len(s) > 0)
    return None


def _line_fig(series_dict: dict, title: str, height: int = 380, y_fmt: str | None = None,
              y_title: str | None = None) -> go.Figure:
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
        title=dict(text=title, font=dict(size=14, color="#C9CCD0")),
        height=height,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        yaxis=dict(title=y_title) if y_title else dict(),
    )
    if y_fmt:
        fig.update_yaxes(tickformat=y_fmt)
    return fig


def fred_chart_block(series_dict: dict, title: str, key_prefix: str,
                     include_ytd: bool = False, y_fmt: str | None = None,
                     y_title: str | None = None, height: int = 380):
    """Render a time-series chart with a 1Y/5Y/Max (or +YTD) window selector."""
    options = WIN_YTD if include_ytd else WIN_STD
    window = st.segmented_control(
        "Window", options=options, default=options[0], key=f"{key_prefix}_win",
    )
    if window is None:
        window = options[0]

    valid = [s for s in series_dict.values() if s is not None and len(s) > 0]
    if not valid:
        st.warning(f"No data available for {title}")
        return
    end = max(s.index[-1] for s in valid)
    start = _window_start(end, window, series_list=valid)

    sliced = {n: (s.loc[s.index >= start] if start is not None else s)
              for n, s in series_dict.items()}
    fig = _line_fig(sliced, title, height=height, y_fmt=y_fmt, y_title=y_title)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")


# ---------------------------------------------------------------------------
# 1. FINANCIAL MARKETS
# ---------------------------------------------------------------------------
st.header("1. Financial Markets")

st.subheader("Global Equity Index Performance")
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
    st.dataframe(styled, use_container_width=True, height=300)

# ---- Correlation Matrix --------------------------------------------------
st.subheader("Cross-Asset Correlation Matrix")
st.caption("EWMA correlation (λ = 0.94) of daily log returns over the last ~10 years. "
           "Click any cell to view that pair's correlation history.")

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


@st.dialog("Correlation History", width="large")
def _show_corr_dialog(a: str, b: str, full_series: pd.Series):
    st.markdown(f"### {a}  ⇄  {b}")
    st.caption("EWMA correlation, λ = 0.94, daily log returns")
    safe = (a + "_" + b).replace(" ", "").replace("(", "").replace(")", "")
    window = st.segmented_control(
        "Window", options=["1Y", "5Y", "Max"], default="1Y",
        key=f"corr_win_{safe}",
    )
    if window is None:
        window = "1Y"
    end = full_series.index[-1]
    if window == "1Y":
        start = end - pd.DateOffset(years=1)
    elif window == "5Y":
        start = end - pd.DateOffset(years=5)
    else:
        start = full_series.index[0]
    s = full_series.loc[full_series.index >= start]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(color="#FF6B35", width=2),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.10)",
        hovertemplate="%{x|%Y-%m-%d}<br>ρ = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#666", line_width=1)
    fig.update_layout(
        height=440, template="plotly_dark",
        yaxis=dict(title="Correlation", range=[-1, 1], zeroline=False),
        xaxis=dict(title=""),
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"corr_dlg_chart_{safe}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current", f"{s.iloc[-1]:.3f}")
    c2.metric(f"{window} Mean", f"{s.mean():.3f}")
    c3.metric(f"{window} Min", f"{s.min():.3f}")
    c4.metric(f"{window} Max", f"{s.max():.3f}")


event = st.plotly_chart(
    heat_fig,
    on_select="rerun",
    selection_mode=("points",),
    key="corr_heatmap",
    use_container_width=True,
)

# Open the dialog when the user clicks a *new* cell. We dedupe so that
# downstream reruns (e.g. interacting with widgets elsewhere) don't reopen
# a dialog the user already dismissed.
def _selection_points(ev):
    if ev is None:
        return []
    sel = getattr(ev, "selection", None)
    if sel is None:
        return []
    if isinstance(sel, dict):
        return sel.get("points", []) or []
    return getattr(sel, "points", []) or []

pts = _selection_points(event)
if pts:
    pt = pts[0]
    a = pt.get("y")
    b = pt.get("x")
    if a and b and a != b:
        new_sel = (a, b)
        if st.session_state.get("_last_corr_click") != new_sel:
            st.session_state["_last_corr_click"] = new_sel
            series = corr_series_dict.get((a, b)) or corr_series_dict.get((b, a))
            if series is not None and len(series) > 0:
                _show_corr_dialog(a, b, series)


# ---------------------------------------------------------------------------
# 2. ECONOMIC CONDITIONS
# ---------------------------------------------------------------------------
st.header("2. Economic Conditions")

with st.spinner("Loading FRED macro series…"):
    y2  = get_fred_series("DGS2",     start="2000-01-01")
    y5  = get_fred_series("DGS5",     start="2000-01-01")
    y10 = get_fred_series("DGS10",    start="2000-01-01")
    y30 = get_fred_series("DGS30",    start="2000-01-01")
    pce = get_fred_series("PCEPILFE", start="1990-01-01")
    mich = get_fred_series("MICH",    start="1990-01-01")

# Row 1: yields | PCE
c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown("**US Treasury Yields**")
    fred_chart_block(
        {"2Y": y2, "5Y": y5, "10Y": y10, "30Y": y30},
        title="US Treasury Yields (%)",
        key_prefix="ust",
        y_fmt=".2f", y_title="Yield (%)",
        height=380,
    )
with c2:
    st.markdown("**Core PCE Price Index**")
    fred_chart_block(
        {"Core PCE": pce},
        title="PCEPILFE — PCE ex Food & Energy (Index, 2017=100)",
        key_prefix="pce",
        y_fmt=".1f", y_title="Index",
        height=380,
    )

# Row 2: Michigan inflation expectations | (empty for symmetry)
c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown("**Inflation Expectations**")
    fred_chart_block(
        {"UMich 1Y Inflation Expectation": mich},
        title="MICH — Univ. of Michigan 1Y Inflation Expectation (%)",
        key_prefix="mich",
        y_fmt=".1f", y_title="%",
        height=380,
    )
with c2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.empty()


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
    title="Money Market Rates (%)",
    key_prefix="mmr",
    y_fmt=".2f", y_title="Rate (%)",
    height=420,
)

st.subheader("Money Market Spreads")
# Smooth out missingness by carrying forward, then take spreads in bps.
if any(len(s) for s in (sofr, effr, onrrp)):
    start_idx = min(
        s.index.min() for s in (sofr, effr, onrrp) if len(s) > 0
    )
    daily_idx = pd.date_range(start=start_idx, end=pd.Timestamp.today(), freq="D")
    sofr_d  = sofr.reindex(daily_idx).ffill()
    effr_d  = effr.reindex(daily_idx).ffill()
    onrrp_d = onrrp.reindex(daily_idx).ffill()
    spread_se = ((sofr_d - effr_d) * 100).dropna()    # bps
    spread_so = ((sofr_d - onrrp_d) * 100).dropna()  # bps
else:
    spread_se = pd.Series(dtype=float)
    spread_so = pd.Series(dtype=float)

fred_chart_block(
    {"SOFR − EFFR (bps)": spread_se, "SOFR − ON RRP (bps)": spread_so},
    title="Money Market Spreads (basis points)",
    key_prefix="mms",
    y_fmt=".0f", y_title="Spread (bps)",
    height=380,
)

st.subheader("Chicago Fed National Financial Conditions Index")
fred_chart_block(
    {"NFCI": nfci},
    title="NFCI (positive = tighter than average, negative = looser)",
    key_prefix="nfci",
    y_fmt=".2f", y_title="Index",
    height=380,
)


# ---------------------------------------------------------------------------
# 4. US vs ROW
# ---------------------------------------------------------------------------
st.header("4. US vs Rest of World")
st.subheader("Equity Market Performance (Indexed to 100)")

with st.spinner("Loading global equity index data…"):
    row_prices = get_row_prices(years=10)

window = st.segmented_control(
    "Window", options=WIN_YTD, default="YTD", key="row_win",
)
if window is None:
    window = "YTD"

if row_prices.empty:
    st.warning("Could not load global equity prices.")
else:
    end = row_prices.index[-1]
    if window == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif window == "1Y":
        start = end - pd.DateOffset(years=1)
    elif window == "5Y":
        start = end - pd.DateOffset(years=5)
    else:
        start = row_prices.index[0]

    sliced = row_prices.loc[row_prices.index >= start].copy().ffill().dropna(how="all")
    indexed = index_to_100(sliced)

    # Distinct colors for the RoW lines, SPX rendered last in bold white.
    palette = [
        "#42A5F5",  # blue
        "#AB47BC",  # purple
        "#FFCA28",  # amber
        "#26C6DA",  # cyan
        "#EF5350",  # red
        "#66BB6A",  # green
        "#FF7043",  # deep orange
        "#EC407A",  # pink
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
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
        ))
    # SPX last so it sits on top, bold + fully opaque
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
        title=dict(text=f"Equity Indices Indexed to 100 — {window} window",
                   font=dict(size=14)),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            itemclick="toggleothers", itemdoubleclick="toggle",
        ),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        yaxis=dict(title="Indexed Value (start = 100)"),
    )
    st.plotly_chart(fig, use_container_width=True, key="row_chart")
    st.caption(
        "SPX rendered bold/opaque as US reference. The RoW lines are kept translucent so the chart "
        "stays readable; **single-click a legend entry to isolate that index**, double-click to "
        "restore all. Hover uses unified x-axis tooltips so you can compare all values at a given date."
    )

st.markdown("---")
st.caption(
    "Methodology notes: index returns from yfinance (Adj Close where available, else Close). "
    "Correlation panel: daily log returns, ~10y window, EWMA with λ = 0.94. "
    "DXY uses FRED's DTWEXBGS (Nominal Broad U.S. Dollar Index). "
    "10Y UST price proxy is the front-month 10Y Treasury Note futures (ZN=F)."
)
