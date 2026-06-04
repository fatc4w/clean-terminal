# Macro Markets Dashboard

A single-page Streamlit dashboard for a macro hedge fund desk. Layout is
"n × 1" (one panel per row, scroll down), with the Economic Conditions
section laid out two charts per row.

## Sections

1. **Financial Markets**
   - 1W / 1M / 3M / YTD returns for SPX, NDQ, KOSPI, Nikkei 225, CSI 300,
     STOXX 600, STI
   - Cross-asset EWMA correlation matrix (λ = 0.94, daily log returns) over
     WTI front-month, SPX, ACWI, Gold, 10Y UST futures (price), Broad DXY
   - **Click any cell** in the correlation matrix → a dialog opens with the
     full EWMA correlation history, with 1Y / 5Y / Max toggles. Close with
     the X button.
2. **Economic Conditions** (2 charts per row, 1Y / 5Y / Max toggles)
   - US Treasury yield curve (2Y / 5Y / 10Y / 30Y on one chart)
   - PCEPILFE (core PCE price index)
   - MICH (UMich 1Y inflation expectations)
3. **Financial Conditions** (1Y / 5Y / Max toggles)
   - Money market rates: SOFR, EFFR, Discount Window Primary, ON RRP Award,
     IORB
   - Money market spreads in bps: SOFR − EFFR, SOFR − ON RRP (forward-filled
     to smooth out missing observations)
   - Chicago Fed NFCI
4. **US vs Rest of World** (YTD / 1Y / 5Y / Max toggles)
   - Equity indices rebased to 100: TAIEX, CSI 300, KOSPI, NIFTY 50,
     IBOVESPA, TASI, JSE All Share, BMV IPC, with SPX overlaid bold/opaque.
     Click a legend entry to isolate one series, double-click to restore.

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## FRED API key

A key is bundled in `data.py` as the default. To override it (recommended
for production), copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and replace the value, or set it in the Streamlit
Cloud "Secrets" panel.

## Files

- `app.py` — UI / layout / Plotly charts / dialog wiring
- `data.py` — yfinance + FRED fetchers, cached for 1 hour per session
- `utils.py` — EWMA correlation math, index-to-100 helper
- `requirements.txt` — pinned minimum versions
- `.streamlit/config.toml` — dark theme
- `.streamlit/secrets.toml.example` — template for the FRED key override

## Notes on data sources

- 10Y Treasury **price** uses the front-month 10Y T-Note futures (`ZN=F`)
  so the conventional inverse relationship with yields shows through.
- DXY uses **DTWEXBGS** (Nominal Broad U.S. Dollar Index) from FRED.
- TASI (Saudi Arabia) and JSE All Share (South Africa) are pulled from
  yfinance via `^TASI.SR` and `^J203.JO`; coverage can be patchy — if
  yfinance returns nothing the dashboard will simply omit them.
