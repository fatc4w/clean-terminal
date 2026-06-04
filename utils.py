"""EWMA correlation utilities and small numeric helpers."""
import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def ewma_corr_pairwise_series(returns: pd.DataFrame, lam: float = 0.94) -> dict:
    """Compute the full time series of EWMA correlations for every asset pair.

    RiskMetrics convention (zero-mean returns):
        sigma2_i,t = lambda * sigma2_i,{t-1} + (1-lambda) * r_i,t^2
        cov_ij,t  = lambda * cov_ij,{t-1}  + (1-lambda) * r_i,t * r_j,t
        rho_ij,t  = cov_ij,t / sqrt(sigma2_i,t * sigma2_j,t)

    pandas' .ewm(adjust=False).mean() implements exactly the same recursion,
    so we use it for speed and numerical stability.
    """
    alpha = 1.0 - lam
    cols = list(returns.columns)
    # second-moments of each column (variances under zero-mean assumption)
    var = {c: (returns[c] ** 2).ewm(alpha=alpha, adjust=False).mean() for c in cols}

    out = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            cov = (returns[a] * returns[b]).ewm(alpha=alpha, adjust=False).mean()
            rho = cov / np.sqrt(var[a] * var[b])
            rho = rho.replace([np.inf, -np.inf], np.nan).dropna()
            out[(a, b)] = rho
    return out


def latest_corr_matrix(corr_series_dict: dict, assets: list) -> pd.DataFrame:
    n = len(assets)
    M = pd.DataFrame(np.eye(n), index=assets, columns=assets)
    for (a, b), s in corr_series_dict.items():
        if len(s) == 0:
            continue
        v = float(s.iloc[-1])
        M.loc[a, b] = v
        M.loc[b, a] = v
    return M


def index_to_100(prices: pd.DataFrame) -> pd.DataFrame:
    """Rebase each column so its first non-NaN value is 100."""
    out = {}
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) == 0:
            out[c] = prices[c]
            continue
        out[c] = (prices[c] / s.iloc[0]) * 100.0
    return pd.DataFrame(out, index=prices.index)
