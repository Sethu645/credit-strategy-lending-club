"""Discrimination, calibration and stability metrics used in credit model validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_stat(y, pd_) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative bad and good distributions."""
    fpr, tpr, _ = roc_curve(y, pd_)
    return float(np.max(tpr - fpr))


def discrimination(y, pd_) -> dict:
    auc = roc_auc_score(y, pd_)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_stat(y, pd_)}


def decile_table(y, pd_, n=10) -> pd.DataFrame:
    """Predicted vs actual default rate by risk decile (1 = riskiest)."""
    d = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_)})
    d["decile"] = pd.qcut(d["pd"].rank(method="first", ascending=False), n, labels=range(1, n + 1))
    t = d.groupby("decile", observed=True).agg(
        n=("y", "size"), predicted_pd=("pd", "mean"), actual_default=("y", "mean"))
    t["capture_of_bads"] = d.groupby("decile", observed=True)["y"].sum().cumsum() / d["y"].sum()
    return t.reset_index()


def psi(expected, actual, n_bins=10) -> float:
    """Population Stability Index between development and validation score distributions.
    < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant shift."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected), minlength=len(edges) + 1) / len(expected)
    a = np.bincount(np.searchsorted(edges, actual), minlength=len(edges) + 1) / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
