"""
Champion model: Weight-of-Evidence (WoE) logistic regression scorecard.

Steps
  1. Bin every candidate feature on the development sample.
     Numeric: quantile bins, reduced until the bad rate is monotonic across bins
              (scorecard reviewers expect monotonic, explainable risk patterns).
     Categorical: one bin per level, rare levels (<1%) grouped into OTHER.
     Missing values always get their own bin.
  2. Compute WoE per bin and Information Value (IV) per feature.
  3. Select features: IV >= 0.02, then drop one of any pair of WoE features with
     |correlation| > 0.6 (keep the higher IV).
  4. Fit logistic regression on WoE values. Drop any feature whose coefficient
     has the wrong sign and refit, so every feature moves risk in the direction
     its bins say it should.
  5. Scale to points: 600 points at 50:1 good:bad odds, 20 points to double the odds.

WoE convention: WoE = ln(%good / %bad). Higher WoE = safer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TARGET = "is_default"

NUMERIC_FEATURES = [
    "subgrade_rank",   # LC's grade, treated as an external bureau-style score
    "loan_amnt", "annual_inc", "dti", "fico", "emp_years", "credit_history_months",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "total_acc", "pub_rec",
    "pub_rec_bankruptcies", "revol_bal", "revol_util", "mort_acc",
    "loan_to_income", "payment_to_income", "revol_to_income", "open_acc_ratio",
]
CATEGORICAL_FEATURES = ["purpose", "home_ownership", "verification_status"]
# addr_state deliberately excluded: geography can act as a proxy for protected
# characteristics, and it adds little once affordability is in the model.

MAX_BINS = 10
MIN_BIN_SHARE = 0.05
RARE_LEVEL_SHARE = 0.01
IV_MIN = 0.02
CORR_MAX = 0.6
EPS = 0.5  # smoothing so empty cells don't give infinite WoE

PDO = 20           # points to double the odds
BASE_SCORE = 600   # score at BASE_ODDS
BASE_ODDS = 50     # good:bad


# ---------------------------------------------------------------- binning ---
@dataclass
class FeatureBins:
    name: str
    kind: str                          # "numeric" or "categorical"
    edges: list[float] | None = None   # numeric: inner cut points
    levels: list[str] | None = None    # categorical: kept levels
    woe: dict = field(default_factory=dict)   # bin label -> WoE
    table: pd.DataFrame | None = None
    iv: float = 0.0

    def assign(self, s: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            idx = np.searchsorted(self.edges, s.to_numpy(dtype=float), side="right")
            labels = pd.Series(idx, index=s.index).astype(str)
            labels[s.isna()] = "MISSING"
            return labels
        labels = s.astype("string").fillna("MISSING")
        labels = labels.where(labels.isin(self.levels + ["MISSING"]), "OTHER")
        return labels.astype(str)

    def transform(self, s: pd.Series) -> pd.Series:
        # unseen bins (e.g. a new category) get WoE 0 = neutral
        return self.assign(s).map(self.woe).fillna(0.0).astype(float)


def _woe_table(labels: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"bin": labels, "bad": y}).groupby("bin")["bad"].agg(["count", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    tot_good, tot_bad = t["good"].sum(), t["bad"].sum()
    t["pct_good"] = (t["good"] + EPS) / (tot_good + EPS * len(t))
    t["pct_bad"] = (t["bad"] + EPS) / (tot_bad + EPS * len(t))
    t["woe"] = np.log(t["pct_good"] / t["pct_bad"])
    t["iv"] = (t["pct_good"] - t["pct_bad"]) * t["woe"]
    t["bad_rate"] = t["bad"] / t["n"]
    t["share"] = t["n"] / t["n"].sum()
    return t


def _numeric_edges(x: pd.Series, y: pd.Series) -> list[float]:
    """Largest number of quantile bins (<= MAX_BINS) with a monotonic bad rate."""
    x_nn, y_nn = x[x.notna()], y[x.notna()]
    x_min = x_nn.min()
    for n_bins in range(MAX_BINS, 1, -1):
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        # Bins are [edge_i, edge_i+1). Drop edges at the minimum value: for
        # zero-heavy counts (inquiries, delinquencies) quantiles repeat 0, and
        # an edge at 0 would create an empty bin below it.
        edges = sorted({float(q) for q in np.quantile(x_nn, qs) if q > x_min})
        if not edges:
            continue
        idx = np.searchsorted(edges, x_nn.to_numpy(dtype=float), side="right")
        counts = np.bincount(idx, minlength=len(edges) + 1)
        if (counts / len(x_nn) < MIN_BIN_SHARE).any():
            continue
        rates = np.bincount(idx, weights=y_nn, minlength=len(edges) + 1) / counts
        diffs = np.diff(rates)
        if (diffs >= 0).all() or (diffs <= 0).all():
            return edges
    # Quantiles too coarse (e.g. a count that is 0 for 95% of rows):
    # fall back to "minimum value" vs "anything above it".
    above = x_nn[x_nn > x_min]
    return [float(above.min())] if len(above) else []


def fit_bins(df: pd.DataFrame, y: pd.Series) -> dict[str, FeatureBins]:
    bins: dict[str, FeatureBins] = {}
    for col in NUMERIC_FEATURES:
        fb = FeatureBins(col, "numeric", edges=_numeric_edges(df[col], y))
        bins[col] = fb
    for col in CATEGORICAL_FEATURES:
        share = df[col].value_counts(normalize=True)
        fb = FeatureBins(col, "categorical", levels=share[share >= RARE_LEVEL_SHARE].index.tolist())
        bins[col] = fb
    for fb in bins.values():
        t = _woe_table(fb.assign(df[fb.name]), y)
        fb.table, fb.woe, fb.iv = t, t["woe"].to_dict(), float(t["iv"].sum())
    return bins


def bin_label(fb: FeatureBins, label: str) -> str:
    """Human-readable range for a numeric bin label."""
    if fb.kind == "categorical" or label == "MISSING":
        return label
    i, e = int(label), fb.edges
    lo = "-inf" if i == 0 else f"{e[i - 1]:,.4g}"
    hi = "inf" if i == len(e) else f"{e[i]:,.4g}"
    return f"[{lo}, {hi})"


def iv_summary(bins: dict[str, FeatureBins]) -> pd.DataFrame:
    def strength(iv):
        return ("useless" if iv < 0.02 else "weak" if iv < 0.1
                else "medium" if iv < 0.3 else "strong")
    s = pd.DataFrame({"feature": list(bins), "iv": [b.iv for b in bins.values()]})
    s["strength"] = s["iv"].map(strength)
    return s.sort_values("iv", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------- scorecard ---
class WoEScorecard:
    def __init__(self):
        self.bins: dict[str, FeatureBins] = {}
        self.features: list[str] = []
        self.dropped: dict[str, str] = {}
        self.model: LogisticRegression | None = None
        self.factor = PDO / np.log(2)
        self.offset = BASE_SCORE - self.factor * np.log(BASE_ODDS)

    def woe_frame(self, df: pd.DataFrame, features=None) -> pd.DataFrame:
        features = features or self.features
        return pd.DataFrame({f: self.bins[f].transform(df[f]) for f in features})

    def fit(self, df: pd.DataFrame) -> "WoEScorecard":
        y = df[TARGET]
        self.bins = fit_bins(df, y)
        ivs = iv_summary(self.bins)

        # 1) IV filter
        keep = ivs.loc[ivs["iv"] >= IV_MIN, "feature"].tolist()
        for f in ivs.loc[ivs["iv"] < IV_MIN, "feature"]:
            self.dropped[f] = "IV < 0.02"

        # 2) correlation filter (keep is already sorted by IV, descending)
        W = self.woe_frame(df, keep)
        corr = W.corr().abs()
        selected: list[str] = []
        for f in keep:
            clash = [s for s in selected if corr.loc[f, s] > CORR_MAX]
            if clash:
                self.dropped[f] = f"corr > {CORR_MAX} with {clash[0]}"
            else:
                selected.append(f)

        # 3) fit, removing wrong-sign coefficients one at a time
        while True:
            X = self.woe_frame(df, selected)
            m = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
            coefs = pd.Series(m.coef_[0], index=selected)
            wrong = coefs[coefs > 0]   # with WoE = ln(good/bad), coefs must be < 0
            if wrong.empty:
                break
            worst = wrong.idxmax()
            self.dropped[worst] = "wrong coefficient sign"
            selected.remove(worst)

        self.features, self.model = selected, m
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.woe_frame(df))[:, 1]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Credit score: higher = safer. score = offset + factor * ln(good/bad odds)."""
        pd_ = np.clip(self.predict_pd(df), 1e-6, 1 - 1e-6)
        return self.offset + self.factor * np.log((1 - pd_) / pd_)

    def points_table(self) -> pd.DataFrame:
        """Points per attribute. Summing a row's points gives its score."""
        b0 = self.model.intercept_[0]
        n = len(self.features)
        rows = []
        for f, b in zip(self.features, self.model.coef_[0]):
            fb = self.bins[f]
            for label, r in fb.table.iterrows():
                pts = -(b * r["woe"] + b0 / n) * self.factor + self.offset / n
                rows.append({
                    "feature": f, "bin": bin_label(fb, label), "n": int(r["n"]),
                    "bad_rate": round(r["bad_rate"], 4), "woe": round(r["woe"], 4),
                    "points": round(pts, 1),
                })
        out = pd.DataFrame(rows)
        # numeric bins in natural order
        out["_o"] = out["bin"].str.extract(r"^\[(-inf|[\d,.\-e+]+)")[0] \
            .str.replace(",", "").replace("-inf", "-1e18").astype(float)
        return out.sort_values(["feature", "_o"], na_position="last").drop(columns="_o") \
            .reset_index(drop=True)

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.features,
            "coef": self.model.coef_[0],
            "iv": [self.bins[f].iv for f in self.features],
        }).sort_values("iv", ascending=False).reset_index(drop=True)
