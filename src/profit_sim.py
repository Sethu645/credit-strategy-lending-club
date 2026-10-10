"""
Strategy layer: turn PD scores into an approval decision and measure the money.

Economics per loan (all realized, from actual Lending Club cash flows)
    economic_profit = payments received * (1 - servicing fee)
                      - collection recovery fees
                      - principal lent
                      - funding cost

    funding cost = annual funding rate * principal * average life
      A 36-month amortizing loan has roughly half its principal outstanding on
      average over 3 years, so average life ~ 1.5 years. The funding rate stands
      in for cost of funds plus a capital charge. It is the key business
      assumption, so results are shown for several values.

Strategy evaluated
    Rank applicants by predicted PD (safest first) and approve the top X%.
    Compare rankers at the same approval rate: random, LC sub-grade alone,
    champion scorecard, challenger GBM.
    Cutoff choice: the PD cutoff that maximized profit on the development
    sample (2007-2014), applied unchanged to the 2015 out-of-time sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SERVICING_FEE = 0.01         # Lending Club charged investors 1% of payments received
AVG_LIFE_YEARS = 1.5         # 36-month amortizing loan
BASE_FUNDING_RATE = 0.03     # base-case annual funding cost (assumption)


def economic_profit(df: pd.DataFrame, funding_rate: float = BASE_FUNDING_RATE) -> pd.Series:
    return (df["total_pymnt"] * (1 - SERVICING_FEE)
            - df["collection_recovery_fee"]
            - df["loan_amnt"]
            - funding_rate * AVG_LIFE_YEARS * df["loan_amnt"])


def profit_curve(pd_scores, profit, grid=None) -> pd.DataFrame:
    """Cumulative profit when approving the safest X% by PD."""
    pd_scores, profit = np.asarray(pd_scores), np.asarray(profit)
    order = np.argsort(pd_scores, kind="stable")
    cum = np.cumsum(profit[order])
    n = len(order)
    grid = np.linspace(0.01, 1.0, 100) if grid is None else np.asarray(grid)
    idx = np.clip((grid * n).astype(int) - 1, 0, n - 1)
    return pd.DataFrame({"approval_rate": grid, "profit": cum[idx]})


def best_cutoff(pd_scores, profit) -> tuple[float, float, float]:
    """PD cutoff that maximizes total profit. Returns (cutoff, approval_rate, profit)."""
    pd_scores, profit = np.asarray(pd_scores), np.asarray(profit)
    order = np.argsort(pd_scores, kind="stable")
    cum = np.cumsum(profit[order])
    k = int(np.argmax(cum))
    return float(pd_scores[order][k]), (k + 1) / len(order), float(cum[k])


def apply_cutoff(pd_scores, cutoff, df: pd.DataFrame, profit) -> dict:
    approve = np.asarray(pd_scores) <= cutoff
    return {
        "approval_rate": approve.mean(),
        "bad_rate_approved": df.loc[approve, "is_default"].mean(),
        "profit": float(np.asarray(profit)[approve].sum()),
    }


def strategy_table(rankers: dict, dev: pd.DataFrame, oot: pd.DataFrame,
                   funding_rate: float = BASE_FUNDING_RATE) -> pd.DataFrame:
    """For each ranker: cutoff set on dev data -> realized on 2015, plus the
    hindsight-best cutoff on 2015 and profit at fixed approval rates.

    rankers: name -> (dev_scores, oot_scores), higher score = riskier.
    """
    p_dev, p_oot = economic_profit(dev, funding_rate), economic_profit(oot, funding_rate)
    rows = []
    for name, (s_dev, s_oot) in rankers.items():
        cut, _, _ = best_cutoff(s_dev, p_dev)
        live = apply_cutoff(s_oot, cut, oot, p_oot)
        _, hind_rate, hind_profit = best_cutoff(s_oot, p_oot)
        curve = profit_curve(s_oot, p_oot, grid=[0.6, 0.7, 0.8, 0.9])
        row = {
            "ranker": name,
            "devcut_approval": live["approval_rate"],
            "devcut_bad_rate": live["bad_rate_approved"],
            "devcut_profit": live["profit"],
            "hindsight_best_approval": hind_rate,
            "hindsight_best_profit": hind_profit,
        }
        for r, p in zip(curve["approval_rate"], curve["profit"]):
            row[f"profit_at_{int(round(r * 100))}pct"] = p
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(1, "approve_all_profit", float(p_oot.sum()))
    return out


def swap_set(pd_champ, pd_chall, df: pd.DataFrame, profit, approval_rate=0.8) -> pd.DataFrame:
    """Champion vs challenger at the same approval rate.
    Swap-in: approved only by challenger. Swap-out: approved only by champion."""
    profit = np.asarray(profit)
    n_appr = int(approval_rate * len(df))
    a_champ = np.zeros(len(df), bool); a_champ[np.argsort(pd_champ, kind="stable")[:n_appr]] = True
    a_chall = np.zeros(len(df), bool); a_chall[np.argsort(pd_chall, kind="stable")[:n_appr]] = True
    groups = {
        "Both approve": a_champ & a_chall,
        "Swap-in (challenger only)": ~a_champ & a_chall,
        "Swap-out (champion only)": a_champ & ~a_chall,
        "Both decline": ~a_champ & ~a_chall,
    }
    return pd.DataFrame([{
        "segment": k, "n": int(m.sum()),
        "bad_rate": df.loc[m, "is_default"].mean(),
        "avg_int_rate": df.loc[m, "int_rate"].mean(),
        "profit": float(profit[m].sum()),
        "profit_per_loan": float(profit[m].mean()),
    } for k, m in groups.items()])


def expected_vs_actual(pd_scores, df: pd.DataFrame, by="grade") -> pd.DataFrame:
    """Predicted vs realized default rate by segment: the monitoring view."""
    d = pd.DataFrame({by: df[by].values, "predicted_pd": np.asarray(pd_scores),
                      "actual_default": df["is_default"].values})
    t = d.groupby(by).agg(n=("actual_default", "size"),
                          predicted_pd=("predicted_pd", "mean"),
                          actual_default=("actual_default", "mean"))
    t["gap_pts"] = (t["actual_default"] - t["predicted_pd"]) * 100
    return t.reset_index()
