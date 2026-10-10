"""
End-to-end pipeline. From the project root:

    python run_pipeline.py

Builds the SQLite database, fits both models, runs the strategy simulation,
and writes tables to reports/ and charts to reports/figures/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import plots
from src.challenger import GBMChallenger
from src.evaluate import decile_table, discrimination, psi
from src.load_data import build_database, load_model_table, run_query, split_out_of_time
from src.profit_sim import (BASE_FUNDING_RATE, best_cutoff, economic_profit,
                            expected_vs_actual, profit_curve, strategy_table, swap_set)
from src.woe_scorecard import WoEScorecard, iv_summary

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
FUNDING_RATES = [0.02, 0.03, 0.04]


def run(verbose: bool = True) -> dict:
    REPORTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    # 1. Data (SQL)
    build_database()
    df = load_model_table()
    dev, oot = split_out_of_time(df)
    run_query("portfolio_summary.sql").to_csv(REPORTS / "portfolio_summary.csv", index=False)
    say(f"dev 2007-2014: {len(dev):,} loans, default {dev.is_default.mean():.1%} | "
        f"OOT 2015: {len(oot):,} loans, default {oot.is_default.mean():.1%}")

    # 2. Models
    champ = WoEScorecard().fit(dev)
    chall = GBMChallenger().fit(dev)
    pd_dev = {"champ": champ.predict_pd(dev), "chall": chall.predict_pd(dev)}
    pd_oot = {"champ": champ.predict_pd(oot), "chall": chall.predict_pd(oot)}

    iv = iv_summary(champ.bins)
    iv.to_csv(REPORTS / "iv_summary.csv", index=False)
    champ.points_table().to_csv(REPORTS / "scorecard_points.csv", index=False)
    champ.coefficients().to_csv(REPORTS / "scorecard_coefficients.csv", index=False)
    pd.Series(champ.dropped, name="reason").rename_axis("feature") \
        .to_csv(REPORTS / "features_dropped.csv")
    chall.feature_importance().to_csv(REPORTS / "challenger_importance.csv", index=False)

    # 3. Validation
    rows = []
    benchmark = {"LC sub-grade only": (dev.subgrade_rank, oot.subgrade_rank),
                 "Champion scorecard": (pd_dev["champ"], pd_oot["champ"]),
                 "Challenger GBM": (pd_dev["chall"], pd_oot["chall"])}
    for name, (s_dev, s_oot) in benchmark.items():
        d, o = discrimination(dev.is_default, s_dev), discrimination(oot.is_default, s_oot)
        rows.append({"model": name, "AUC_dev": d["AUC"], "AUC_2015": o["AUC"],
                     "Gini_2015": o["Gini"], "KS_2015": o["KS"]})
    perf = pd.DataFrame(rows)
    perf["score_PSI"] = [np.nan,
                         psi(champ.score(dev), champ.score(oot)),
                         psi(pd_dev["chall"], pd_oot["chall"])]
    perf.to_csv(REPORTS / "model_performance.csv", index=False)
    decile_table(oot.is_default, pd_oot["champ"]).to_csv(REPORTS / "champion_deciles_2015.csv", index=False)
    say(perf.round(3).to_string(index=False))

    # 4. Strategy
    rng = np.random.default_rng(0)
    # tiny jitter breaks ties inside a sub-grade so ranking is well defined
    rankers = {
        "Random": (rng.random(len(dev)), rng.random(len(oot))),
        "LC sub-grade only": (dev.subgrade_rank + rng.random(len(dev)) * 1e-3,
                              oot.subgrade_rank + rng.random(len(oot)) * 1e-3),
        "Champion scorecard": (pd_dev["champ"], pd_oot["champ"]),
        "Challenger GBM": (pd_dev["chall"], pd_oot["chall"]),
    }
    strat = pd.concat([strategy_table(rankers, dev, oot, r).assign(funding_rate=r)
                       for r in FUNDING_RATES], ignore_index=True)
    strat.to_csv(REPORTS / "strategy_results.csv", index=False)
    base = strat[strat.funding_rate == BASE_FUNDING_RATE]
    say(base.drop(columns="funding_rate").round(3).to_string(index=False))

    p_oot = economic_profit(oot, BASE_FUNDING_RATE)
    swap = swap_set(pd_oot["champ"], pd_oot["chall"], oot, p_oot, approval_rate=0.8)
    swap.to_csv(REPORTS / "swap_set_80pct.csv", index=False)
    eva = expected_vs_actual(pd_oot["champ"], oot)
    eva.to_csv(REPORTS / "expected_vs_actual_2015.csv", index=False)

    # 5. Figures
    curves = {n: profit_curve(s[1], p_oot) for n, s in rankers.items()}
    p_dev = economic_profit(dev, BASE_FUNDING_RATE)
    devcut_points = {}
    for n, (s_dev, s_oot) in rankers.items():
        cut, _, _ = best_cutoff(s_dev, p_dev)
        appr = np.asarray(s_oot) <= cut
        devcut_points[n] = (appr.mean(), float(p_oot[appr].sum()))
    plots.profit_curves(curves, float(p_oot.sum()), devcut_points,
                        FIGS / "profit_curve.png", BASE_FUNDING_RATE)
    plots.expected_vs_actual(eva, FIGS / "expected_vs_actual.png")
    plots.information_value(iv, champ.features, FIGS / "information_value.png")
    say(f"wrote tables to {REPORTS.name}/ and charts to {FIGS.relative_to(ROOT)}/")

    return dict(df=df, dev=dev, oot=oot, champ=champ, chall=chall, pd_dev=pd_dev,
                pd_oot=pd_oot, perf=perf, strategy=strat, swap=swap, eva=eva, iv=iv)


if __name__ == "__main__":
    run()
