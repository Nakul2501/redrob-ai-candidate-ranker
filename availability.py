"""
bias_audit.py
=============
Two things, run after scoring the full pool:

1. FEATURE IMPORTANCE REPORT
   Since the ranker is an explicit weighted-rule system (not a trained
   black box), "feature importance" is knowable exactly: it's the WEIGHTS
   and CORE_CLUSTER_WEIGHTS dicts in scoring.py, plus the average
   contribution each component actually made across the scored pool. This
   module reports both: the *declared* weights and the *realized* average
   contribution, so a reviewer can see whether any single component is
   silently dominating in practice.

2. SUBGROUP SCORE AUDIT
   Reports mean final_score and top-100 selection rate broken out by
   country, current_company_size, and current_industry -- attributes the
   scoring code does not directly optimize for. Large, unexplained
   disparities here are a signal to go back and inspect the rule set,
   not evidence of a "correct" answer by themselves (this JD is
   deliberately narrow, so real disparity by e.g. industry is expected --
   IT-services-only candidates SHOULD score lower, per the JD itself). The
   report exists so that disparity is visible and attributable, not buried.

Nothing here changes ranking. This module is diagnostic only.
"""

from __future__ import annotations

import pandas as pd

from .scoring import TECH_WEIGHTS, CORE_CLUSTER_WEIGHTS


def feature_importance_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """Reports the three TECH_WEIGHTS components (which combine additively
    into base_score) plus location/notice separately, noting that the
    latter two act as a multiplier on base_score rather than an additive
    term -- see scoring.py's TECH_WEIGHTS docstring for why."""
    rows = []
    for comp, declared_weight in TECH_WEIGHTS.items():
        realized_mean = results_df[comp].mean()
        realized_contribution = declared_weight * realized_mean
        rows.append({
            "component": comp,
            "role": "additive (base_score)",
            "declared_weight": declared_weight,
            "avg_component_score_0_1": round(realized_mean, 4),
            "avg_realized_contribution_to_base_score": round(realized_contribution, 4),
        })
    for comp in ["location_logistics_fit", "notice_period_fit"]:
        realized_mean = results_df[comp].mean()
        rows.append({
            "component": comp,
            "role": "multiplicative (logistics_multiplier, range 0.5-1.0x on base_score)",
            "declared_weight": 0.5,  # each contributes half of the multiplier's swing
            "avg_component_score_0_1": round(realized_mean, 4),
            "avg_realized_contribution_to_base_score": None,
        })
    df = pd.DataFrame(rows)
    return df


def core_cluster_importance_report(component_breakdowns: list[dict]) -> pd.DataFrame:
    rows = []
    for cname, weight in CORE_CLUSTER_WEIGHTS.items():
        vals = [bd["core_ml_fit"].get(cname, 0.0) for bd in component_breakdowns]
        rows.append({
            "core_cluster": cname,
            "declared_weight_within_core_ml_fit": weight,
            "avg_cluster_score_0_1": round(sum(vals) / len(vals), 4) if vals else 0.0,
        })
    return pd.DataFrame(rows).sort_values("avg_cluster_score_0_1", ascending=False)


def subgroup_audit(df: pd.DataFrame, group_col: str, top_n_flag_col: str = "in_top_100") -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(group_col).agg(
        n=("final_score", "size"),
        mean_final_score=("final_score", "mean"),
        top100_count=(top_n_flag_col, "sum"),
    ).reset_index()
    g["top100_rate"] = g["top100_count"] / g["n"]
    return g.sort_values("mean_final_score", ascending=False)
