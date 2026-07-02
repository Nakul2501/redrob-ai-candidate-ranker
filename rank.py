#!/usr/bin/env python3
"""
rank.py
=======
Redrob Hackathon -- Intelligent Candidate Discovery & Ranking Challenge.

Usage:
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv --top-k 100 --audit-dir ./audit

Design constraints this respects (submission_spec.md Section 3):
    - CPU only, no GPU
    - No network calls (pure rule-based scoring, no hosted LLM/API calls)
    - Streams the candidate file rather than loading it all into memory
    - Designed to finish well under 5 minutes / 16GB RAM for 100K candidates
      (rough local benchmark logged at the end of the run)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import features as feat_mod
from src import io_utils
from src import reasoning as reasoning_mod
from src import scoring
from src import bias_audit


def parse_args():
    ap = argparse.ArgumentParser(description="Rank candidates for the Redrob AI Senior AI Engineer JD.")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl / .jsonl.gz / .json")
    ap.add_argument("--out", required=True, help="Output CSV path (submission format)")
    ap.add_argument("--top-k", type=int, default=100, help="Number of ranked rows to output (default 100)")
    ap.add_argument("--audit-dir", default=None, help="Optional directory to write bias/audit reports to")
    ap.add_argument("--limit", type=int, default=None, help="Debug only: stop after N candidates")
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    all_results = []
    all_feats_light = []  # keep small subset for audit grouping, avoid holding everything heavy
    n_seen = 0
    n_honeypot = 0

    for candidate in io_utils.iter_candidates(args.candidates):
        n_seen += 1
        if args.limit and n_seen > args.limit:
            break
        try:
            feat = feat_mod.extract(candidate)
            result = scoring.score_candidate(feat)
        except Exception as e:  # noqa: BLE001 -- never let one malformed record kill a 100K-row run
            print(f"[warn] skipping candidate at row {n_seen} due to error: {e}", file=sys.stderr)
            continue

        if result["honeypot_flag"]:
            n_honeypot += 1

        result["_reasoning"] = reasoning_mod.build_reasoning(feat, result)
        result["_profile"] = feat["profile"]
        all_results.append(result)
        all_feats_light.append({
            "candidate_id": feat["candidate_id"],
            "country": feat["country"],
            "current_company_size": feat["profile"].get("current_company_size"),
            "current_industry": feat["profile"].get("current_industry"),
        })

        if n_seen % 20000 == 0:
            print(f"[info] processed {n_seen} candidates ({time.time() - t0:.1f}s elapsed)", file=sys.stderr)

    elapsed_score = time.time() - t0
    print(f"[info] scored {n_seen} candidates in {elapsed_score:.1f}s "
          f"({n_honeypot} flagged as honeypots and excluded)", file=sys.stderr)

    # rank: highest final_score first, tie-break candidate_id ascending
    ranked = sorted(
        [r for r in all_results if not r["honeypot_flag"]],
        key=lambda r: (-r["final_score"], r["candidate_id"]),
    )

    top_k = ranked[: args.top_k]
    if len(top_k) < args.top_k:
        print(f"[warn] only {len(top_k)} non-honeypot candidates available; "
              f"requested top-{args.top_k}. (Expected on the small sample file -- "
              f"the full 100K pool will have plenty of headroom.)", file=sys.stderr)

    # normalize score column to (0,1], monotonic non-increasing by construction
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(top_k, start=1):
            writer.writerow([r["candidate_id"], i, f"{r['final_score']:.4f}", r["_reasoning"]])

    print(f"[info] wrote {len(top_k)} rows to {out_path}", file=sys.stderr)

    # ---- audit reports (optional) ----
    if args.audit_dir:
        audit_dir = Path(args.audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame([{
            "candidate_id": r["candidate_id"],
            "final_score": r["final_score"],
            "fit_score": r["fit_score"],
            "core_ml_fit": r["core_ml_fit"],
            "nice_to_have_fit": r["nice_to_have_fit"],
            "seniority_experience_fit": r["seniority_experience_fit"],
            "location_logistics_fit": r["location_logistics_fit"],
            "notice_period_fit": r["notice_period_fit"],
            "disqualifier_multiplier": r["disqualifier_multiplier"],
            "availability_multiplier": r["availability_multiplier"],
            "honeypot_flag": r["honeypot_flag"],
        } for r in all_results])

        top_ids = {r["candidate_id"] for r in top_k}
        results_df["in_top_100"] = results_df["candidate_id"].isin(top_ids)

        meta_df = pd.DataFrame(all_feats_light)
        merged = results_df.merge(meta_df, on="candidate_id", how="left")

        fi = bias_audit.feature_importance_report(results_df)
        fi.to_csv(audit_dir / "feature_importance.csv", index=False)

        core_fi = bias_audit.core_cluster_importance_report(
            [r["component_breakdown"] for r in all_results]
        )
        core_fi.to_csv(audit_dir / "core_cluster_importance.csv", index=False)

        for col in ["country", "current_company_size", "current_industry"]:
            sub = bias_audit.subgroup_audit(merged, col)
            if not sub.empty:
                sub.to_csv(audit_dir / f"subgroup_audit_{col}.csv", index=False)

        results_df.sort_values("final_score", ascending=False).to_csv(
            audit_dir / "all_candidates_scored.csv", index=False
        )

        print(f"[info] wrote audit reports to {audit_dir}", file=sys.stderr)

    total_elapsed = time.time() - t0
    print(f"[info] total runtime {total_elapsed:.1f}s for {n_seen} candidates", file=sys.stderr)


if __name__ == "__main__":
    main()
