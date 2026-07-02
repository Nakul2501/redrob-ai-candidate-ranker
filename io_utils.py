"""
reasoning.py
============
Generates the `reasoning` column text for each row of the submission CSV.

Every sentence produced here is built directly from a real field in the
candidate's record or a real computed feature -- never from a template that
just swaps in a name. This is deliberate: submission_spec.md Section 3
Stage-4 review explicitly checks for hallucination ("does every claim
correspond to something actually in the candidate's profile?"), templated
sameness, and rank-consistency (a rank-95 candidate should not read like a
glowing rank-5 writeup).

Structure per row: [role + experience fact] + [strongest matched evidence,
named specifically] + [honest concern, if the score components surfaced
one] + [behavioral note, if materially good or bad].
"""

from __future__ import annotations


def _top_matched_clusters(feat: dict, k: int = 2) -> list[str]:
    hits = {**feat["core_cluster_hits"]}
    ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
    return [name.replace("_", " ") for name, cnt in ranked[:k] if cnt > 0]


def build_reasoning(feat: dict, result: dict) -> str:
    profile = feat["profile"]
    title = profile.get("current_title", "Unknown title")
    company = profile.get("current_company", "an unnamed company")
    yoe = feat["yoe"]
    city = profile.get("location", "location unspecified")

    parts = [f"{title} at {company}, {yoe:.1f} yrs experience ({city})."]

    if result["honeypot_flag"]:
        parts.append("Excluded as a likely honeypot: " + "; ".join(result["honeypot_reasons"]) + ".")
        return " ".join(parts)

    matched = _top_matched_clusters(feat)
    if matched:
        parts.append(f"Evidence of {', '.join(matched)} found in career history/summary.")
    elif result["core_ml_fit"] > 0.3:
        parts.append("Some core-skill overlap via claimed skills, weaker direct evidence in career history.")
    else:
        parts.append("Limited evidence of embeddings/retrieval, vector-DB, or eval-framework experience.")

    if feat["shipped_hits"] > 0:
        parts.append("Profile language suggests a shipped production system, not just research/tooling exposure.")

    # Honest concerns -- pull from disqualifier reasons and low sub-scores
    concerns = []
    if result["disqualifier_reasons"]:
        concerns.extend(result["disqualifier_reasons"][:1])
    if result["seniority_experience_fit"] < 0.4:
        concerns.append("experience profile doesn't closely match the target band/role")
    if result["location_logistics_fit"] < 0.4:
        concerns.append("location/relocation logistics are a stretch against JD preference")
    nd = result["component_breakdown"]["notice_period_fit"].get("notice_period_days")
    if nd is not None and nd > 60:
        concerns.append(f"{nd}-day notice period is on the long side")
    if concerns:
        parts.append("Concern: " + concerns[0] + ".")

    # Behavioral note
    if result["availability_multiplier"] <= 0.7 and feat.get("days_since_active") is not None:
        parts.append(f"Engagement is weak (inactive {feat['days_since_active']}d, "
                      f"response rate {feat['signals'].get('recruiter_response_rate')}); "
                      f"down-weighted for availability.")
    elif result["availability_multiplier"] >= 1.05:
        parts.append("Strong platform engagement (recent activity, responsive to recruiters).")

    return " ".join(parts)
