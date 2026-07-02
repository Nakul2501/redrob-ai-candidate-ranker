"""
scoring.py
==========
Combines the extracted features (features.py) into a final candidate score.

COMPONENT WEIGHTS (the "feature importance" the hackathon asks for)
---------------------------------------------------------------------
    core_ml_fit             0.40   -- the four "absolutely need" clusters
    nice_to_have_fit         0.10   -- "won't reject you for" clusters
    seniority_experience_fit 0.20   -- years band, title, shipped-system evidence
    location_logistics_fit   0.15   -- Pune/Noida preference, India, relocation
    notice_period_fit        0.15   -- sub-30-day preference

These five weighted components sum to a base "fit_score" in [0, 1]. That
base score is then modified -- multiplicatively, not additively -- by:

    disqualifier_multiplier   in (0, 1]   -- JD's explicit "do NOT want" list
    availability_multiplier   in [0.45, 1.15] -- behavioral engagement (redrob_signals)

and finally honeypot detection acts as a hard gate (excluded outright, not
just down-weighted -- see honeypot.py for rationale).

    final_score = fit_score * disqualifier_multiplier * availability_multiplier
                  (or 0 / excluded if honeypot)

WHY WEIGHTS LOOK LIKE THIS (bias-mitigation choices, stated explicitly)
---------------------------------------------------------------------
- core_ml_fit dominates (0.40) because the JD is explicit that skills are
  "teachable" but this specific technical bar is the one hard requirement
  cluster ("Things you absolutely need").
- Institution/education tier is NOT used as a scoring input at all. The JD
  never mentions pedigree, and using it would introduce a prestige bias the
  brief doesn't ask for. field_of_study is used only as a very small
  corroborating nudge inside seniority_experience_fit, never as a gate.
- current_company brand name is NOT used as a positive signal (JD
  explicitly does not want "Google/Meta" credentialism -- see JD's opening
  section). Company data is only used to detect the consulting-only
  disqualifier, which the JD explicitly asks for.
- expected_salary_range is NOT used anywhere. The JD gives no budget, and
  scoring on salary expectations would be an unjustified, easily-biased
  filter not requested by the brief.
- anonymized_name is NEVER read by any scoring code (not present in
  features.py's returned dict at all beyond raw passthrough), which avoids
  any risk of the ranker picking up name-correlated proxies.
- search_appearance_30d / saved_by_recruiters_30d (popularity signals) are
  capped at very low weight in availability.py specifically to avoid a
  feedback loop that would compound existing visibility advantages.
"""

from __future__ import annotations

import math

from . import jd_profile as jd
from .availability import availability_multiplier
from .disqualifiers import disqualifier_multiplier
from .honeypot import is_honeypot

# Component weights -- edit here to retune.
#
# IMPORTANT STRUCTURAL CHOICE: these do NOT combine as one flat weighted sum
# of five independent components. If they did, a candidate with zero
# AI/ML/search relevance but a convenient location and short notice period
# could still land ~0.4+ purely on logistics -- which contradicts the JD's
# explicit stance ("we'd rather see 10 great matches than 1000 maybes").
#
# Instead:
#   base_score = TECH_WEIGHTS-weighted(core_ml_fit, nice_to_have_fit, seniority_experience_fit)
#   logistics_multiplier = 0.5 + 0.5 * avg(location_logistics_fit, notice_period_fit)   in [0.5, 1.0]
#   fit_score = base_score * logistics_multiplier
#
# So location and notice period can scale a technically-qualified candidate
# up or down by up to 2x, but can never manufacture a high score out of a
# candidate with no genuine technical/role fit. This mirrors how the JD
# actually talks about these dimensions: skills/production evidence are the
# hard bar ("Things you absolutely need"); location and notice period are
# framed as softer preferences ("Pune/Noida-preferred but flexible," "still
# in scope but the bar gets higher").
TECH_WEIGHTS = {
    "core_ml_fit": 0.55,
    "nice_to_have_fit": 0.15,
    "seniority_experience_fit": 0.30,
}
assert abs(sum(TECH_WEIGHTS.values()) - 1.0) < 1e-9

# kept for the audit report / backwards-compatible column naming
WEIGHTS = {
    **TECH_WEIGHTS,
    "location_logistics_fit": None,   # applied via logistics_multiplier, see above
    "notice_period_fit": None,        # applied via logistics_multiplier, see above
}

CORE_CLUSTER_WEIGHTS = {
    "embeddings_retrieval": 0.30,
    "vector_db_hybrid_search": 0.30,
    "eval_frameworks": 0.25,
    "python": 0.15,
}
assert abs(sum(CORE_CLUSTER_WEIGHTS.values()) - 1.0) < 1e-9


def _cluster_trust_score(trust_map: dict[str, float], terms: list[str]) -> float:
    """Sum of trust-adjusted weights for claimed skills matching a cluster's
    terms (substring match either direction), soft-capped via log so a
    candidate can't dominate a cluster by listing dozens of near-duplicate
    skill tags."""
    total = 0.0
    matched = 0
    for skill_name, trust in trust_map.items():
        if any(term in skill_name or skill_name in term for term in terms):
            total += trust
            matched += 1
    if matched == 0:
        return 0.0
    # diminishing returns past ~3 well-trusted matching skills
    return min(1.0, math.log1p(total) / math.log1p(3.0))


def _cluster_evidence_score(evidence_hit_count: int) -> float:
    """Evidence-pool (career/summary text) hit count -> [0,1], diminishing returns."""
    if evidence_hit_count <= 0:
        return 0.0
    return min(1.0, math.log1p(evidence_hit_count) / math.log1p(4.0))


def core_ml_fit(feat: dict) -> tuple[float, dict]:
    breakdown = {}
    score = 0.0
    for cname, weight in CORE_CLUSTER_WEIGHTS.items():
        terms = jd.CORE_CLUSTERS[cname]
        trust_component = _cluster_trust_score(feat["trust"], terms)
        evidence_component = _cluster_evidence_score(feat["core_cluster_hits"][cname])
        # evidence (what they DID) is weighted higher than trust-adjusted
        # skill tags (what they CLAIM) -- directly encodes the JD's
        # "reasoning about the gap between what the JD says and means."
        cluster_score = 0.45 * trust_component + 0.55 * evidence_component
        breakdown[cname] = round(cluster_score, 4)
        score += weight * cluster_score

    # Shipped-system bonus: direct evidence of having shipped a
    # ranking/search/recsys system to real users at scale, independent of
    # which specific cluster it fell into.
    shipped_bonus = min(0.15, math.log1p(feat["shipped_hits"]) / math.log1p(3.0) * 0.15)
    score = min(1.0, score + shipped_bonus)
    breakdown["shipped_system_bonus"] = round(shipped_bonus, 4)
    return round(score, 4), breakdown


def nice_to_have_fit(feat: dict) -> tuple[float, dict]:
    breakdown = {}
    scores = []
    for cname, terms in jd.NICE_TO_HAVE_CLUSTERS.items():
        trust_component = _cluster_trust_score(feat["trust"], terms)
        evidence_component = _cluster_evidence_score(feat["nice_cluster_hits"][cname])
        cluster_score = 0.4 * trust_component + 0.6 * evidence_component
        breakdown[cname] = round(cluster_score, 4)
        scores.append(cluster_score)
    avg = sum(scores) / len(scores) if scores else 0.0
    return round(avg, 4), breakdown


def seniority_experience_fit(feat: dict) -> tuple[float, dict]:
    yoe = feat["yoe"]
    # triangular credit: full credit at [5,9], peak at [6,8]
    if jd.EXPERIENCE_IDEAL_LOW <= yoe <= jd.EXPERIENCE_IDEAL_HIGH:
        years_score = 1.0
    elif jd.EXPERIENCE_BAND_LOW <= yoe < jd.EXPERIENCE_IDEAL_LOW:
        years_score = 0.75 + 0.25 * (yoe - jd.EXPERIENCE_BAND_LOW) / (jd.EXPERIENCE_IDEAL_LOW - jd.EXPERIENCE_BAND_LOW)
    elif jd.EXPERIENCE_IDEAL_HIGH < yoe <= jd.EXPERIENCE_BAND_HIGH:
        years_score = 1.0 - 0.25 * (yoe - jd.EXPERIENCE_IDEAL_HIGH) / (jd.EXPERIENCE_BAND_HIGH - jd.EXPERIENCE_IDEAL_HIGH)
    elif yoe < jd.EXPERIENCE_BAND_LOW:
        # below band -- soft decay, JD says band is flexible if other signals strong
        years_score = max(0.15, 0.75 - 0.15 * (jd.EXPERIENCE_BAND_LOW - yoe))
    else:
        # above band
        years_score = max(0.25, 0.75 - 0.08 * (yoe - jd.EXPERIENCE_BAND_HIGH))

    ml_years = feat["ml_career_months"] / 12.0
    # JD ideal: "4-5 [years] in applied ML/AI roles at product companies"
    ml_years_score = min(1.0, ml_years / 4.5)

    product_years = feat["product_company_months"] / 12.0
    product_score = min(1.0, product_years / max(yoe, 1) ) if yoe else 0.0

    title_score = 0.0
    if feat["title_is_relevant"]:
        title_score = 1.0
    elif feat["title_is_architecture_only"]:
        title_score = 0.3
    else:
        title_score = 0.35  # unrelated title, but career history may still carry evidence

    education_nudge = 0.05 if feat["education_relevant"] else 0.0

    score = (
        0.30 * years_score
        + 0.30 * ml_years_score
        + 0.15 * product_score
        + 0.20 * title_score
        + education_nudge
    )
    score = min(1.0, score)
    breakdown = {
        "years_band_score": round(years_score, 4),
        "applied_ml_years_score": round(ml_years_score, 4),
        "product_company_ratio_score": round(product_score, 4),
        "title_relevance_score": round(title_score, 4),
        "education_nudge": education_nudge,
        "ml_years_estimate": round(ml_years, 2),
    }
    return round(score, 4), breakdown


def location_logistics_fit(feat: dict) -> tuple[float, dict]:
    city = feat["city_token"]
    country = feat["country"]
    willing = feat["signals"].get("willing_to_relocate", False)

    if country == "india":
        if city in jd.PREFERRED_CITIES:
            base = 1.0
        elif city in jd.EXPLICIT_WELCOME_CITIES:
            base = 0.9
        elif city in jd.OTHER_INDIA_TECH_HUBS:
            base = 0.75
        else:
            base = 0.6  # other India city -- JD is hybrid/flexible, not restrictive
    else:
        base = 0.35 if willing else 0.15  # case-by-case, no visa sponsorship

    breakdown = {"base_location_score": round(base, 4), "country": country, "city": city, "willing_to_relocate": willing}
    return round(base, 4), breakdown


def notice_period_fit(feat: dict) -> tuple[float, dict]:
    days = feat["signals"].get("notice_period_days")
    if days is None:
        return 0.5, {"notice_period_days": None}
    if days <= jd.NOTICE_FULL_CREDIT_DAYS:
        score = 1.0
    elif days >= jd.NOTICE_ZERO_CREDIT_DAYS:
        score = 0.25
    else:
        frac = (days - jd.NOTICE_FULL_CREDIT_DAYS) / (jd.NOTICE_ZERO_CREDIT_DAYS - jd.NOTICE_FULL_CREDIT_DAYS)
        score = 1.0 - 0.75 * frac
    return round(score, 4), {"notice_period_days": days}


def score_candidate(feat: dict) -> dict:
    """Full scoring pipeline for one candidate's feature dict. Returns a
    dict with the final score plus the full audit trail (used for both the
    reasoning generator and the bias audit report)."""

    honeypot_flag, honeypot_sc, honeypot_reasons = is_honeypot(feat)

    core, core_bd = core_ml_fit(feat)
    nice, nice_bd = nice_to_have_fit(feat)
    seniority, seniority_bd = seniority_experience_fit(feat)
    location, location_bd = location_logistics_fit(feat)
    notice, notice_bd = notice_period_fit(feat)

    base_score = (
        TECH_WEIGHTS["core_ml_fit"] * core
        + TECH_WEIGHTS["nice_to_have_fit"] * nice
        + TECH_WEIGHTS["seniority_experience_fit"] * seniority
    )
    logistics_multiplier = 0.5 + 0.5 * (0.5 * location + 0.5 * notice)
    fit_score = base_score * logistics_multiplier

    dq_mult, dq_reasons = disqualifier_multiplier(feat)
    avail_mult, avail_notes = availability_multiplier(feat)

    if honeypot_flag:
        final = 0.0
    else:
        final = fit_score * dq_mult * avail_mult

    return {
        "candidate_id": feat["candidate_id"],
        "final_score": round(final, 6),
        "fit_score": round(fit_score, 6),
        "core_ml_fit": core,
        "nice_to_have_fit": nice,
        "seniority_experience_fit": seniority,
        "location_logistics_fit": location,
        "notice_period_fit": notice,
        "disqualifier_multiplier": dq_mult,
        "disqualifier_reasons": dq_reasons,
        "availability_multiplier": avail_mult,
        "availability_notes": avail_notes,
        "honeypot_flag": honeypot_flag,
        "honeypot_score": honeypot_sc,
        "honeypot_reasons": honeypot_reasons,
        "component_breakdown": {
            "core_ml_fit": core_bd,
            "nice_to_have_fit": nice_bd,
            "seniority_experience_fit": seniority_bd,
            "location_logistics_fit": location_bd,
            "notice_period_fit": notice_bd,
        },
    }
