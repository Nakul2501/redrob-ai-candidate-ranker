"""
features.py
============
Turns one raw candidate JSON record (matching candidate_schema.json) into a
flat, structured feature dict. No scoring decisions happen here -- this
module only extracts and normalizes; scoring.py decides what the numbers
mean.

Keeping extraction and scoring separate is what makes the "feature
importance" story honest: you can print out every candidate's feature dict
and see exactly what evidence the final score was built from.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from . import jd_profile as jd

_WORD_RE = re.compile(r"[a-z0-9+#.\-]+")


def _norm(text: str) -> str:
    return (text or "").lower()


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def build_text_blobs(candidate: dict) -> dict[str, str]:
    """Separate text pools. Kept separate (not one mega-blob) so scoring can
    weight "what they claim" (skills) differently from "what they actually
    did" (career_history descriptions) -- that separation is central to
    catching keyword-stuffers."""
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []

    headline = _norm(profile.get("headline", ""))
    summary = _norm(profile.get("summary", ""))
    current_title = _norm(profile.get("current_title", ""))

    career_text = " \n ".join(
        _norm(f"{c.get('title', '')} {c.get('description', '')}") for c in career
    )
    skills_text = " \n ".join(_norm(s.get("name", "")) for s in skills)

    return {
        "headline": headline,
        "summary": summary,
        "current_title": current_title,
        "career_text": career_text,
        "skills_text": skills_text,
        # "evidence" text = everything except the bare skills list. This is
        # the pool used to detect plain-language / shipped-system evidence,
        # deliberately excluding the skills tag list so a candidate can't
        # get credit here just by listing a keyword.
        "evidence_text": " \n ".join([headline, summary, career_text]),
        # "claims" text = everything including skills, used only for the
        # raw-keyword-presence check (which gets discounted by trust factor
        # separately -- see skill_trust below).
        "all_text": " \n ".join([headline, summary, career_text, skills_text]),
    }


def cluster_hit_count(text: str, terms: list[str]) -> int:
    """Count distinct terms from a cluster that appear as a substring hit in
    text. Phrase-aware (multi-word terms match as substrings, not just
    tokens), which is what lets loose natural-language phrasing match."""
    hits = 0
    for term in terms:
        if term in text:
            hits += 1
    return hits


def skill_trust_map(candidate: dict) -> dict[str, float]:
    """
    For every claimed skill, compute a trust-adjusted weight in [0, ~1.4]:

      trust = proficiency_weight * duration_credibility * assessment_credibility

    - proficiency_weight: self-reported level, base signal (weak alone).
    - duration_credibility: how long they say they've used it. A self-rated
      "expert" with 0-3 months duration is not credible -- this is exactly
      the CAND_0000001-style pattern (advanced/expert skills claimed with
      the underlying platform assessment score sitting in the 30s-60s).
    - assessment_credibility: if Redrob's own skill_assessment_scores has an
      entry for this skill, it is the strongest signal (an actual measured
      score, not self-report) and dominates the trust computation.

    Endorsements are intentionally given very little weight: they are the
    easiest signal to inflate/game and the JD gives no reason to trust them
    as evidence of real capability.
    """
    prof_weight = {"beginner": 0.35, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
    assessment = _safe_get(candidate, "redrob_signals", "skill_assessment_scores", default={}) or {}

    out: dict[str, float] = {}
    for s in candidate.get("skills", []) or []:
        name = _norm(s.get("name", ""))
        if not name:
            continue
        pw = prof_weight.get(s.get("proficiency", ""), 0.4)
        dur = s.get("duration_months", 0) or 0
        duration_cred = min(1.0, dur / 18.0)  # full credit at 18+ months of claimed use
        # small floor so a brand-new-but-real skill isn't zeroed out
        duration_cred = max(duration_cred, 0.15)

        assess_score = None
        for k, v in assessment.items():
            if _norm(k) == name:
                assess_score = v
                break

        if assess_score is not None:
            assess_cred = assess_score / 100.0
            # Verified assessment score dominates: 70% assessment, 30% the
            # self-report/duration story.
            trust = 0.7 * assess_cred + 0.3 * (pw * duration_cred)
        else:
            trust = pw * duration_cred

        # small endorsement bonus, capped, low weight, log-scaled so it
        # can't be gamed by a huge raw number
        endorsements = s.get("endorsements", 0) or 0
        import math
        endorsement_bonus = min(0.08, math.log1p(endorsements) / 60.0)
        trust = min(1.4, trust + endorsement_bonus)

        out[name] = round(trust, 4)
    return out


def extract(candidate: dict, today: date | None = None) -> dict[str, Any]:
    """Extract the full structured feature set for one candidate."""
    today = today or date.today()
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    text = build_text_blobs(candidate)
    trust = skill_trust_map(candidate)

    # ---- cluster evidence, computed on the *evidence* pool (excludes bare
    # skill tags) so plain-language / production-shipped candidates score
    # even if they never named the JD's exact vocabulary in a skills tag.
    evidence = text["evidence_text"]
    all_text = text["all_text"]

    core_cluster_hits = {
        cname: cluster_hit_count(evidence, terms) for cname, terms in jd.CORE_CLUSTERS.items()
    }
    core_cluster_hits_claims_only = {
        cname: cluster_hit_count(all_text, terms) for cname, terms in jd.CORE_CLUSTERS.items()
    }
    nice_cluster_hits = {
        cname: cluster_hit_count(evidence, terms) for cname, terms in jd.NICE_TO_HAVE_CLUSTERS.items()
    }

    shipped_hits = cluster_hit_count(evidence, jd.SHIPPED_SYSTEM_SIGNALS)
    research_only_hits = cluster_hit_count(evidence, jd.RESEARCH_ONLY_SIGNALS)

    cv_speech_robotics_hits = cluster_hit_count(all_text, jd.CV_SPEECH_ROBOTICS_CLUSTER)
    nlp_ir_hits = cluster_hit_count(all_text, jd.NLP_IR_CLUSTER)
    framework_surface_hits = cluster_hit_count(all_text, jd.FRAMEWORK_SURFACE_CLUSTER)

    # ---- title / seniority
    current_title = text["current_title"]
    title_is_relevant = any(k in current_title for k in jd.RELEVANT_TITLE_KEYWORDS)
    title_is_architecture_only = any(k in current_title for k in jd.ARCHITECTURE_ONLY_TITLE_KEYWORDS)

    # ---- career history derived stats
    total_months_claimed = sum((c.get("duration_months", 0) or 0) for c in career)
    yoe = profile.get("years_of_experience", 0) or 0
    consistency_ratio = (total_months_claimed / 12.0) / yoe if yoe > 0 else 1.0

    industries = [_norm(c.get("industry", "")) for c in career] + [_norm(profile.get("current_industry", ""))]
    companies = [_norm(c.get("company", "")) for c in career] + [_norm(profile.get("current_company", ""))]
    is_consulting_only_career = bool(career) and all(
        (ind in jd.CONSULTING_INDUSTRY_LABELS) or (comp in jd.KNOWN_CONSULTING_FIRMS)
        for ind, comp in zip(
            [_norm(c.get("industry", "")) for c in career],
            [_norm(c.get("company", "")) for c in career],
        )
    )

    # architecture-only for 18+ months: current role is architecture-titled
    # AND has run >= 18 months
    current_role = next((c for c in career if c.get("is_current")), None)
    architecture_only_18mo = False
    if current_role:
        cur_title_norm = _norm(current_role.get("title", ""))
        if any(k in cur_title_norm for k in jd.ARCHITECTURE_ONLY_TITLE_KEYWORDS):
            if (current_role.get("duration_months", 0) or 0) >= 18:
                architecture_only_18mo = True

    # title-chaser: 3+ role changes, each held < 18 months, with escalating
    # level words
    short_stints = [c for c in career if (c.get("duration_months", 0) or 0) < 18]
    title_chaser = len(career) >= 3 and len(short_stints) >= max(2, len(career) - 1)

    # recent-LLM-only-experience disqualifier proxy: framework surface hits
    # present, core infra clusters (vector db / eval) absent, and total
    # experience is itself short
    recent_llm_only_pattern = (
        framework_surface_hits > 0
        and core_cluster_hits.get("vector_db_hybrid_search", 0) == 0
        and core_cluster_hits.get("eval_frameworks", 0) == 0
    )

    # closed-source-only proxy
    closed_source_only_proxy = (
        yoe >= 5
        and (signals.get("github_activity_score", -1) in (None, -1))
        and cluster_hit_count(evidence, jd.NICE_TO_HAVE_CLUSTERS["open_source"]) == 0
    )

    # ---- applied-ML tenure proxy: months spent in roles whose title or
    # description shows AI/ML/search/ranking evidence, and months spent at
    # non-consulting ("product") companies. These back the JD's "6-8 years
    # total, of which 4-5 are in applied ML/AI roles at product companies"
    # ideal-candidate description.
    ml_career_months = 0
    product_company_months = 0
    for c in career:
        role_text = _norm(f"{c.get('title', '')} {c.get('description', '')}")
        dur = c.get("duration_months", 0) or 0
        role_is_ml = (
            any(k in role_text for k in jd.RELEVANT_TITLE_KEYWORDS)
            or cluster_hit_count(role_text, jd.CORE_CLUSTERS["embeddings_retrieval"]) > 0
            or cluster_hit_count(role_text, jd.CORE_CLUSTERS["vector_db_hybrid_search"]) > 0
            or cluster_hit_count(role_text, jd.SHIPPED_SYSTEM_SIGNALS) > 0
        )
        if role_is_ml:
            ml_career_months += dur
        ind = _norm(c.get("industry", ""))
        comp = _norm(c.get("company", ""))
        if not (ind in jd.CONSULTING_INDUSTRY_LABELS or comp in jd.KNOWN_CONSULTING_FIRMS):
            product_company_months += dur

    # ---- education (used lightly -- see bias note in scoring.py)
    fields_of_study = " ".join(_norm(e.get("field_of_study", "")) for e in education)
    education_relevant = any(
        k in fields_of_study for k in ["computer science", "data science", "statistics",
                                        "machine learning", "artificial intelligence",
                                        "information technology", "mathematics"]
    )

    # ---- location
    location = _norm(profile.get("location", ""))
    country = _norm(profile.get("country", ""))
    city_token = location.split(",")[0].strip()

    # ---- honeypot-style internal-consistency flags (kept raw here; scored
    # in honeypot.py)
    impossible_skill_claims = sum(
        1
        for s in candidate.get("skills", []) or []
        if s.get("proficiency") in ("advanced", "expert") and (s.get("duration_months", 0) or 0) <= 2
    )

    last_active = _parse_date(signals.get("last_active_date"))
    days_since_active = (today - last_active).days if last_active else None

    return {
        "candidate_id": candidate.get("candidate_id"),
        "profile": profile,
        "career": career,
        "education": education,
        "signals": signals,
        "text": text,
        "trust": trust,
        "core_cluster_hits": core_cluster_hits,
        "core_cluster_hits_claims_only": core_cluster_hits_claims_only,
        "nice_cluster_hits": nice_cluster_hits,
        "shipped_hits": shipped_hits,
        "research_only_hits": research_only_hits,
        "cv_speech_robotics_hits": cv_speech_robotics_hits,
        "nlp_ir_hits": nlp_ir_hits,
        "framework_surface_hits": framework_surface_hits,
        "title_is_relevant": title_is_relevant,
        "title_is_architecture_only": title_is_architecture_only,
        "total_months_claimed": total_months_claimed,
        "ml_career_months": ml_career_months,
        "product_company_months": product_company_months,
        "yoe": yoe,
        "consistency_ratio": consistency_ratio,
        "is_consulting_only_career": is_consulting_only_career,
        "architecture_only_18mo": architecture_only_18mo,
        "title_chaser": title_chaser,
        "recent_llm_only_pattern": recent_llm_only_pattern,
        "closed_source_only_proxy": closed_source_only_proxy,
        "education_relevant": education_relevant,
        "location": location,
        "country": country,
        "city_token": city_token,
        "impossible_skill_claims": impossible_skill_claims,
        "days_since_active": days_since_active,
    }
