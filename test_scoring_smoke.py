"""
Smoke tests for the scoring pipeline.

Run with: python -m pytest tests/ -v
(or just: python tests/test_scoring_smoke.py)

These don't require the real candidate pool -- they construct minimal
synthetic profiles that exercise specific rules (keyword-stuffer trap,
strong genuine fit, consulting-only disqualifier, honeypot pattern,
availability down-weighting) and assert the expected direction of the
score, not an exact value. This is what a reviewer checking "does the
methodology actually do what the README claims" would run first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features as feat_mod
from src import scoring


def _base_candidate(**overrides):
    c = {
        "candidate_id": "CAND_0000000",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Engineer",
            "summary": "",
            "location": "Bangalore, Karnataka",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "Software Engineer",
            "current_company": "Acme Corp",
            "current_company_size": "201-500",
            "current_industry": "Technology",
        },
        "career_history": [{
            "company": "Acme Corp", "title": "Software Engineer",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 60,
            "is_current": True, "industry": "Technology", "company_size": "201-500",
            "description": "",
        }],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 80, "signup_date": "2024-01-01",
            "last_active_date": "2026-06-25", "open_to_work_flag": True,
            "profile_views_received_30d": 10, "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.5, "avg_response_time_hours": 24,
            "skill_assessment_scores": {}, "connection_count": 100,
            "endorsements_received": 10, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20, "max": 30},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": 10, "search_appearance_30d": 5,
            "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.8,
            "offer_acceptance_rate": -1, "verified_email": True,
            "verified_phone": True, "linkedin_connected": True,
        },
    }
    for k, v in overrides.items():
        c[k] = v
    return c


def score(candidate):
    f = feat_mod.extract(candidate)
    return scoring.score_candidate(f)


def test_keyword_stuffer_scores_low():
    """Skills listed as 'expert' with near-zero duration and no evidence in
    career_history should NOT score highly despite lots of AI-sounding tags."""
    c = _base_candidate(
        skills=[
            {"name": "NLP", "proficiency": "expert", "endorsements": 50, "duration_months": 1},
            {"name": "RAG", "proficiency": "expert", "endorsements": 50, "duration_months": 1},
            {"name": "Fine-tuning LLMs", "proficiency": "expert", "endorsements": 50, "duration_months": 1},
        ],
        career_history=[{
            "company": "Acme Corp", "title": "Marketing Manager",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 60,
            "is_current": True, "industry": "Marketing", "company_size": "201-500",
            "description": "Ran social media campaigns and email marketing funnels.",
        }],
    )
    r = score(c)
    assert r["final_score"] < 0.35, f"keyword stuffer scored too high: {r['final_score']}"


def test_genuine_fit_scores_high():
    """A candidate with real evidence of retrieval/vector-DB/eval work at a
    product company, India-based, reasonable notice, should score well."""
    c = _base_candidate(
        profile={**_base_candidate()["profile"], "current_title": "Machine Learning Engineer"},
        career_history=[{
            "company": "Swiggy", "title": "Machine Learning Engineer",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 60,
            "is_current": True, "industry": "Food Delivery", "company_size": "1001-5000",
            "description": (
                "Built and shipped an embedding-based retrieval system replacing "
                "keyword search, using FAISS for approximate nearest neighbor search "
                "and a hybrid BM25 + dense retrieval pipeline in production serving "
                "real users. Set up NDCG and MRR based offline evaluation and ran A/B "
                "tests to validate ranking model changes."
            ),
        }],
        skills=[
            {"name": "Python", "proficiency": "expert", "endorsements": 20, "duration_months": 60},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 15, "duration_months": 30},
        ],
    )
    r = score(c)
    assert r["final_score"] > 0.35, f"genuine fit scored too low: {r['final_score']}"


def test_consulting_only_career_penalized():
    c = _base_candidate(
        profile={**_base_candidate()["profile"], "current_company": "Wipro", "current_industry": "IT Services"},
        career_history=[{
            "company": "Wipro", "title": "Software Engineer",
            "start_date": "2018-01-01", "end_date": None, "duration_months": 96,
            "is_current": True, "industry": "IT Services", "company_size": "10001+",
            "description": "Built retrieval and ranking systems using embeddings and vector search.",
        }],
    )
    f = feat_mod.extract(c)
    assert f["is_consulting_only_career"] is True
    r = score(c)
    from src.disqualifiers import disqualifier_multiplier
    mult, reasons = disqualifier_multiplier(f)
    assert mult < 1.0
    assert any("consulting" in r for r in reasons)


def test_honeypot_detected():
    c = _base_candidate(
        skills=[
            {"name": "NLP", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "RAG", "proficiency": "expert", "endorsements": 5, "duration_months": 0},
            {"name": "LoRA", "proficiency": "advanced", "endorsements": 5, "duration_months": 1},
        ],
        profile={**_base_candidate()["profile"], "years_of_experience": 15.0},
        # career history covers only 2 years despite 15 yrs claimed
        career_history=[{
            "company": "Acme Corp", "title": "ML Engineer",
            "start_date": "2024-01-01", "end_date": None, "duration_months": 24,
            "is_current": True, "industry": "Technology", "company_size": "201-500",
            "description": "",
        }],
    )
    r = score(c)
    assert r["honeypot_flag"] is True
    assert r["final_score"] == 0.0


def test_inactive_unresponsive_candidate_downweighted():
    strong_active = _base_candidate(
        profile={**_base_candidate()["profile"], "current_title": "Machine Learning Engineer"},
        career_history=[{
            "company": "Swiggy", "title": "Machine Learning Engineer",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 60,
            "is_current": True, "industry": "Food Delivery", "company_size": "1001-5000",
            "description": "Built embedding retrieval, vector search with FAISS, NDCG offline eval in production.",
        }],
    )
    strong_inactive = _base_candidate(
        profile={**_base_candidate()["profile"], "current_title": "Machine Learning Engineer"},
        career_history=strong_active["career_history"],
        redrob_signals={
            **strong_active["redrob_signals"],
            "last_active_date": "2025-10-01",  # long inactive relative to 2026-07
            "open_to_work_flag": False,
            "recruiter_response_rate": 0.05,
            "interview_completion_rate": 0.1,
        },
    )
    r_active = score(strong_active)
    r_inactive = score(strong_inactive)
    assert r_inactive["final_score"] < r_active["final_score"]
    assert r_inactive["availability_multiplier"] < r_active["availability_multiplier"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
