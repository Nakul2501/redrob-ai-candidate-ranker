"""
disqualifiers.py
=================
Implements the JD's explicit "we will / probably will not move forward"
list as a multiplicative penalty on the final score, rather than a binary
exclude. The JD uses graduated language ("we will not," "we will probably
not"), so we mirror that with graduated multipliers instead of hard
zeroing -- a heuristic detector can misfire, and a multiplicative penalty
lets a candidate with an overwhelming core-fit score still surface if the
disqualifier signal is a false positive, while still pushing genuine misses
far down the list.

Every rule below cites the JD sentence it encodes.
"""

from __future__ import annotations

from . import jd_profile as jd


def disqualifier_multiplier(feat: dict) -> tuple[float, list[str]]:
    mult = 1.0
    reasons: list[str] = []

    # JD: "If you've spent your career in pure research environments...
    # without any production deployment -- we will not move forward."
    if feat["research_only_hits"] >= 2 and feat["shipped_hits"] == 0:
        mult *= 0.15
        reasons.append("career reads as research-only with no evidence of production deployment")

    # JD: "If your 'AI experience' consists primarily of recent (<12 months)
    # projects using LangChain to call OpenAI -- we will probably not move
    # forward, unless you can demonstrate substantial pre-LLM-era ML
    # production experience."
    if feat["recent_llm_only_pattern"] and feat["yoe"] < 3:
        mult *= 0.35
        reasons.append("AI experience appears limited to framework-level LLM API usage, "
                        "no vector-DB/eval-infra depth, and limited overall experience")

    # JD: "If you are a senior engineer who hasn't written production code
    # in the last 18 months because you've moved into 'architecture' or
    # 'tech lead' roles -- we will probably not move forward. This role
    # writes code."
    if feat["architecture_only_18mo"]:
        mult *= 0.4
        reasons.append("current role has been architecture/tech-lead titled for 18+ months")

    # JD: "People who have only worked at consulting firms (TCS, Infosys,
    # Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career."
    if feat["is_consulting_only_career"]:
        mult *= 0.3
        reasons.append("entire career history is at IT-services/consulting firms")

    # JD: "People whose primary expertise is computer vision, speech, or
    # robotics without significant NLP/IR exposure."
    if feat["cv_speech_robotics_hits"] >= 2 and feat["nlp_ir_hits"] == 0:
        mult *= 0.45
        reasons.append("profile centers on computer vision/speech/robotics with no NLP/IR exposure")

    # JD: "People whose work has been entirely on closed-source proprietary
    # systems for 5+ years... without external validation."
    if feat["closed_source_only_proxy"]:
        mult *= 0.85  # mildest penalty -- weakest-evidence disqualifier
        reasons.append("5+ years experience with no GitHub activity or open-source signal "
                        "(no external validation)")

    # JD: "Title-chasers... switching companies every 1.5 years."
    if feat["title_chaser"]:
        mult *= 0.7
        reasons.append("career history shows a pattern of short (<18mo) stints across roles")

    # JD: "Framework enthusiasts... GitHub full of LangChain tutorials."
    if feat["framework_surface_hits"] >= 2 and feat["core_cluster_hits"]["vector_db_hybrid_search"] == 0 \
            and feat["core_cluster_hits"]["eval_frameworks"] == 0:
        mult *= 0.6
        reasons.append("heavy on trendy-framework surface area, no systems-depth evidence "
                        "(vector DB / eval infra)")

    # Outside India + not willing to relocate -- JD: "Outside India:
    # case-by-case, but we don't sponsor work visas."
    if feat["country"] and feat["country"] != "india" and not feat["signals"].get("willing_to_relocate", False):
        mult *= 0.5
        reasons.append("based outside India and not flagged willing to relocate (no visa sponsorship)")

    return round(mult, 4), reasons
