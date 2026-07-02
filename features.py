"""
availability.py
================
JD (hackathon note): "Your ranking system should also weigh behavioral
signals -- a perfect-on-paper candidate who hasn't logged in for 6 months
and has a 5% recruiter response rate is, for hiring purposes, not actually
available. Down-weight them appropriately."

redrob_signals_doc.docx: "These behavioral signals are often more
predictive of whether a candidate can actually be hired than their static
profile... incorporate them as a multiplier or modifier on top of
skill-match scoring."

We build a bounded multiplier in [0.45, 1.15] so behavioral signals can
meaningfully suppress an unreachable candidate but can never let engagement
alone outrank a genuinely unqualified profile (the multiplier is applied on
top of the fit score, not added to it).

Bias note: search_appearance_30d and saved_by_recruiters_30d are given very
low weight deliberately. They reflect how often *other* recruiters already
looked at someone, which mostly measures profile popularity/visibility --
using them heavily would let already-popular profiles compound their
advantage independent of actual fit, which is a feedback-loop bias we don't
want to introduce.
"""

from __future__ import annotations


def availability_multiplier(feat: dict) -> tuple[float, list[str]]:
    s = feat["signals"]
    notes: list[str] = []
    component = 0.0  # accumulates in roughly [-1, +1], then mapped to multiplier

    # Recency of activity -- strongest signal, most directly matches the
    # JD's own example ("hasn't logged in for 6 months").
    days = feat["days_since_active"]
    if days is None:
        recency_score = -0.1
    elif days <= 14:
        recency_score = 0.35
    elif days <= 30:
        recency_score = 0.2
    elif days <= 90:
        recency_score = 0.0
    elif days <= 180:
        recency_score = -0.3
        notes.append(f"inactive for {days} days")
    else:
        recency_score = -0.55
        notes.append(f"inactive for {days}+ days")
    component += recency_score

    # Open to work
    if s.get("open_to_work_flag"):
        component += 0.15
    else:
        component -= 0.1
        notes.append("not flagged open_to_work")

    # Recruiter response rate
    rr = s.get("recruiter_response_rate")
    if rr is not None:
        component += (rr - 0.4) * 0.4  # centered around a "typical" 0.4
        if rr < 0.15:
            notes.append(f"very low recruiter response rate ({rr:.2f})")

    # Interview completion rate -- shows up when they do get to interview stage
    icr = s.get("interview_completion_rate")
    if icr is not None:
        component += (icr - 0.5) * 0.2
        if icr < 0.3:
            notes.append(f"low interview completion rate ({icr:.2f})")

    # Offer acceptance rate: -1 sentinel means "no history," treat as
    # neutral, not negative.
    oar = s.get("offer_acceptance_rate")
    if oar is not None and oar >= 0:
        component += (oar - 0.5) * 0.1

    # Verification / trust (small weight -- basic hygiene, not a quality signal)
    verified_count = sum([
        bool(s.get("verified_email")),
        bool(s.get("verified_phone")),
        bool(s.get("linkedin_connected")),
    ])
    component += (verified_count - 1.5) * 0.02

    # Profile completeness -- small weight
    pc = s.get("profile_completeness_score")
    if pc is not None:
        component += (pc - 70) / 100.0 * 0.08

    # Light-touch popularity signal (deliberately low weight -- see bias note)
    saved = s.get("saved_by_recruiters_30d", 0) or 0
    component += min(0.05, saved / 200.0)

    # Map component (~[-1, 1]) to a multiplier in [0.45, 1.15]
    multiplier = 1.0 + component * 0.4
    multiplier = max(0.45, min(1.15, multiplier))
    return round(multiplier, 4), notes
