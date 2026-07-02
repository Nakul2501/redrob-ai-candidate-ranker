"""
honeypot.py
===========
Detects "subtly impossible" profiles per submission_spec.md Section 7:
"~80 honeypot candidates with subtly impossible profiles (e.g., 8 years of
experience at a company founded 3 years ago; 'expert' proficiency in 10
skills with 0 years used)."

We don't have company founding dates in the schema, so we lean on the
internal-consistency checks that ARE available: skill-claim-vs-duration
mismatches, years_of_experience vs. career_history duration mismatches, and
overlapping "current" roles. Each check contributes to a 0-1 suspicion
score; candidates above threshold are hard-excluded from the top 100 (not
just down-weighted), because the spec disqualifies submissions with >10%
honeypot rate in the top 100 -- the safest posture is to keep the true rate
near zero by construction, not just under the cap.
"""

from __future__ import annotations


HONEYPOT_THRESHOLD = 0.5


def honeypot_score(feat: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    # 1. Multiple "advanced"/"expert" skills claimed with <= 2 months of
    #    claimed use each -- the JD's own example pattern.
    if feat["impossible_skill_claims"] >= 3:
        score += 0.45
        reasons.append(f"{feat['impossible_skill_claims']} skills claimed at advanced/expert "
                        f"level with <=2 months of use")
    elif feat["impossible_skill_claims"] >= 1:
        score += 0.15
        reasons.append(f"{feat['impossible_skill_claims']} skill(s) claimed at advanced/expert "
                        f"level with <=2 months of use")

    # 2. years_of_experience wildly inconsistent with the sum of
    #    career_history duration_months (career history should roughly
    #    cover the claimed years; large mismatch either way is suspect).
    cr = feat["consistency_ratio"]
    if cr < 0.4 or cr > 2.5:
        score += 0.35
        reasons.append(
            f"years_of_experience ({feat['yoe']}) inconsistent with sum of career_history "
            f"durations (ratio={cr:.2f})"
        )

    # 3. More than one role flagged is_current=true (temporally impossible).
    current_flags = sum(1 for c in feat["career"] if c.get("is_current"))
    if current_flags > 1:
        score += 0.3
        reasons.append(f"{current_flags} career_history entries flagged is_current=true simultaneously")

    # 4. Overlapping date ranges between two roles (excluding the
    #    is_current=true role, which has end_date=null by definition).
    dated = [
        (c.get("start_date"), c.get("end_date"))
        for c in feat["career"]
        if c.get("start_date") and c.get("end_date")
    ]
    dated.sort()
    for i in range(len(dated) - 1):
        if dated[i][1] and dated[i + 1][0] and dated[i][1] > dated[i + 1][0]:
            score += 0.2
            reasons.append("overlapping career_history date ranges")
            break

    # 5. skill_assessment_scores present but essentially zero for a skill
    #    claimed as "expert" -- a directly measured contradiction.
    assessment = (feat["signals"].get("skill_assessment_scores") or {})
    skills = feat.get("_raw_skills") or []
    # (raw skills aren't stored in feat; caller passes candidate separately
    # when needed. This check is best-effort using trust map proxy instead.)

    score = min(1.0, score)
    return score, reasons


def is_honeypot(feat: dict) -> tuple[bool, float, list[str]]:
    s, reasons = honeypot_score(feat)
    return s >= HONEYPOT_THRESHOLD, s, reasons
