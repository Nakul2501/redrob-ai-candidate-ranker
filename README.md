# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A transparent, rule-based ranking system for the "Senior AI Engineer —
Founding Team" JD at Redrob AI. No GPU, no network calls, no LLM API calls
during ranking — a pure-Python/pandas scoring pipeline that runs the full
100,000-candidate pool in **~100 seconds** and **<1GB RAM**, well inside the
spec's 5-minute / 16GB / CPU-only / no-network budget.

## Why rule-based, not a trained model or LLM-per-candidate

1. **The compute budget makes LLM-per-candidate a non-starter.** The spec
   is explicit: no hosted LLM calls, no GPU, 5 minutes for 100K candidates.
   Even a fast local model can't clear that bar reliably.
2. **The JD's own trap is keyword-matching, not model choice.**
   `sample_submission.csv` ranks an *HR Manager* #1 for having "9 AI core
   skills" — that's not a modeling failure, it's a feature-design failure.
   Getting the *features* right (what counts as evidence, what counts as a
   red flag) matters more here than the scoring function on top of them.
3. **Transparency is explicitly rewarded.** Stage 4 review checks
   reasoning quality, methodology coherence, and whether a human did real
   engineering. A fully auditable rule system, where every score component
   traces back to a specific JD sentence, is easier to defend at Stage 5
   (the interview) than "the embedding model decided."

## Quickstart

```bash
pip install -r requirements.txt

# Full run against the real 100K pool (place candidates.jsonl.gz in data/,
# or point --candidates at wherever you unpacked it):
python rank.py --candidates ./data/candidates.jsonl.gz --out ./submission.csv --audit-dir ./audit

# Validate before uploading:
python validate_submission.py ./submission.csv   # copy of the organizer's validator
```

Single reproduction command (matches `submission_metadata_template.yaml`):

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runs in ~100s / <1GB RAM for 100K candidates on a single CPU core (measured
on this environment: 100,000 synthetic records processed in 97–102s, peak
RSS ~730MB — see `docs/benchmark.md` if you want to reproduce the
measurement).

### Demo run (bundled 50-candidate sample)

```bash
python rank.py --candidates data/sample_candidates.json --out demo_submission.csv --top-k 50 --audit-dir demo_audit
```

This is what's included as `demo_submission.csv` and `demo_audit/` in this
repo — generated from the actual `sample_candidates.json` you were given,
so you can sanity-check the reasoning against real profiles immediately.

## Repo layout

```
rank.py                  # CLI entrypoint (the single reproduction command)
src/
  jd_profile.py           # hand-curated JD requirements: keyword clusters, weights, thresholds
  io_utils.py              # streaming JSON/JSONL/JSONL.GZ reader (memory-safe for 100K+ rows)
  features.py               # raw candidate JSON -> structured feature dict (no scoring decisions)
  scoring.py                 # combines features into the final score (the "model")
  disqualifiers.py            # JD's explicit "do NOT want" list, as graduated penalties
  availability.py              # behavioral engagement multiplier from redrob_signals
  honeypot.py                   # impossible-profile detector (hard-excludes from top 100)
  reasoning.py                   # generates the grounded, per-row `reasoning` text
  bias_audit.py                   # feature-importance + subgroup-disparity reports
tests/test_scoring_smoke.py  # unit tests against known-good/known-bad synthetic profiles
data/sample_candidates.json  # the 50-candidate sample you were given (bundled for the demo run)
Dockerfile                   # sandbox reproducibility (see submission_spec Section 10.5)
demo_submission.csv          # output of the demo run above
demo_audit/                  # audit reports from the demo run
```

## Methodology (≤200 words, for `submission_metadata.yaml`)

Rule-based ranker over five components, computed per candidate from
`career_history` descriptions (not just `skills` tags — this is what
catches keyword-stuffers and rewards plain-language Tier-5 candidates who
never say "RAG" but clearly shipped one). Core technical fit (embeddings,
vector DB/hybrid search, Python, eval frameworks — the JD's "absolutely
need" list) and role/seniority fit combine additively into a base score;
location and notice-period fit apply as a bounded 0.5–1.0x multiplier
rather than an independent additive term, so logistics can't manufacture a
high score for a technically-irrelevant candidate. Skill claims are
trust-discounted using Redrob's own `skill_assessment_scores` and claimed
`duration_months` — a self-rated "expert" with 1 month of use and no
verified assessment carries little weight. The JD's explicit disqualifier
list (consulting-only careers, architecture-only tenure, research-only
backgrounds, CV/speech/robotics-without-NLP, framework-surface-only
profiles) applies as graduated multiplicative penalties. A behavioral
engagement multiplier from `redrob_signals` down-weights disengaged
candidates. Honeypot detection (skill/duration/experience-consistency
checks) hard-excludes suspicious profiles from the top 100. Runtime:
~100s / <1GB RAM for 100K candidates on CPU.

## What each JD requirement maps to in code

| JD statement | Where it's encoded |
|---|---|
| "Things you absolutely need" (embeddings, vector DB, Python, eval frameworks) | `jd_profile.CORE_CLUSTERS`, `scoring.core_ml_fit`, weight 0.55 of base_score |
| "Things we'd like you to have but won't reject you for" | `jd_profile.NICE_TO_HAVE_CLUSTERS`, `scoring.nice_to_have_fit`, weight 0.15 |
| "6-8 years... 4-5 in applied ML/AI roles at product companies" | `scoring.seniority_experience_fit`, weight 0.30, uses `ml_career_months`/`product_company_months` proxies |
| "Pure research... no production deployment — we will not move forward" | `disqualifiers.py`: `research_only_hits >= 2 and shipped_hits == 0` → 0.15x |
| "AI experience... recent LangChain-to-OpenAI, no pre-LLM production experience" | `disqualifiers.py`: `recent_llm_only_pattern and yoe < 3` → 0.35x |
| "Hasn't written production code in 18 months (architecture/tech lead)" | `disqualifiers.py`: `architecture_only_18mo` → 0.4x |
| "Only worked at consulting firms (TCS, Infosys, Wipro...)" | `disqualifiers.py`: `is_consulting_only_career` → 0.3x |
| "CV, speech, or robotics without significant NLP/IR exposure" | `disqualifiers.py`: `cv_speech_robotics_hits >= 2 and nlp_ir_hits == 0` → 0.45x |
| "Entirely closed-source... without external validation" | `disqualifiers.py`: `closed_source_only_proxy` → 0.85x |
| "Title-chasers... switching every 1.5 years" | `disqualifiers.py`: `title_chaser` → 0.7x |
| "Framework enthusiasts... LangChain tutorials, no systems depth" | `disqualifiers.py`: `framework_surface_hits >= 2` and no infra/eval evidence → 0.6x |
| "Pune/Noida-preferred but flexible... no visa sponsorship" | `scoring.location_logistics_fit`, applied as part of the 0.5–1.0x logistics multiplier |
| "Sub-30-day notice preferred... 30+ still in scope" | `scoring.notice_period_fit`, same multiplier |
| "Perfect-on-paper but hasn't logged in for 6 months, 5% response rate — down-weight" | `availability.py`, multiplier 0.45–1.15x on the whole score |
| "~80 honeypot candidates with subtly impossible profiles" | `honeypot.py`, hard-excludes from output entirely |
| "Reasoning... specific facts, no hallucination, rank consistency" | `reasoning.py`, builds every sentence from real extracted fields, never a fixed template |

## Bias-mitigation choices (explicit, not incidental)

- **Institution/education tier is not scored.** The JD never mentions
  pedigree; using it would introduce prestige bias the brief doesn't ask
  for. `field_of_study` contributes a small (+0.05, capped) nudge only.
- **Company brand is not a positive signal.** The JD explicitly pushes back
  on "Google/Meta" credentialism. Company name is only used to detect the
  JD's own consulting-firm disqualifier — a rule the JD explicitly requests.
- **Salary expectations are never scored.** No budget was given in the JD;
  filtering on `expected_salary_range_inr_lpa` would be an unjustified,
  easily-biased gate not requested by the brief.
- **`anonymized_name` is never read** by any scoring code, closing off any
  path for a name-correlated proxy to leak into the score.
- **Popularity signals capped low.** `search_appearance_30d` and
  `saved_by_recruiters_30d` get very low weight in `availability.py`
  specifically to avoid compounding an existing visibility advantage
  (a feedback-loop bias, not a fit signal).
- **Skill *claims* are weighted below skill *evidence*.** Within each core
  cluster, career-history/summary evidence is weighted 0.55 vs. 0.45 for
  trust-adjusted skill tags — directly countering keyword-stuffing.
- **`bias_audit.py`** ships a post-hoc subgroup report (score/selection
  rate by country, company size, industry) so disparities are visible and
  attributable rather than hidden. It doesn't change ranking; it's a
  diagnostic you should read before submitting, and mention in your
  Stage-5 interview if asked how you checked for bias.

## Honeypot handling

Section 7 of the submission spec caps honeypot rate at 10% of the top 100.
Rather than aim under that cap, `honeypot.py` tries to keep the *true* rate
near zero by hard-excluding any candidate whose internal-consistency score
crosses a threshold (multiple "expert" skills claimed with ≤2 months of
use; `years_of_experience` inconsistent with the sum of `career_history`
durations; overlapping/duplicate "current" roles). This can't check the
"company founded 3 years ago, 8 years there" pattern from the spec's
example because `candidate_schema.json` doesn't include company founding
dates — noted as a known gap, not silently ignored.

## Known limitations / next steps if extending this

- Concept matching is lexicon-based (curated keyword/phrase clusters in
  `jd_profile.py`), not embedding-based semantic similarity. This keeps the
  pipeline fully deterministic, dependency-light, and trivially within the
  compute budget, but it will miss true paraphrases the lexicon doesn't
  anticipate (e.g. a candidate who describes retrieval work using
  vocabulary not in `CORE_CLUSTERS`). Expanding the lexicon based on what
  the audit report's low-`core_ml_fit`-but-high-`shipped_hits` candidates
  actually wrote is the highest-leverage next iteration.
- `ml_career_months` / `product_company_months` are proxies computed from
  title+description keyword matches per role, not ground truth.
- The honeypot detector is necessarily incomplete given the schema (see
  above) — it should be re-validated against the real 100K pool's honeypot
  rate once you have it (there's no ground-truth honeypot label to check
  against locally, so this can only be sanity-checked qualitatively).
- No test set from the hidden ground truth exists locally, so all
  validation here is against the JD's own stated preferences and the
  bundled 50-candidate sample — not against NDCG/MAP directly.

## Tests

```bash
python tests/test_scoring_smoke.py
```

Five smoke tests assert the *direction* the JD explicitly cares about:
keyword-stuffer scores low, genuine shipped-system fit scores high,
consulting-only career gets penalized, honeypot pattern gets excluded,
disengaged candidate gets down-weighted relative to an identical-but-active
one.

## Sandbox

`Dockerfile` builds a minimal image that runs `rank.py` against the bundled
50-candidate sample (`data/sample_candidates.json`) end-to-end, per
submission_spec.md Section 10.5 ("accept a small candidate sample, run
end-to-end, produce a ranked CSV, complete within budget"). Build and push
this to your chosen registry, or deploy `rank.py` + `src/` directly to
HuggingFace Spaces / Streamlit Cloud / Colab — whichever you already have
credentials for. This repo doesn't hard-code a platform choice.

```bash
docker build -t redrob-ranker .
docker run --rm -v $(pwd)/out:/app/out redrob-ranker
```
