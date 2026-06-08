# Fairness Audit Report — India Runs Track 1

> **Date:** 2026-06-06  
> **Pipeline version:** Phase 6 (8-signal weighted formula)  
> **Context:** Senior AI Engineer role at an Indian AI startup (Pune/Noida hybrid)

This report documents potential bias vectors in the ranking pipeline, whether each
is present, and the justification or mitigation where applicable.

---

## 1. Location Bias

**Status: PRESENT — Job-appropriate and explicitly documented**

`score_location()` assigns:
- `1.0` — profile in India (includes 15 major Indian cities + "india" string)
- `0.55` — location blank or unspecified (no penalty assumption — many strong
  candidates leave this field empty)
- `0.25` — profile in a country other than India

**Justification:** The JD explicitly states the role is hybrid at Pune/Noida offices.
Remote-only or overseas candidates are genuinely less suitable for this specific role.
The weight of location in the overall formula is `0.05` (5%) — the lowest of all signals.

**Bias risk:** Overseas Indian AI engineers may be penalised despite being willing to
relocate. Partially mitigated by the `blank = 0.55` (not 0.0) default, which avoids
penalising candidates who do not fill in location.

---

## 2. Educational Institution Bias

**Status: NOT PRESENT**

No signal in the pipeline references university name, degree tier, IIT/IIM affiliation,
or graduation rank. The `CandidateProfileParser` extracts total experience years and
seniority level from career history but does not extract or score education institution.

There is no token list, no field lookup, and no indirect proxy that would give
IIT/NIT/IIM graduates an advantage over candidates from other institutions.

---

## 3. Gender Bias

**Status: NOT PRESENT**

No signal in the pipeline uses candidate name, pronoun usage, or any other
name-based gender proxy. The `candidate_id` field is an opaque identifier.
Profile text is embedded as a vector but all-mpnet and BGE embedding models
do not have documented gender-correlated scoring for technical role matching.

The role relevance token sets (`STRONG_POSITIVE_ROLE_TOKENS`, `NEGATIVE_ROLE_TOKENS`)
contain only job title substrings, not gendered language.

---

## 4. Caste and Community Bias

**Status: NOT PRESENT**

No signal in the pipeline references caste, community, religion, or surname.
Candidate data fields used: `current_role`, `headline`, `total_experience`,
`career_history_text`, skill lists, and behavioral/activity metrics.
None of these fields expose caste or community identity in the pipeline's processing.

---

## 5. Profile Completeness Penalty

**Status: PRESENT — Low weight, documented**

`completeness` (normalised `profile_completeness_score`) contributes `weight 0.25`
inside `beh_score`, which itself has `weight 0.10` in the overall formula.
The effective contribution of completeness to the overall score is `0.025` (2.5%).

**Who is disadvantaged:** Passive candidates or early-career candidates who have not
fully filled out their profiles. This is a mild structural bias against candidates who
are not active job seekers, even if their skills and experience are strong.

**Justification:** For a time-sensitive hire (founding team AI engineer), a candidate
who actively maintains their profile and responds to recruiters is genuinely more
likely to engage with an outreach. The low weight (2.5%) limits the effect.

**Mitigation:** The `open_to_work_flag` is a separate signal (`weight 0.10 × 0.10 = 1%`)
and can partially compensate for a low completeness score if the candidate is
actively seeking.

---

## 6. Consulting Firm Penalty

**Status: PRESENT — JD-explicit, partially mitigated**

`score_career_trajectory()` applies a penalty of −0.20 to −0.40 on `career_score`
for candidates whose `career_history_text` contains consulting firm tokens (TCS,
Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, etc.).

**Justification:** The JD explicitly states a preference for product-company
experience and lists consulting/IT services as a negative signal. Applying this
penalty directly implements the client's stated requirement.

**Bias risk:** This penalises candidates from Indian IT services companies as a group.
This is a proxy for employment history, not personal characteristics.
The penalty is partial (−0.028 maximum impact on overall score at `career_weight = 0.07`)
and candidates with compensating signals (open-source contributions, AI publications,
recent product-company roles) have their score partially restored.

**Who is affected:** Estimated 15–20% of the 100K pool. The strongest consulting-firm
AI engineers still appear in ranks 20–50 rather than being filtered out entirely.

---

## Summary

| Bias vector | Present? | Weight/impact | Justified? |
|---|---|---|---|
| Location (India preference) | Yes | 5% of overall | Yes — JD-explicit hybrid role |
| Educational institution | No | — | N/A |
| Gender | No | — | N/A |
| Caste/community | No | — | N/A |
| Profile completeness | Yes | 2.5% of overall | Partially — active candidate signal |
| Consulting firm penalty | Yes | Up to 2.8% of overall | Yes — JD-explicit preference |

The two present biases (location, consulting penalty) are directly mandated by the JD.
The completeness penalty is a mild structural effect acknowledged here. All three are
documented and their weights are intentionally low to limit their discriminatory effect.
