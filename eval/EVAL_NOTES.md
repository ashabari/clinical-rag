# Evaluation notes: run-to-run variance and a rejected prompt change

This file documents two findings from the post-RRF-fix correction pass
that affect how the headline eval numbers should be read, but don't
change the code itself (the system prompt change described below was
tested and reverted -- the committed `src/prompt_builder.py` is
unchanged from before this investigation).

## 1. Run-to-run variance in LLM-judged scores

Three consecutive runs of `python -m eval.run_eval` against the
identical, committed pipeline (commit 697ba7a) produced:

| run | mean faithfulness | mean relevance |
|-----|--------------------|-----------------|
| A   | 0.738              | 0.898           |
| C   | 0.756              | 0.862           |
| D   | 0.753              | 0.919           |
| **range** | **0.738-0.756** | **0.862-0.919** |

`retrieval_match` -- which involves no LLM sampling -- was **bit-identical
across all three runs**: 75% (6/8), the same 6 questions matching every
time. So retrieval itself is fully deterministic; all of the above
variance comes from generation + LLM-judge sampling.

By category (faithfulness / relevance), min-max across the 3 runs:

| category  | faithfulness | relevance   |
|-----------|--------------|-------------|
| ambiguous | 0.40 - 0.50  | 0.67 - 0.79 |
| factoid   | 0.68 - 0.73  | 0.94 - 0.96 |
| multi_hop | 0.70 - 0.80  | 0.78 - 0.92 |
| negation  | 0.93 - 0.98  | 0.83 - 1.00 |
| refusal   | 0.93 - 1.00  | 1.00        |

**Implication:** a single eval run's headline numbers carry roughly
+/-0.05-0.1 of per-category noise that has nothing to do with the
pipeline or any change under test. This is a *separate* issue from the
self-judging bias concern (Claude judging Claude's own output) -- it's
plain sampling variance, and it means any reported number should be
read as a range or a multi-run average, not a single-run point estimate.

## 2. Tested and rejected: softened "never use outside knowledge" rule

**Hypothesis:** the ambiguous category's low scores (faithfulness
~0.45 across runs) were caused by the system prompt's strict "answer
ONLY from context, never use outside knowledge" rule (rule 1) giving
the model nothing to fall back on for under-specified questions.

**Change tested:** rewrote the system prompt to (a) allow general
medical background explanations when clearly separated from
patient-specific content, and (b) added a new rule: if the question
doesn't specify which patient/visit, answer based on what's in the
retrieved context and note that the question didn't specify.

**Result (single run, vs. Run A baseline above):**

| category  | faithfulness: before -> after | relevance: before -> after |
|-----------|-------------------------------|------------------------------|
| ambiguous | 0.467 -> 0.333 (worse)         | 0.790 -> 0.700 (worse)       |
| factoid   | 0.710 -> 0.820                 | 0.950 -> 0.970               |
| multi_hop | 0.700 -> 0.700                 | 0.867 -> 0.850               |
| negation  | 0.975 -> 0.900 (worse)         | 0.825 -> 0.875               |
| refusal   | 0.933 -> 0.767 (worse)         | 1.000 -> 1.000               |
| **overall** | **0.738 -> 0.706**           | **0.898 -> 0.891**           |

The targeted category (ambiguous) got *worse*, and the overall score
regressed. Given finding #1 above, part of this swing is within normal
run-to-run noise -- but the *mechanism* visible in the actual answer
text is informative regardless:

For q16 ("What happened during the visit?"), the new rule caused the
model to recognise that the top-3 retrieved chunks span **two unrelated
patients' records** (a Psychiatry/Psychology visit and a separate
Chiropractic visit), and produce a branching answer ("If you're
referring to the Psychiatry visit... If you're referring to the
Chiropractic visit..."). This is arguably *more* honest about what
retrieval actually returned -- but it roughly doubles answer length
with two competing narratives and awkward citation placement, which the
LLM judge appears to penalise as weak grounding.

**Conclusion:** the ambiguous category's weakness is not primarily a
system-prompt issue -- it's that, for patient-less queries, the top-3
retrieved chunks can span multiple unrelated patients with no principled
way to choose, and any answer (confident-but-narrow, or
honest-but-branching) scores poorly under the current judge. This is
consistent with the retrieval-side root cause already identified for
this category. The system prompt was reverted to its previous version;
no code change resulted from this experiment.
