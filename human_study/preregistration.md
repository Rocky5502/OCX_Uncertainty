# Preregistration

Status: draft for institutional review. Freeze this document before the first
participant is recruited.

## Confirmatory Question

Does targeted source-specific uncertainty guidance increase post-review
executable correctness relative to generic review?

## Hypotheses

- H1: targeted guidance has higher final executable correctness than generic
  review.
- H2: uncertainty display has higher failure-detection accuracy than generic
  review.
- H3: targeted guidance has higher repair success than uncertainty display.
- H4: targeted guidance reduces accepted-incorrect outputs and unnecessary
  edits without increasing median completion time.

H1 is primary. H2--H4 are secondary. Source and expertise interactions are
exploratory unless a power analysis completed before recruitment supports them.

## Sampling and Stopping Rule

Recruitment stops after 40 completed sessions or at the declared recruitment
deadline, whichever occurs first. The target analyzable sample is at least 36.
No interim outcome analysis will determine stopping.
Up to 100 unique invitations may be distributed to achieve this completion
target; nonresponding invitees are not participants and are not analyzed.

## Exclusions

Exclude a participant only for: missing consent; ineligibility; fewer than nine
completed scored tasks; unrecoverable technical problems on more than three
tasks; failure to complete the tutorial; or interaction shorter than 30 seconds
on at least four tasks. Poor task performance is not an exclusion criterion.
Episode-level evaluator infrastructure errors are missing, not incorrect.
Eligibility requires at least one year of Python experience and at least one of
monthly-or-more code review, monthly-or-more repository development, or unit
testing familiarity of at least 2/5.

## Primary Estimand

The primary estimand is the within-participant marginal percentage-point
difference in executable correctness between targeted guidance and generic
review on the constructed stimulus set.

## Analysis

Report condition numerators, denominators, percentages, participant-clustered
bootstrap 95% confidence intervals, and paired condition contrasts. The primary
model is an episode-level logistic GEE with participant clusters, task fixed
effects, condition, period, and continuous expertise. A participant-level paired
bootstrap is the robustness analysis. Apply Holm correction to the three
pairwise condition comparisons.

The continuous expertise score is frozen as the unweighted mean of six
normalized components: min(programming years/10, 1), min(Python years/8, 1),
code-review frequency/5, repository-development frequency/5, testing
familiarity/5, and dependency-tracing familiarity/5. Roles may overlap and are
reported descriptively; they are not used to create mutually exclusive
expertise groups.

Completion time is capped at the task limit and analyzed on the log scale, with
timeout frequency reported separately. Confidence calibration uses Brier score,
five-bin calibration plots, and descriptive ten-bin ECE. Source and expertise
interactions are labeled secondary.

## Missing Data

Do not impute correctness. Episodes with evaluator infrastructure errors remain
missing and are reported. Questionnaire missingness is reported by item.

## Claims

Use causal language only for randomized condition effects within this stimulus
set. Do not generalize the constructed case mix to benchmark prevalence, all
developers, or all LLM-generated code.
