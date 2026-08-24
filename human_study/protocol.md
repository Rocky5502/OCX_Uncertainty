# Protocol: Uncertainty-Guided Repository Code Review

## Objective

Determine whether source-specific uncertainty and targeted intervention
guidance improve developers' ability to identify and repair repository-level
code-generation failures relative to generic review assistance.

## Design

The confirmatory human study uses a randomized, counterbalanced within-subject
design. Forty adults will be recruited, with at least 36 valid completions
targeted. Every participant completes twelve scored tasks and two unscored
practice tasks. Each scored task is encountered exactly once.

Up to 100 unique invitation codes are issued to accommodate nonresponse.
Recruitment closes to new entrants after 40 completed sessions or at the
declared deadline. Invitation count is not the participant sample size.

The three conditions are generic review, uncertainty display, and targeted
guidance. Starting code, retrieved evidence, task text, editing capability,
time limit, and private executable evaluation are identical across conditions.
Only the displayed risk trace and recommended action differ.

## Participants

Eligible participants are software engineers, AI researchers, or PhD students
with at least one year of Python experience and familiarity with code review,
testing, or repository-level development. Categories may overlap. Expertise is
modeled continuously using programming years, Python years, review frequency,
and AI-tool usage rather than treating role labels as disjoint groups.

Operationally, familiarity requires at least one of: code review at least
monthly, repository development at least monthly, or unit-testing familiarity
of at least 2 on the five-point scale.

## Stimuli

The twelve cases comprise eight initially incorrect outputs and four correct
controls. Incorrect cases are balanced across dominant API, repository-context,
similar-code, and generation risk signals. Correct controls measure false alarms
and unnecessary edits. Stimulus selection is deterministic and recorded before
human outcomes are observed.

This is a constructed case-control set. Raw correctness percentages estimate
performance on these cases only and must not be interpreted as benchmark-wide
failure prevalence.

## Session

1. Consent and eligibility: 5 minutes.
2. Background questionnaire: 5 minutes.
3. Tutorial and two unscored practice tasks: 10 minutes.
4. Twelve scored tasks, six-minute limit each: approximately 60 minutes.
5. Post-study questionnaire: 8 minutes.

Participants judge the starting implementation, optionally edit it, submit a
complete target function, report confidence from 0 to 100, and rate difficulty
and guidance usefulness. Official executable tests run only after submission.

## Outcomes

The primary outcome is final executable correctness. Secondary outcomes are
failure-detection accuracy, repair success, accepted-incorrect rate,
unnecessary-edit rate, completion time, edit size, confidence calibration,
perceived difficulty, workload, and guidance usefulness.

## AI-Agent Replication

Ten independent LLM review sessions may be run on the same frozen stimuli and
condition schedules. These records are exploratory, labeled `AGENT_EXPLORATORY`,
and analyzed separately. They are never included in the human sample size or
human inferential tests.
