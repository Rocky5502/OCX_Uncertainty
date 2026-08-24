# OpenCoderX Human Developer Study

This directory releases the protocol, instruments, schemas, local interface,
and analysis code for uncertainty-guided repository code review. It contains no
participant-level responses or identifying data.

## Design

The study uses a randomized, counterbalanced within-subject design:

1. `generic_review`: task, fixed evidence, and starting implementation.
2. `uncertainty_display`: generic review plus aggregate and source risks.
3. `targeted_guidance`: uncertainty display plus one source-specific action.

Each participant reviews twelve scored tasks once, four per condition, after
two practice tasks. Three assignment groups rotate every task through every
condition. The preregistered recruitment target was 40 completions; the
documented cohort comprised 24 uncompensated technical volunteers, yielding a
12-task protocol scope of 288 episodes.

## Evidence Boundary

- Human and automated-reviewer populations are never pooled.
- Starting code, evidence, and task text are fixed across conditions.
- Tests and reference implementations remain unavailable during review.
- Synthetic records are software fixtures and are never participant evidence.
- The public release contains neither response-level records nor an
  institutional ethics determination; it therefore reports no human-condition
  effect estimate.

## Local Interface

```bash
PYTHONPATH=. .venv/bin/python human_study/prepare_study.py
PYTHONPATH=. .venv/bin/python human_study/validate_study.py
PYTHONPATH=. .venv/bin/python human_study/serve_study.py --port 8765
```

The server is intended for a trusted institutional network or a secured
deployment. Empirical mode requires a completed ethics-determination file and
new invitation codes. Do not expose the development server directly to the
public Internet.

After an independently authorized collection, scoring and analysis use:

```bash
PYTHONPATH=. .venv/bin/python human_study/score_responses.py \
  --responses human_study/data/episodes.jsonl
PYTHONPATH=. .venv/bin/python human_study/analyze_study.py \
  --participants human_study/data/participants.jsonl \
  --tutorials human_study/data/tutorials.jsonl \
  --withdrawals human_study/data/withdrawals.jsonl \
  --poststudy human_study/data/poststudy.jsonl \
  --ethics-approval-file /secure/path/ethics_approval.json \
  --episodes results/human_study/scored_human_episodes.csv
```

Invitation codes, private stimuli, and collected records are ignored by Git.
