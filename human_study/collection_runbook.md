# Human Study Collection Runbook

## Before Recruitment

1. Obtain the applicable institutional ethics approval or exemption.
2. Complete every field in `ethics_approval.template.json`, store the approved
   copy outside the public repository, and keep the signed determination.
3. Freeze `protocol.md`, `preregistration.md`, consent, compensation, the
   recruitment deadline, and the analysis commit before the first participant.
4. Run `make human-study-prepare human-study-validate` and archive
   `human_study/frozen/integrity.json` with the study records.
5. Assign participant codes independently of role or expected performance.

## Collection

Start the server only on a secured institutional host:

```bash
PYTHONPATH=. .venv/bin/python human_study/serve_study.py \
  --mode empirical \
  --ethics-approval-file /secure/path/ethics_approval.json \
  --data-dir /secure/path/opencoderx_human_data \
  --host 127.0.0.1 --port 8765
```

Do not expose the development server directly to the public Internet. Back up
the append-only JSONL files daily. Keep identity/contact records separate from
participant codes and never commit them.

Recruitment stops at 40 completed sessions or the preregistered deadline. Do
not inspect condition outcomes while recruitment is active.

## Closeout and Analysis

1. Copy the de-identified JSONL records to the approved analysis machine.
2. Preserve the raw files read-only and record SHA-256 hashes.
3. Score submitted functions with the frozen executable evaluator.
4. Run the preregistered analysis with tutorial and withdrawal records.
5. Audit participant flow, evaluator errors, numerators, denominators, and the
   `paper_eligible` field before using any generated LaTeX.

```bash
PYTHONPATH=. .venv/bin/python human_study/score_responses.py \
  --responses /secure/path/opencoderx_human_data/episodes.jsonl

PYTHONPATH=. .venv/bin/python human_study/analyze_study.py \
  --participants /secure/path/opencoderx_human_data/participants.jsonl \
  --tutorials /secure/path/opencoderx_human_data/tutorials.jsonl \
  --withdrawals /secure/path/opencoderx_human_data/withdrawals.jsonl \
  --poststudy /secure/path/opencoderx_human_data/poststudy.jsonl \
  --ethics-approval-file /secure/path/ethics_approval.json \
  --episodes results/human_study/scored_human_episodes.csv
```

Synthetic dry-run artifacts are software fixtures only. They are never human
observations and must not enter a paper, abstract, response letter, or claim.
