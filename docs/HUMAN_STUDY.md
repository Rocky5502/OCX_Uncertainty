# Human Study Protocol

The human-centered component uses a randomized, counterbalanced within-subject
review design with three conditions: generic review, aggregate uncertainty
display, and source-specific targeted guidance. Each participant receives two
practice tasks and twelve scored repository review tasks.

## Released Materials

- protocol and preregistration
- consent, recruitment, and questionnaires
- JSON schemas and append-only collection server
- preparation, validation, scoring, and analysis code
- ethics-determination template

## Cohort Boundary

The documented cohort comprised 24 uncompensated technical volunteers and a
12-task workload, corresponding to 288 task episodes. Participant-level
responses and identifying data are not released. Because the artifact does not
contain response-level records or an institutional ethics determination, it
does not estimate or claim condition effects for human participants.

## Running The Interface

The release does not redistribute source-bearing task stimuli. Reconstruct the
official benchmark inputs as described in `docs/DATASETS.md`, then generate a
fresh local study package before starting the interface:

```bash
python human_study/prepare_study.py
python human_study/validate_study.py
python human_study/serve_study.py --port 8765
```

Empirical mode fails closed unless a completed approval/determination file is
supplied. Researchers reusing the protocol must obtain the determination
required by their institution, recruit independently, generate new invitation
codes, protect participant data, and define retention and withdrawal handling.

Synthetic dry runs may validate software only. They must never be pooled with
human observations or presented as participant evidence.
