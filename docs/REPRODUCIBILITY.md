# Reproducibility Guide

## Offline Reproduction

Create an isolated environment and install the released analysis and validation stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
make release-check
```

The root `requirements.txt` covers the public offline analyses, figures, integrity checks, and reviewer-study validation. Heavy embedding/model dependencies are intentionally optional; install them only when needed with:

```bash
python -m pip install -e ".[embeddings,datasets]"
```

`make release-check` verifies package tests, task and cell counts, public-record redactions, aggregate reviewer results, table regeneration, figure regeneration, file sizes, and secret/path scans without using an API.

## Full Technical Rerun

A full rerun additionally requires:

- official benchmark distributions and source repositories;
- reconstructed executable environments and public tests;
- the frozen repository-knowledge index;
- access to the exact model IDs or a declared non-equivalent replication;
- an API credential stored outside version control;
- provider budget and rate-limit approval.

Run the preparation and leakage audits before generation. Do not tune frozen thresholds on confirmatory task outcomes. Preserve failed and length-limited responses, retries, token use, latency, served-model metadata, and hashes.

## What The Public Artifact Reproduces

Committed source-free records reproduce the released aggregate tables, plots, intervals, and integrity counts associated with them. They do not reproduce model text generation or third-party test execution without the official benchmark inputs.

Any additional manuscript-referenced post-hoc analysis must be released together with its runnable analysis script, minimal source-free frozen inputs (or an existing public input path), generated summaries, seeds, bootstrap/masking settings, figures/tables, provenance metadata, and integrity checks. Such analyses must make no new model calls and must not expose generated source code, private benchmark material, provider identifiers, credentials, or participant-level data.
