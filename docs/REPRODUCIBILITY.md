# Reproducibility Guide

## Offline Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,analysis]"
make release-check
```

This verifies package tests, task and cell counts, public-record redactions,
aggregate reviewer results, table regeneration, figure regeneration, file
sizes, and secret/path scans without using an API.

## Full Technical Rerun

A full rerun additionally requires:

- official benchmark distributions and source repositories;
- reconstructed executable environments and public tests;
- the frozen repository-knowledge index;
- access to the exact model IDs or a declared non-equivalent replication;
- an API credential stored outside version control;
- provider budget and rate-limit approval.

Run the preparation and leakage audits before generation. Do not tune frozen
thresholds on confirmatory task outcomes. Preserve failed and length-limited
responses, retries, token use, latency, served-model metadata, and hashes.

## What The Public Artifact Reproduces

The source-free records reproduce all released aggregate tables, plots,
intervals, and integrity counts. They do not reproduce model text generation or
third-party test execution without the official benchmark inputs.
