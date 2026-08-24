# Third-Party Notices

OpenCoderX source code is MIT-licensed. The artifact also interoperates with
benchmarks, model APIs, and Python packages governed by their own terms.

## Benchmarks

- **ExecRepoBench**: obtain the dataset from its official distribution. Each
  underlying repository remains subject to its own license.
- **CrossCodeEval**: official code and data tooling are Apache-2.0 licensed;
  underlying repositories retain their original licenses.
- **RepoExec**: the official benchmark repository is MIT licensed; source
  projects and packaged test applications retain their own notices.
- **CoderEval**: obtain the dataset and execution environment from the official
  project and comply with the licenses of its source repositories.

The released `data/manifests/` files contain identifiers, metadata, selection
rules, and hashes only. They do not redistribute repository source, reference
implementations, private tests, or upstream archives.

## Prior Implementations

The release incorporates and extends the authors' earlier MIT-licensed
OpenCoder V1 implementation. The `opencoder/baselines/alliancecoder.py` module
is a clean-room adapter based on the published AllianceCoder method
description. It does not import or redistribute the AllianceCoder repository,
which did not expose a license during the release audit.

## Models And Providers

Model names are used for scientific identification. No model weights are
redistributed. Users are responsible for provider terms, access permissions,
pricing, rate limits, data handling, and model-specific acceptable-use rules.

## Python Dependencies

Runtime and optional dependencies retain their upstream licenses. Consult the
installed package metadata before redistribution.
