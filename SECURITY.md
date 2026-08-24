# Security Policy

## Supported Version

Security fixes are applied to the current `main` branch.

## Reporting

Do not open a public issue containing a credential, private benchmark payload,
participant record, provider response identifier, or exploitable generated
code. Contact the repository owner privately through GitHub instead.

## Credential Handling

- Store API credentials in a local `.env` file or secret manager.
- Never place credentials in YAML, JSON, manifests, prompts, logs, or results.
- Rotate a key immediately if it is exposed in a terminal capture or commit.
- Treat provider response identifiers and private test output as sensitive.

## Executing Generated Code

Generated code is untrusted. Run it in an isolated environment with constrained
filesystem, network, process, memory, and time access. Review dependencies and
licenses before execution or deployment.
