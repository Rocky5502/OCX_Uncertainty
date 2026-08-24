"""OpenCoder command-line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from .data.loaders import load_dataset
from .pipeline import Pipeline, PipelineConfig


def _backend_from_env() -> str:
    return os.environ.get("OPENCODER_LLM_BACKEND", "")


def cmd_run(args):
    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    cfg.llm_backend = _backend_from_env() or cfg.llm_backend
    cfg.llm_model = os.environ.get("OPENCODER_LLM_MODEL", cfg.llm_model)
    pipe = Pipeline(cfg)

    examples = list(load_dataset(args.dataset, args.dataset_path, limit=args.limit))
    if not examples:
        print("No examples loaded.", file=sys.stderr)
        sys.exit(1)

    results = []
    for ex in examples:
        _, retrievers = pipe.index_example(
            ex,
            fallback_repo_root=args.repo_root,
            describe_limit=args.describe_limit,
        )
        r = pipe.run(ex, retrievers)
        results.append(asdict(r))
        print(f"[{r.example_id}] tests_passed={r.test_report.get('passed')} "
              f"u_agg={r.uncertainty_trace['aggregate']:.3f} "
              f"pass@1={r.pass_at_k.get('pass@1', 'NA')} repairs={r.repair_rounds}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote {args.out}")


def cmd_eval(args):
    args.limit = args.limit
    cmd_run(args)


def main(argv=None):
    p = argparse.ArgumentParser("opencoder")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (("run", cmd_run), ("eval", cmd_eval)):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=None)
        sp.add_argument("--dataset", default="execrepobench")
        sp.add_argument("--dataset-path", default=None)
        sp.add_argument("--repo-root", default=None,
                        help="Repo to index when examples don't specify one.")
        sp.add_argument("--describe-limit", type=int, default=200,
                        help="Cap NL-description calls for large repos.")
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--out", default=None)
        sp.set_defaults(fn=fn)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
