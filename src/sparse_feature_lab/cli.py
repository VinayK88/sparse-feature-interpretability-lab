from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_experiment
from .visualize import write_recovery_svg


def main() -> int:
    parser = argparse.ArgumentParser(description="Train sparse autoencoders and measure feature recovery")
    parser.add_argument("--samples", type=int, default=2400)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=Path("reports/baseline.json"))
    parser.add_argument("--figure", type=Path, default=Path("reports/feature_recovery.svg"))
    args = parser.parse_args()

    result = run_experiment(samples=args.samples, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_recovery_svg(result, args.figure)

    summary = result["summary"]
    print(f"mean best cosine:              {summary['mean_best_cosine']:.3f}")
    print(f"seed stability (std):          {summary['seed_std_best_cosine']:.3f}")
    print(f"features recovered @ 0.80:     {summary['mean_recovered_at_0_80']:.1%}")
    print(f"median intervention specificity: {summary['median_intervention_specificity']:.2f}x")
    print(f"wrote {args.output} and {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
