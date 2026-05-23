"""Aggregate per-seed JSON summaries into one paper-ready table.

Walks ``results/`` and groups by experiment, computing mean / std /
median / bootstrap 95% CI across the 5 canonical seeds.

Outputs:
    results/clauseguard_5seeds.json     master aggregated dict
    results/clauseguard_5seeds.md       markdown table
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clauseguard.eval.metrics import bootstrap_ci


log = logging.getLogger("aggregate_results")


SEEDS = (42, 123, 456, 789, 1337)
EXPERIMENTS = (
    "train_fever",
    "eval_halueval_qa",
    "eval_halueval_dialogue",
    "eval_halueval_summarization",
    "eval_factscore",
    "eval_medhall",
    "recourse_eval",
    "adversarial_eval",
    "sat_receipts",
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = Path(__file__).resolve().parents[1] / "results"
    summaries: dict[str, dict[int, dict]] = defaultdict(dict)
    for exp in EXPERIMENTS:
        for seed in SEEDS:
            p = root / exp / f"seed_{seed}" / "summary.json"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    summaries[exp][seed] = json.load(f)

    aggregated: dict[str, dict] = {}
    for exp, by_seed in summaries.items():
        if not by_seed:
            continue
        all_keys = sorted({k for d in by_seed.values() for k, v in d.items() if isinstance(v, (int, float))})
        per_metric: dict[str, dict] = {}
        for k in all_keys:
            vals = [by_seed[s][k] for s in by_seed if k in by_seed[s] and isinstance(by_seed[s][k], (int, float))]
            if not vals:
                continue
            mean, lo, hi = bootstrap_ci(vals)
            per_metric[k] = {
                "n_seeds": len(vals),
                "mean": float(mean),
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
                "values": vals,
            }
        aggregated[exp] = per_metric
    out = root / "clauseguard_5seeds.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    log.info("Wrote aggregated -> %s", out)

    md_lines = ["# ClauseGuard aggregated 5-seed results\n"]
    for exp, metrics in aggregated.items():
        md_lines.append(f"## {exp}\n")
        md_lines.append("| Metric | n | Mean | 95% CI |")
        md_lines.append("|---|---|---|---|")
        for k, v in metrics.items():
            md_lines.append(
                f"| {k} | {v['n_seeds']} | {v['mean']:.4f} | "
                f"[{v['ci95_lo']:.4f}, {v['ci95_hi']:.4f}] |"
            )
        md_lines.append("")
    md_out = root / "clauseguard_5seeds.md"
    md_out.write_text("\n".join(md_lines))
    log.info("Wrote markdown -> %s", md_out)
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
