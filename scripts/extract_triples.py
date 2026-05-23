"""Batched triple extraction for a benchmark, writing the result back
into the cached JSONL so subsequent runs skip the extractor entirely.

Usage:
    python scripts/extract_triples.py --dataset fever --split train --max-samples 5000
    python scripts/extract_triples.py --dataset halueval --subset qa --extractor qwen
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clauseguard.data import factscore, fever, halueval, medhall
from clauseguard.extraction.triple_extractor import RegexTripleExtractor
from clauseguard.utils.io import cache_path, read_jsonl, write_jsonl


log = logging.getLogger("extract_triples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract triples for a benchmark.")
    parser.add_argument("--dataset", choices=("fever", "halueval", "factscore", "medhall"), required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", default="qa")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--extractor", choices=("regex", "qwen"), default="regex")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.extractor == "regex":
        extractor = RegexTripleExtractor()
    else:
        from clauseguard.extraction.qwen_extractor import make_qwen_extractor

        _, extractor = make_qwen_extractor()

    if args.dataset == "fever":
        rows = fever.load_fever(split=args.split, max_samples=args.max_samples)
        p = cache_path("fever", f"{args.split}.jsonl")
    elif args.dataset == "halueval":
        rows = halueval.load_halueval(args.subset, max_samples=args.max_samples)
        p = cache_path("halueval", f"{args.subset}.jsonl")
    elif args.dataset == "factscore":
        rows = factscore.load_factscore(max_samples=args.max_samples)
        p = cache_path("factscore", "atomic.jsonl")
    elif args.dataset == "medhall":
        rows = medhall.load_medhall(max_samples=args.max_samples)
        p = cache_path("medhall", "all.jsonl")
    else:
        raise ValueError(args.dataset)

    log.info("Extracting triples for %d rows -> %s", len(rows), p)
    for i, row in enumerate(rows):
        row["claim_triples"] = [
            t.__dict__ for t in extractor.extract(row.get("claim", ""))
        ]
        row["evidence_triples"] = [
            t.__dict__
            for sent in (row.get("evidence_sentences") or [])
            for t in extractor.extract(sent)
        ]
        if (i + 1) % 500 == 0:
            log.info("  %d / %d", i + 1, len(rows))
    write_jsonl(p, rows)
    log.info("Wrote %d rows with triples to %s", len(rows), p)


if __name__ == "__main__":
    main()
