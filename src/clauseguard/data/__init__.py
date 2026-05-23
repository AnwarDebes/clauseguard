"""Dataset loaders: FEVER, HaluEval, FActScore, MedHallBench.

Each loader returns a list of dicts with the same shape:

    [
        {
            "claim_id": str,
            "claim": str,
            "label": int,           # 0=Support, 1=Refute, 2=NotEnoughInfo
            "evidence_sentences": list[str],
            "evidence_triples": list[Triple] | None,   # populated by triple extractor
            "claim_triples": list[Triple] | None,
            "source": str,                              # benchmark name
        }, ...
    ]

The benchmark-specific peculiarities (FEVER's evidence-set parsing,
HaluEval's binary labels, FActScore's atomic-claim slicing) are
normalised here so all downstream code sees a single shape.
"""
from .fever import load_fever
from .halueval import load_halueval
from .factscore import load_factscore
from .medhall import load_medhall

__all__ = ["load_fever", "load_halueval", "load_factscore", "load_medhall"]
