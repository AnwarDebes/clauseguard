"""Triple extraction: LLM output -> atomic claims -> (s, r, o) triples.

Public types live in :mod:`clauseguard.extraction.triple_extractor`. The
default extractor uses Qwen2.5-1.5B-Instruct via HuggingFace transformers
(deterministic with temperature 0). A lightweight regex-based extractor
is provided for tests and CI runs without GPU.
"""
from .triple_extractor import (
    REL_VOCAB,
    Claim,
    ClaimDecomposer,
    Triple,
    TripleExtractor,
    canonicalise_entity,
    canonicalise_relation,
)

__all__ = [
    "Triple",
    "Claim",
    "ClaimDecomposer",
    "TripleExtractor",
    "REL_VOCAB",
    "canonicalise_entity",
    "canonicalise_relation",
]
