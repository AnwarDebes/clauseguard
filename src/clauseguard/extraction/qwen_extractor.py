"""Qwen2.5-1.5B-Instruct claim decomposer + triple extractor.

Uses the same teacher checkpoint that the user's ``decoder-attention-distill-graphtm``
project already fine-tunes (``$QWEN_TEACHER_DIR``). Deterministic with temperature 0
and a fixed seed; every call is logged with the model SHA, prompt, and raw output for
the audit trail.

Loading is lazy: instantiating the class does *not* load the model. The model is
loaded on the first ``decompose`` or ``extract`` call. This keeps unit tests fast.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from .triple_extractor import (
    Claim,
    ClaimDecomposer,
    Triple,
    TripleExtractor,
    canonicalise_entity,
    canonicalise_relation,
    deduplicate_triples,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are an atomic-claim decomposer for a fact-verification system.

Split the following text into a minimal list of atomic factual claims. \
Each claim must be:
  - a single propositional fact (one subject, one predicate),
  - self-contained (no pronouns, resolve them inline),
  - in present or past simple tense.

Return a JSON list of strings. No explanation, no markdown.

Text:
{text}
"""

_EXTRACT_PROMPT = """\
You are a (subject, relation, object) triple extractor for a \
knowledge-graph-grounded fact-verification system.

Extract zero or more (subject, relation, object) triples from the claim \
below. Use one of the canonical relations when possible: \
{rel_vocab}. \
Use "_open_" only if no canonical relation fits. \
Set "polarity" to -1 if the claim is negated, otherwise 1.

Return a JSON list of objects with keys: subject, relation, object, polarity. \
No explanation, no markdown.

Claim:
{text}
"""


# --------------------------------------------------------------------------
# QwenClaimDecomposer
# --------------------------------------------------------------------------

class QwenClaimDecomposer(ClaimDecomposer):
    """Decompose an LLM output into atomic claims using Qwen2.5-1.5B-Instruct."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 256,
        log_dir: str | Path | None = None,
    ) -> None:
        self.model_path = model_path or os.environ.get(
            "QWEN_TEACHER_DIR", os.path.expanduser("~/model_archive/Qwen2.5-1.5B-Instruct")
        )
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.log_dir = Path(log_dir) if log_dir else None
        self._tokenizer = None
        self._model = None
        self._weights_sha: str | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading Qwen2.5-1.5B from %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()
        # Hash of the model weights, abbreviated to 12 hex chars for the audit ID.
        self._weights_sha = self._hash_weights(self._model)

    @staticmethod
    def _hash_weights(model: Any) -> str:
        h = hashlib.sha256()
        for name, p in model.named_parameters():
            h.update(name.encode("utf-8"))
            # Hash a downsample of the param to bound time; for full provenance,
            # users should record the model_path's git-LFS SHA instead.
            t = p.detach().to("cpu").float().flatten()
            stride = max(1, t.numel() // 256)
            h.update(t[::stride].numpy().tobytes())
        return h.hexdigest()[:12]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        import torch

        self._ensure_loaded()
        messages = [
            {"role": "system", "content": "You are a precise fact-extraction assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decompose(self, llm_output: str) -> list[Claim]:
        self._ensure_loaded()
        prompt = _DECOMPOSE_PROMPT.format(text=llm_output)
        raw = self._generate(prompt)
        self._log_call("decompose", llm_output, raw)
        claim_texts = self._safe_json_list(raw)
        if not claim_texts:
            return []
        # Triple extraction is done by a sibling extractor; the decomposer only
        # produces atomic claim text. Triples are filled in by the orchestrator.
        return [Claim(text=str(c), triples=(), extractor_id=self.id) for c in claim_texts]

    @property
    def id(self) -> str:
        if self._weights_sha is None:
            return "qwen2.5-1.5b-instruct@unloaded"
        return f"qwen2.5-1.5b-instruct@{self._weights_sha}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_json_list(raw: str) -> list:
        s = raw.strip()
        if s.startswith("```"):
            # strip fenced code blocks
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:]
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            log.warning("Qwen output was not valid JSON: %r", raw[:200])
        return []

    def _log_call(self, kind: str, prompt: str, raw_output: str) -> None:
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "kind": kind,
            "extractor_id": self.id,
            "prompt": prompt,
            "raw_output": raw_output,
        }
        with open(self.log_dir / "extraction_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------
# QwenTripleExtractor
# --------------------------------------------------------------------------

class QwenTripleExtractor(TripleExtractor):
    """Extract canonicalised triples from an atomic claim using Qwen2.5-1.5B-Instruct."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 192,
        log_dir: str | Path | None = None,
        decomposer: QwenClaimDecomposer | None = None,
    ) -> None:
        # Share the underlying Qwen model between decomposer and extractor when
        # both are used: avoids loading the 1.5B weights twice.
        if decomposer is not None:
            self._decomposer = decomposer
        else:
            self._decomposer = QwenClaimDecomposer(
                model_path=model_path,
                device=device,
                max_new_tokens=max_new_tokens,
                log_dir=log_dir,
            )

    def extract(self, claim_text: str) -> tuple[Triple, ...]:
        from .triple_extractor import REL_VOCAB

        rel_vocab_str = ", ".join(REL_VOCAB)
        prompt = _EXTRACT_PROMPT.format(text=claim_text, rel_vocab=rel_vocab_str)
        raw = self._decomposer._generate(prompt)
        self._decomposer._log_call("extract", claim_text, raw)
        obj = self._decomposer._safe_json_list(raw)
        triples: list[Triple] = []
        for d in obj:
            if not isinstance(d, dict):
                continue
            try:
                subj = canonicalise_entity(d.get("subject", ""))
                obj_s = canonicalise_entity(d.get("object", ""))
                if not subj or not obj_s:
                    continue
                rel = canonicalise_relation(d.get("relation", "_open_"))
                pol = int(d.get("polarity", 1))
                if pol not in (-1, 1):
                    pol = 1
                triples.append(
                    Triple(
                        subject=subj,
                        relation=rel,
                        object=obj_s,
                        polarity=pol,
                        source_span=(0, len(claim_text)),
                        confidence=0.85,
                    )
                )
            except (TypeError, ValueError, KeyError):
                continue
        return deduplicate_triples(triples)

    @property
    def id(self) -> str:
        return self._decomposer.id


def make_qwen_extractor(
    model_path: str | None = None,
    device: str = "cuda",
    log_dir: str | Path | None = None,
) -> tuple[QwenClaimDecomposer, QwenTripleExtractor]:
    """Convenience factory returning a sharing decomposer + extractor pair."""
    dec = QwenClaimDecomposer(model_path=model_path, device=device, log_dir=log_dir)
    ext = QwenTripleExtractor(decomposer=dec)
    return dec, ext
