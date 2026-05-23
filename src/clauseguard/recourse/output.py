"""Render a human-readable recourse report.

For a Refuted (or NEI) verification with applied edits, produce a
short markdown block summarising:

* the verifier's original label and class scores,
* each applied edit, with rationale,
* the final label and class scores,
* whether the flip succeeded.

The report is what the EU AI Act Article 14 reviewer reads. We keep
it short (target: under 200 words per case) so a human can triage many
in a single sitting.
"""
from __future__ import annotations

from typing import Any

from .search import SearchTrace


def render_recourse_report(
    *,
    claim_text: str,
    original_label_name: str,
    original_scores: list[float],
    final_label_name: str,
    final_scores: list[float],
    trace: SearchTrace,
    case_id: str = "",
) -> str:
    lines: list[str] = []
    header = f"# Recourse report{(' for ' + case_id) if case_id else ''}\n"
    lines.append(header)
    lines.append(f"**Claim**: {claim_text}\n")
    lines.append(
        f"**Original verdict**: {original_label_name} "
        f"(scores: {', '.join(f'{s:+.1f}' for s in original_scores)})\n"
    )
    if not trace.applied_edits:
        lines.append("No evidence edits could be found to flip the verdict.\n")
        return "\n".join(lines)
    lines.append(f"**Applied {len(trace.applied_edits)} edit(s)**:\n")
    for i, edit in enumerate(trace.applied_edits, 1):
        new_triple = edit.new_triple.render() if edit.new_triple else "(removed)"
        lines.append(
            f"{i}. **{edit.op}** at evidence-triple #{edit.target_idx}: "
            f"`{new_triple}`\n"
            f"   *{edit.rationale}*\n"
        )
    lines.append(
        f"\n**Final verdict**: {final_label_name} "
        f"(scores: {', '.join(f'{s:+.1f}' for s in final_scores)})\n"
    )
    lines.append(
        f"**Flipped?** {'yes' if trace.flipped else 'no'}; "
        f"latency {trace.total_latency_ms:.0f} ms; "
        f"candidates tried {trace.candidates_tried}.\n"
    )
    return "\n".join(lines)
