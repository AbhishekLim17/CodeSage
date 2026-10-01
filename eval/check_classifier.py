r"""Measure the overview classifier on questions you wrote: how many whole-repository questions it recognises, and how
many narrow ones it wrongly takes for overviews.

    python eval/check_classifier.py                              # the shipped held-out set
    python eval/check_classifier.py my_questions.json            # your own

The file is JSON with two lists of strings, ``"overview"`` (questions about the whole repository) and ``"specific"``
(questions about one piece of it, including ones that merely sound like overviews). The most useful thing you can do for
this classifier is to write twenty of each in the words *you* would use, before reading its patterns: the shipped set was
written by the author of the patterns, so it can only guard against regressions, not show how well they generalise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from codebase_ai.retrieval.query import overview_intent

DEFAULT = Path(__file__).parent / "questions" / "classifier_heldout.json"


def measure(cases: dict) -> dict:
    overview = list(cases.get("overview", []))
    specific = list(cases.get("specific", []))
    missed = [q for q in overview if overview_intent(q) is None]
    flagged = [(q, overview_intent(q)) for q in specific if overview_intent(q) is not None]
    return {
        "overview": len(overview),
        "recognised": len(overview) - len(missed),
        "missed": missed,
        "specific": len(specific),
        "flagged": flagged,
    }


def report(result: dict) -> str:
    lines = []
    if result["overview"]:
        lines.append(f"Recognised {result['recognised']} of {result['overview']} overview questions ({result['recognised'] / result['overview']:.0%}).")
    if result["specific"]:
        lines.append(f"Wrongly flagged {len(result['flagged'])} of {result['specific']} specific questions.")
    if result["missed"]:
        lines += ["", "Missed overview questions:", *[f"  - {q}" for q in result["missed"]]]
    if result["flagged"]:
        lines += ["", "Specific questions taken for overviews:", *[f"  - {q}  (as {intent})" for q, intent in result["flagged"]]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    try:
        cases = json.loads(args.file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(cases, dict) or not all(isinstance(v, list) for k, v in cases.items() if not k.startswith("_")):
        print('error: expected a JSON object with "overview" and "specific" lists', file=sys.stderr)
        return 2
    print(report(measure(cases)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
