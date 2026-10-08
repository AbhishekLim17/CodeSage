r"""Export answers for human grading, blinded (research plan, step 3.6; research/ANNOTATION_GUIDELINES.md, part B).

    python eval/export_annotation.py RESULTS.json [RESULTS.json ...] --out grading/HG-A
    python eval/export_annotation.py results/*/P1-frozen/*.json --sample 50 --out grading/HG-B   # 50 questions, every model
    python eval/export_annotation.py results/*/P3-frozen/*7b.json --ids hg_c_ids.txt --out grading/HG-C

Writes, in ``--out``:

* ``main/``: every answer, for the grader who grades the whole set;
* ``second/``: a random ``--second-share`` of them (default 30%), for the second grader, so agreement can be measured;
* in each: ``packet.md`` (to read: question, key facts, the answer, the code it cites), ``claims.csv`` (one row per
  sentence of the answer as a first split, to label for support) and ``answers.csv`` (one row per answer: correctness,
  flag, minutes);
* ``key.json``: which model, prompt and run each answer code stands for. **Keep it away from the graders.**

Graders see an opaque code per answer, never the model, the prompt or any judge verdict, and each batch is in its own
random order. The same answer has the same code in both batches. ``--seed`` makes the codes, the order and the second
grader's share reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from codebase_ai.evaluation import load_questions
from codebase_ai.rag.context import code_fence

SUPPORT = ("supported", "unsupported", "contradicted", "not a claim")
CORRECTNESS = ("correct", "partly correct", "wrong")
UNANSWERABLE = ("abstained", "hallucinated", "partial")

_FENCE = re.compile(r"^\s*(```|~~~)")
_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)]|#{1,6})\s+")  # list bullets, numbered items, headings
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LEADING_CITATIONS = re.compile(r"((?:\[(?:\d+|\?)\][.,;]?\s*)+)(.*)", re.DOTALL)
_CITATION = re.compile(r"\[(\d+|\?)\]")
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "cf.")


def split_claims(answer: str) -> list[str]:
    """A first split of an answer into claims: one per sentence of prose, outside code blocks, without list markers.

    Only a starting point (guidelines B1): the grader splits a row further or marks it "not a claim".
    """
    claims: list[str] = []
    in_code = False
    for line in answer.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
            continue
        text = _MARKER.sub("", line).strip()
        if in_code or not text:
            continue
        for position, part in enumerate(_SENTENCE_END.split(text)):
            if position == 0:
                claims.append(part)
                continue
            lead = _LEADING_CITATIONS.fullmatch(part)
            if lead:  # "It sends it. [2] Then..." : the citation belongs to the sentence before
                claims[-1] = f"{claims[-1]} {lead.group(1).strip()}"
                part = lead.group(2).strip()
            if not part:
                continue
            if claims[-1].lower().endswith(_ABBREVIATIONS):  # "e.g. this" is not a sentence of its own
                claims[-1] = f"{claims[-1]} {part}"
            else:
                claims.append(part)
    return claims


def cell(text: str) -> str:
    """Spreadsheet apps run a cell that starts with = + - @ as a formula; a leading apostrophe keeps it text."""
    return f"'{text}" if text[:1] in ("=", "+", "-", "@") else text


def collect(result_files: list[Path]) -> list[dict]:
    """Every answered question in the result files, with what grading needs and what the key records."""
    items = []
    questions_by_file: dict[str, dict] = {}
    for path in result_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        env = payload["env"]
        qfile = env["questions_file"]
        if qfile not in questions_by_file:
            questions_by_file[qfile] = {q.id: q for q in load_questions(qfile)}
        for row in payload["results"]:
            if row.get("error"):
                continue
            question = questions_by_file[qfile][row["id"]]
            items.append({
                "question_id": row["id"],
                "question": row["question"],
                "answerable": question.answerable,
                "key_facts": list(question.key_facts),
                "answer": row["answer"],
                "cited_code": [list(c) for c in row.get("cited_code", [])],
                "source": {
                    "results_file": str(path.resolve()),  # absolute, as judge_claims.py records it
                    "model": env.get("model"),
                    "prompt": (env.get("prompt") or {}).get("name", "P1"),
                    "retrieval": env.get("retrieval"),
                    "questions_file": qfile,
                },
            })
    return items


def packet(name: str, batch: list[dict]) -> str:
    instructions = (
        "Grade with the annotation guidelines, part B. In `claims.csv`, give each row a **support** label: "
        f"{' / '.join(SUPPORT)}. The rows are a first split, one per sentence: to split one further, add rows with the "
        "same answer code and the claim number plus a letter (3a, 3b). In `answers.csv`, give each answer a "
        f"**correctness** grade: {' / '.join(CORRECTNESS)} (for a question marked unanswerable: "
        f"{' / '.join(UNANSWERABLE)}), write `flagged` in **flag** if something is odd, and the **minutes** it took. "
        "Do not edit the answer text."
    )
    lines = [f"# Grading batch: {name} ({len(batch)} answers)", "", instructions, ""]
    for number, item in enumerate(batch, start=1):
        lines += ["---", "", f"## {number}. Answer {item['code']}", "", f"**Question:** {item['question']}", ""]
        if not item["answerable"]:
            lines += ["**Unanswerable:** the question writer found no answer in the repository.", ""]
        elif item["key_facts"]:
            lines += ["**Key facts:**", "", *(f"- {fact}" for fact in item["key_facts"]), ""]
        else:
            lines += ["**Key facts:** none recorded.", ""]
        lines += ["**Answer:**", "", *(f"> {line}" for line in item["answer"].splitlines()), ""]
        if not item["cited_code"]:
            lines += ["**Cited code:** none (the answer cites nothing).", ""]
        for cited, location, code in sorted(item["cited_code"]):
            fence = code_fence(code)
            lines += [f"**[{cited}] {location}**", "", fence, code, fence, ""]
    return "\n".join(lines)


def write_batch(folder: Path, name: str, batch: list[dict]) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "packet.md").write_text(packet(name, batch), encoding="utf-8")
    rows = 0
    # utf-8-sig: Excel reads a CSV as UTF-8 only when it starts with a byte-order mark.
    with (folder / "claims.csv").open("w", encoding="utf-8-sig", newline="") as f:
        out = csv.writer(f)
        out.writerow(["answer", "claim", "cites", "text", "support", "note"])
        for item in batch:
            for number, claim in enumerate(split_claims(item["answer"]), start=1):
                cites = " ".join(dict.fromkeys(_CITATION.findall(claim)))  # each number once, in order
                out.writerow([item["code"], number, cites, cell(claim), "", ""])
                rows += 1
    with (folder / "answers.csv").open("w", encoding="utf-8-sig", newline="") as f:
        out = csv.writer(f)
        out.writerow(["answer", "correctness", "flag", "note", "minutes"])
        out.writerows([item["code"], "", "", "", ""] for item in batch)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path, help="result files written by run_answers.py / run_matrix.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", type=Path, default=None, help="only these question ids (one per line)")
    parser.add_argument("--sample", type=int, default=None, help="a random N questions (all their answers are kept)")
    parser.add_argument("--second-share", type=float, default=0.3, help="share also given to the second grader")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        print(f"error: {args.out} is not empty; a batch is written once, so that grades always match their key", file=sys.stderr)
        return 2
    items = collect(args.results)
    files_of: dict[str, set[str]] = {}
    for item in items:
        files_of.setdefault(item["question_id"], set()).add(item["source"]["questions_file"])
    clashing = sorted(qid for qid, files in files_of.items() if len(files) > 1)
    if clashing:
        print(f"error: question ids repeat across question files ({', '.join(clashing[:5])}); make them unique", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    if args.ids is not None:
        wanted = {line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        items = [item for item in items if item["question_id"] in wanted]
    if args.sample is not None:
        pool = sorted({item["question_id"] for item in items})
        chosen = set(rng.sample(pool, min(args.sample, len(pool))))
        items = [item for item in items if item["question_id"] in chosen]
    if not items:
        print("error: no answers to export", file=sys.stderr)
        return 2

    codes = rng.sample(range(16**5), len(items))
    for item, code in zip(items, codes, strict=True):
        item["code"] = f"A{code:05X}"  # a letter first, so a spreadsheet never reads it as a number
    main_batch = rng.sample(items, len(items))
    second = rng.sample(items, round(len(items) * args.second_share))

    claims = write_batch(args.out / "main", "main", main_batch)
    if second:
        write_batch(args.out / "second", "second", second)
    key = {
        "meta": {
            "created": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "seed": args.seed,
            "results": [str(p) for p in args.results],
            "answers": len(items),
            "second_grader": len(second),
        },
        "answers": {item["code"]: {"question_id": item["question_id"], **item["source"]} for item in items},
    }
    (args.out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(
        f"{len(items)} answers ({claims} claim rows) in {args.out / 'main'}; {len(second)} of them also in "
        f"{args.out / 'second'}.\nGive main/ to the main grader and second/ to the second grader (not the question's "
        f"writer). Keep {args.out / 'key.json'} yourself: it says which model and prompt wrote each answer."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
