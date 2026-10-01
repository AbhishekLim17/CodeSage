r"""Does demoting release notes (CHANGELOG, HISTORY, NEWS...) help, and does it do any harm? (M5 ablation)

On `psf/requests`, a question about redirects ranked the project's HISTORY.md above the code that implements them.
This runs labelled questions through the current default retrieval (``vector`` with test demotion) and through the same
retrieval with release notes demoted by a few strengths, and compares them question by question with the paired
statistics used for the earlier decisions. The `requests` suite is where release notes compete with code (and includes
four questions that really are about releases, which must not be hurt); the other suites have no release notes, so
they show that nothing changes where the feature does not apply.

    python eval/run_changelog_eval.py \
        --suite requests path/to/requests eval/questions/requests.jsonl \
        --suite magnaflow path/to/MagnaFlow eval/questions/magnaflow.jsonl \
        --suite codebase_ai . eval/questions/codebase_ai.jsonl \
        --suite rich .venv/Lib/site-packages/rich eval/questions/rich.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from codebase_ai.config import get_settings
from codebase_ai.evaluation import EvalQuestion, EvalReport, evaluate, load_questions, to_markdown
from codebase_ai.index.embedder import create_embedder
from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.ingest.walker import WalkLimits
from codebase_ai.metrics import paired_comparison
from codebase_ai.retrieval.retriever import Retriever

BASELINE = "vector + demote tests"
STRENGTHS = (0.5, 0.25)  # score multipliers for release notes; smaller demotes harder


def config_names() -> list[str]:
    return [BASELINE, *(f"{BASELINE} + demote release notes x{s:g}" for s in STRENGTHS)]


def run_suite(name: str, repo: Path, questions_file: Path, index_root: Path, embedder) -> dict:
    settings = get_settings()
    questions = load_questions(questions_file)
    index = RepoIndex(repo, index_root)
    try:
        Indexer(
            index,
            embedder,
            limits=WalkLimits(settings.max_file_bytes, settings.max_config_bytes),
            chunk_max_lines=settings.chunk_max_lines,
            window_lines=settings.window_lines,
            window_overlap=settings.window_overlap,
            batch_size=settings.embed_batch_size,
        ).run()
        penalties = [1.0, *STRENGTHS]
        reports = []
        for config, penalty in zip(config_names(), penalties, strict=True):
            retriever = Retriever(
                index,
                embedder,
                mode="vector",
                top_k=settings.retrieve_top_k,
                budget_tokens=settings.context_token_budget,
                test_penalty=settings.test_penalty,
                changelog_penalty=penalty,
            )
            reports.append(evaluate(retriever, questions, name=config))
        return {"name": name, "questions": questions, "reports": reports}
    finally:
        index.close()


def pooled(suites: list[dict]) -> list[EvalReport]:
    names = [r.name for r in suites[0]["reports"]]
    return [
        EvalReport(name=n, results=[q for s in suites for r in s["reports"] if r.name == n for q in r.results])
        for n in names
    ]


def paired_rows(reports: list[EvalReport]) -> str:
    base = reports[0]
    rows = [
        "| Compared with the baseline | Questions | MRR difference | 95% interval | Better / worse / tied | Sign test p |",
        "|---|---|---|---|---|---|",
    ]
    for other in reports[1:]:
        r = paired_comparison([x.rank for x in other.results], [x.rank for x in base.results])
        mark = " (*)" if r.significant else ""
        rows.append(
            f"| {other.name.removeprefix(BASELINE + ' + ')} | {r.n} | {r.mrr_diff:+.3f}{mark} | "
            f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.better} / {r.worse} / {r.tied} | {r.sign_p:.3f} |"
        )
    rows += ["", "(*) = the 95% interval excludes zero, i.e. the difference is unlikely to be noise."]
    return "\n".join(rows)


def changed_questions(suite: dict) -> str:
    """Per question: where the right file ranked without and with the strongest demotion, for the ones that moved."""
    by_id: dict[str, EvalQuestion] = {q.id: q for q in suite["questions"]}
    base, strongest = suite["reports"][0], suite["reports"][-1]
    lines = ["| Question | Rank without | Rank with |", "|---|---|---|"]
    moved = False
    for a, b in zip(base.results, strongest.results, strict=True):
        if a.rank != b.rank:
            moved = True
            lines.append(f"| {a.id}: {by_id[a.id].question} | {a.rank or 'none'} | {b.rank or 'none'} |")
    return "\n".join(lines) if moved else "No question changed rank."


def render(suites: list[dict], env: dict) -> str:
    intro = (
        f"Run on {env['date']} · {env['machine']} · embedding model `{env['embedding_model']}`. "
        "Generated by `eval/run_changelog_eval.py`."
    )
    all_reports = pooled(suites)
    parts = [intro, "", "### Pooled: all suites", "", to_markdown(all_reports), "", paired_rows(all_reports)]
    for suite in suites:
        parts += [
            "",
            f"### {suite['name']}",
            "",
            to_markdown(suite["reports"]),
            "",
            paired_rows(suite["reports"]),
            "",
            "Questions whose right-file rank changed (without vs with the strongest demotion):",
            "",
            changed_questions(suite),
        ]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", nargs=3, action="append", metavar=("NAME", "REPO", "QUESTIONS"), required=True)
    parser.add_argument("--index-root", type=Path, default=None, help="where indexes live (default: INDEX_DIR)")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=Path(__file__).parent / "results" / "changelog_eval.json")
    args = parser.parse_args()

    settings = get_settings()
    embedder = create_embedder(settings)
    index_root = args.index_root or settings.index_dir
    suites = [run_suite(n, Path(r).resolve(), Path(q), index_root, embedder) for n, r, q in args.suite]
    env = {
        "date": datetime.now(tz=UTC).date().isoformat(),
        "machine": platform.platform(),
        "embedding_model": embedder.model_id,
    }
    text = render(suites, env)
    print(text)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "env": env,
                "suites": {
                    s["name"]: {r.name: {"summary": r.summary, "questions": [asdict(x) for x in r.results]} for r in s["reports"]}
                    for s in suites
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.out_md:
        args.out_md.write_text(text, encoding="utf-8")
        print(f"wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
