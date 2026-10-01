r"""Run the retrieval evaluation over several repositories and write a report (M2).

For every suite: index the repo (incremental, so re-runs are cheap), then score each retrieval configuration on two
kinds of query: the labelled natural-language questions, and identifier lookups derived from their gold symbols
("where is computeCriticalPath?"). Results are pooled across suites and also shown per suite.

    python eval/run_eval.py \
        --suite magnaflow  path/to/MagnaFlow  eval/questions/magnaflow.jsonl \
        --suite codebase_ai .                 eval/questions/codebase_ai.jsonl \
        --out-md docs/RETRIEVAL_EVAL.md
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from bench_embeddings import ANALYSIS_HEADING, keep_analysis

from codebase_ai.config import get_settings
from codebase_ai.evaluation import (
    EvalQuestion,
    EvalReport,
    evaluate,
    load_questions,
    symbol_lookup_questions,
    to_markdown,
)
from codebase_ai.index.embedder import create_embedder
from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.ingest.walker import WalkLimits
from codebase_ai.metrics import paired_comparison
from codebase_ai.retrieval.retriever import Retriever


def configurations(keyword_weights: list[float], test_penalty: float) -> list[tuple[str, dict]]:
    """Retriever settings for each ablation. Every config states ``test_penalty`` explicitly: the baselines must be
    *without* test demotion whatever the library default is, or the demotion ablation measures nothing."""
    configs: list[tuple[str, dict]] = [
        ("vector", {"mode": "vector", "test_penalty": 1.0}),
        ("keyword", {"mode": "keyword", "test_penalty": 1.0}),
        ("hybrid", {"mode": "hybrid", "keyword_weight": 1.0, "test_penalty": 1.0}),
    ]
    configs += [
        (f"hybrid kw={w:g}", {"mode": "hybrid", "keyword_weight": w, "test_penalty": 1.0})
        for w in keyword_weights
        if w != 1.0
    ]
    if test_penalty < 1.0:
        configs += [
            (f"{name} + demote tests", {**kwargs, "test_penalty": test_penalty})
            for name, kwargs in list(configs)
            if name in ("vector", "hybrid", "hybrid kw=0.5")
        ]
    return configs


# (A, B): is configuration A better than B? Compared question by question on the pooled questions.
PAIRS = [
    ("hybrid", "vector"),
    ("hybrid kw=0.5", "vector"),
    ("keyword", "vector"),
    ("vector + demote tests", "vector"),
    ("hybrid + demote tests", "hybrid"),
    ("hybrid + demote tests", "vector + demote tests"),
    ("hybrid kw=0.5 + demote tests", "vector + demote tests"),
]


def paired_table(pooled: list[EvalReport]) -> str:
    """Markdown table of paired comparisons; a difference only counts if its 95% interval excludes zero."""
    by_name = {r.name: r for r in pooled}
    rows = [
        "| A vs B | Questions | MRR difference (A - B) | 95% interval | A better / worse / tied | Sign test p |",
        "|---|---|---|---|---|---|",
    ]
    for a, b in PAIRS:
        if a not in by_name or b not in by_name:
            continue
        r = paired_comparison([x.rank for x in by_name[a].results], [x.rank for x in by_name[b].results])
        mark = " (*)" if r.significant else ""
        rows.append(
            f"| {a} vs {b} | {r.n} | {r.mrr_diff:+.3f}{mark} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | "
            f"{r.better} / {r.worse} / {r.tied} | {r.sign_p:.3f} |"
        )
    rows.append("")
    rows.append("(*) = the 95% interval excludes zero, i.e. the difference is unlikely to be noise.")
    return "\n".join(rows)


def pool(name: str, reports: list[EvalReport]) -> EvalReport:
    """Every question from every suite counted once (micro-average)."""
    return EvalReport(name=name, results=[r for report in reports for r in report.results])


def run_configs(index: RepoIndex, embedder, configs, questions: list[EvalQuestion]) -> list[EvalReport]:
    settings = get_settings()
    return [
        evaluate(
            Retriever(
                index,
                embedder,
                top_k=settings.retrieve_top_k,
                budget_tokens=settings.context_token_budget,
                **kwargs,
            ),
            questions,
            name=name,
        )
        for name, kwargs in configs
    ]


def run_suite(name: str, repo: Path, questions_file: Path, index_root: Path, embedder, configs) -> dict:
    settings = get_settings()
    questions = load_questions(questions_file)
    symbol_questions = symbol_lookup_questions(questions)
    index = RepoIndex(repo, index_root)
    try:
        indexer = Indexer(
            index,
            embedder,
            limits=WalkLimits(settings.max_file_bytes, settings.max_config_bytes),
            chunk_max_lines=settings.chunk_max_lines,
            window_lines=settings.window_lines,
            window_overlap=settings.window_overlap,
            batch_size=settings.embed_batch_size,
        )
        report = indexer.run()
        print(
            f"[{name}] index: {report.files_indexed} new/changed, {report.files_unchanged} unchanged, "
            f"{index.keyword.count()} chunks ({report.seconds:.0f}s)",
            flush=True,
        )
        indexed = set(index.manifest.files())
        labelled = {f for q in questions for f in (*q.gold_files, *q.acceptable_files)}
        if labelled - indexed:
            print(f"[{name}] WARNING labelled files not in the index: {sorted(labelled - indexed)[:5]}", flush=True)

        reports = run_configs(index, embedder, configs, questions)
        symbol_reports = run_configs(index, embedder, configs, symbol_questions)
        for nl, sym in zip(reports, symbol_reports, strict=True):
            print(
                f"[{name}] {nl.name:<28} questions mrr={nl.summary['strict_mrr']:.3f} "
                f"hit@1={nl.summary['strict_hit@1']:.2f} | symbols mrr={sym.summary['strict_mrr']:.3f} "
                f"hit@1={sym.summary['strict_hit@1']:.2f}",
                flush=True,
            )
        return {
            "name": name,
            "repo": repo.name,
            "questions": questions,
            "symbol_questions": symbol_questions,
            "reports": reports,
            "symbol_reports": symbol_reports,
            "files": len(indexed),
            "chunks": index.keyword.count(),
        }
    finally:
        index.close()


def render(suites: list[dict], pooled: list[EvalReport], pooled_symbols: list[EvalReport], env: dict) -> str:
    total = sum(len(s["questions"]) for s in suites)
    total_symbols = sum(len(s["symbol_questions"]) for s in suites)
    intro = (
        f"Run on {env['date']} · {env['machine']} · embedding model `{env['embedding_model']}`. "
        "Generated by `eval/run_eval.py`; raw per-question results are in `eval/results/retrieval_eval.json` "
        "(not committed)."
    )
    legend = (
        "Columns: **Hit@k** = a gold file is among the top k files of the ranking (strict: `gold_files` only). "
        "**MRR** = mean reciprocal rank (cut at 10). **Lenient** also accepts `acceptable_files` (prose that "
        "legitimately answers the question). **In context** = a gold file is among the sources that fit the "
        "12,000-token budget, i.e. what an LLM would be shown. **Symbols** = share of questions whose gold symbols "
        "appear in that context. **Tokens** = mean estimated context size. Configs: `kw=` is the weight of keyword "
        "results in the fusion (default 1); `demote tests` multiplies the score of test-file chunks by 0.5 unless "
        "the query mentions tests."
    )
    lines = [
        "# Retrieval evaluation (M2)",
        "",
        intro,
        "",
        "## Setup",
        "",
        "| Suite | Repository | Files | Chunks | Questions | Symbol lookups |",
        "|---|---|---|---|---|---|",
    ]
    for s in suites:
        lines.append(
            f"| {s['name']} | `{s['repo']}` | {s['files']} | {s['chunks']} | "
            f"{len(s['questions'])} | {len(s['symbol_questions'])} |"
        )
    lines += [
        "",
        legend,
        "",
        f"## Pooled: natural-language questions ({total})",
        "",
        to_markdown(pooled),
        "",
        "### Paired comparisons (natural-language questions)",
        "",
        paired_table(pooled),
        "",
        f"## Pooled: identifier lookups ({total_symbols})",
        "",
        (
            "One query per question that has a gold symbol: the bare identifier (e.g. `computeCriticalPath`), "
            "scored against the same gold files."
        ),
        "",
        to_markdown(pooled_symbols),
        "",
        "### Paired comparisons (identifier lookups)",
        "",
        paired_table(pooled_symbols),
    ]
    for s in suites:
        lines += [
            "",
            f"## {s['name']}: natural-language questions ({len(s['questions'])})",
            "",
            to_markdown(s["reports"]),
            "",
            f"### {s['name']}: identifier lookups ({len(s['symbol_questions'])})",
            "",
            to_markdown(s["symbol_reports"]),
        ]
        by_id = {q.id: q for q in s["questions"]}
        primary = next(r for r in s["reports"] if r.name == "hybrid")
        misses = primary.misses(5)
        lines += ["", f"**hybrid: natural-language questions with no gold file in the top 5 ({len(misses)})**", ""]
        if not misses:
            lines.append("None.")
        for miss in misses:
            q = by_id[miss.id]
            got = ", ".join(f"`{f}`" for f in miss.top_files[:3]) or "nothing"
            gold = ", ".join(f"`{g}`" for g in q.gold_files)
            lines.append(f"- `{q.id}` {q.question}  \n  gold: {gold}; got: {got}")
    lines += ["", ANALYSIS_HEADING, "", "_(write the interpretation here; it is kept when this file is regenerated)_", ""]
    return "\n".join(lines)


def payload_for(reports: list[EvalReport]) -> dict:
    return {r.name: {"summary": r.summary, "questions": [asdict(x) for x in r.results]} for r in reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", nargs=3, action="append", metavar=("NAME", "REPO", "QUESTIONS"), required=True)
    parser.add_argument("--index-root", type=Path, default=None, help="where indexes live (default: INDEX_DIR)")
    parser.add_argument("--keyword-weights", default="0.5,2", help="extra hybrid keyword weights to compare")
    parser.add_argument("--test-penalty", type=float, default=0.5, help="score multiplier for test files; 1 = skip")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=Path(__file__).parent / "results" / "retrieval_eval.json")
    args = parser.parse_args()

    settings = get_settings()
    index_root = args.index_root or settings.index_dir
    embedder = create_embedder(settings)
    weights = [float(w) for w in args.keyword_weights.split(",") if w.strip()]
    configs = configurations(weights, args.test_penalty)

    suites = [
        run_suite(name, Path(repo).resolve(), Path(questions), index_root, embedder, configs)
        for name, repo, questions in args.suite
    ]
    pooled = [pool(name, [s["reports"][i] for s in suites]) for i, (name, _) in enumerate(configs)]
    pooled_symbols = [pool(name, [s["symbol_reports"][i] for s in suites]) for i, (name, _) in enumerate(configs)]
    print("\npooled (natural-language | identifier lookups):", flush=True)
    for nl, sym in zip(pooled, pooled_symbols, strict=True):
        a, b = nl.summary, sym.summary
        print(
            f"  {nl.name:<28} hit@1={a['strict_hit@1']:.2f} hit@5={a['strict_hit@5']:.2f} mrr={a['strict_mrr']:.3f}"
            f" | hit@1={b['strict_hit@1']:.2f} hit@5={b['strict_hit@5']:.2f} mrr={b['strict_mrr']:.3f}",
            flush=True,
        )

    env = {
        "date": datetime.now(tz=UTC).date().isoformat(),
        "machine": platform.platform(),
        "embedding_model": embedder.model_id,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": env,
        "pooled": {r.name: {"summary": r.summary} for r in pooled},
        "pooled_symbols": {r.name: {"summary": r.summary} for r in pooled_symbols},
        "suites": {s["name"]: payload_for(s["reports"]) for s in suites},
        "symbol_suites": {s["name"]: payload_for(s["symbol_reports"]) for s in suites},
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.out_md:
        report_md = render(suites, pooled, pooled_symbols, env)
        if args.out_md.exists():
            report_md = keep_analysis(args.out_md.read_text(encoding="utf-8"), report_md)
        args.out_md.write_text(report_md, encoding="utf-8")
        print(f"wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
