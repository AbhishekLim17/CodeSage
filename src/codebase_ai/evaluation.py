"""Retrieval evaluation: labelled questions, per-question results, and an aggregate report.

A question file is JSONL, one object per line::

    {"id": "q01", "type": "explain", "question": "How does login check suspension?",
     "gold_files": ["src/contexts/AuthContext.jsx"],
     "acceptable_files": ["docs/ARCHITECTURE.md"],      # optional: prose that also answers it
     "gold_symbols": ["orgIsSuspended"],                # optional
     "gold_dirs": ["src/services", "scripts"]}          # optional: for overview questions, the areas an answer needs

``gold_dirs`` (overview questions only) are the areas of the repository an answer needs to mention; they score *area
coverage*, the share of those directories that the context shows (code from inside them, or a line in the repository
map).

*Strict* metrics count only ``gold_files``. *Lenient* metrics also accept ``acceptable_files``, so an overview
document that legitimately answers the question is not scored as a miss. Both are always reported.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

from codebase_ai.metrics import DEFAULT_KS, first_gold_rank, summarize
from codebase_ai.retrieval.overview import OverviewRetriever
from codebase_ai.retrieval.repo_map import map_lists_directory
from codebase_ai.retrieval.retriever import Retriever, Source


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    type: str
    gold_files: tuple[str, ...]
    acceptable_files: tuple[str, ...] = ()
    gold_symbols: tuple[str, ...] = ()
    gold_dirs: tuple[str, ...] = ()
    key_facts: tuple[str, ...] = ()  # statements a correct answer must contain; shown to human graders

    @property
    def lenient_files(self) -> frozenset[str]:
        return frozenset(self.gold_files) | frozenset(self.acceptable_files)


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """Read and validate a JSONL question file; raises ``ValueError`` naming the offending line."""
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not valid JSON ({exc.msg})") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")  # noqa: TRY004 - bad file content, not a type bug
        for key in ("id", "question", "gold_files"):
            if not raw.get(key):
                raise ValueError(f"{path}:{number}: missing or empty '{key}'")
        if raw["id"] in seen:
            raise ValueError(f"{path}:{number}: duplicate id '{raw['id']}'")
        seen.add(raw["id"])
        for key in ("gold_files", "acceptable_files", "gold_symbols", "gold_dirs", "key_facts"):
            value = raw.get(key, [])
            if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
                raise ValueError(f"{path}:{number}: '{key}' must be a list of non-empty strings")
        questions.append(
            EvalQuestion(
                id=raw["id"],
                question=raw["question"],
                type=raw.get("type", "other"),
                gold_files=tuple(raw["gold_files"]),
                acceptable_files=tuple(raw.get("acceptable_files", [])),
                gold_symbols=tuple(raw.get("gold_symbols", [])),
                gold_dirs=tuple(raw.get("gold_dirs", [])),
                key_facts=tuple(raw.get("key_facts", [])),
            )
        )
    if not questions:
        raise ValueError(f"{path}: no questions found")
    return questions


def symbol_lookup_questions(questions: Sequence[EvalQuestion]) -> list[EvalQuestion]:
    """Derive an identifier-lookup query from each question that has a gold symbol.

    The query is just the bare symbol, as a developer would type it into a search box ("computeCriticalPath"), and it
    is scored against the same ``gold_files`` and ``acceptable_files``. Natural-language questions favour embeddings;
    this slice checks the other kind of query, where exact-name matching should matter.
    """
    return [
        EvalQuestion(
            id=f"{q.id}-sym",
            question=q.gold_symbols[0],
            type="symbol",
            gold_files=q.gold_files,
            acceptable_files=q.acceptable_files,
        )
        for q in questions
        if q.gold_symbols
    ]


def symbol_in_sources(symbols: Sequence[str], source_texts: Sequence[str]) -> bool:
    """True if any gold symbol (last segment of ``Class.method``) appears as a whole word in any source text."""
    names = [s.rsplit(".", 1)[-1] for s in symbols]
    patterns = [re.compile(rf"(?<![\w$]){re.escape(name)}(?![\w$])") for name in names]
    return any(p.search(text) for p in patterns for text in source_texts)


def area_coverage(sources: Sequence[Source], gold_dirs: Sequence[str]) -> float | None:
    """Share of ``gold_dirs`` that the context shows: code from inside the directory, or a mention in the map.

    ``None`` when there are no gold directories. A directory named only in the repository map counts, because telling
    the model the area exists is what the map is for; the metric therefore says whether an overview *mentions* the
    right areas, not whether it explains them (that needs a judge; see the answer-quality evaluation).
    """
    if not gold_dirs:
        return None

    def covered(directory: str) -> bool:
        wanted = directory.strip("/")
        for source in sources:
            if source.role == "map":
                if map_lists_directory(source.text, wanted):
                    return True
            elif source.path.startswith(wanted + "/"):
                return True
        return False

    return sum(covered(d) for d in gold_dirs) / len(gold_dirs)


@dataclass(frozen=True)
class QuestionResult:
    id: str
    type: str
    rank: int | None  # first gold file in the ranking (strict)
    lenient_rank: int | None  # first gold-or-acceptable file
    in_context: bool  # a gold file is among the sources that fit the budget
    lenient_in_context: bool
    symbol_in_context: bool | None  # None when the question has no gold_symbols
    tokens: int
    sources: int
    milliseconds: float
    top_files: tuple[str, ...]  # best-ranked files, for diagnosing misses
    area_coverage: float | None = None  # share of gold_dirs shown in the context; None when there are none
    overview: bool = False  # the retriever treated it as a question about the whole repository (added the map)


@dataclass
class EvalReport:
    name: str
    results: list[QuestionResult] = field(default_factory=list)

    def _summary_for(self, results: Sequence[QuestionResult]) -> dict[str, float]:
        if not results:
            return {}
        strict = summarize([r.rank for r in results], DEFAULT_KS)
        lenient = summarize([r.lenient_rank for r in results], DEFAULT_KS)
        labelled = [r.symbol_in_context for r in results if r.symbol_in_context is not None]
        summary = {f"strict_{k}": v for k, v in strict.items()}
        summary.update({f"lenient_{k}": v for k, v in lenient.items()})
        summary["context_recall"] = fmean(1.0 if r.in_context else 0.0 for r in results)
        summary["lenient_context_recall"] = fmean(1.0 if r.lenient_in_context else 0.0 for r in results)
        if labelled:
            summary["symbol_in_context"] = fmean(1.0 if hit else 0.0 for hit in labelled)
        covered = [r.area_coverage for r in results if r.area_coverage is not None]
        if covered:
            summary["area_coverage"] = fmean(covered)
        summary["overview_rate"] = fmean(1.0 if r.overview else 0.0 for r in results)
        summary["avg_tokens"] = fmean(r.tokens for r in results)
        summary["avg_sources"] = fmean(r.sources for r in results)
        summary["avg_ms"] = fmean(r.milliseconds for r in results)
        return summary

    @property
    def summary(self) -> dict[str, float]:
        return self._summary_for(self.results)

    def by_type(self) -> dict[str, dict[str, float]]:
        kinds = sorted({r.type for r in self.results})
        return {kind: self._summary_for([r for r in self.results if r.type == kind]) for kind in kinds}

    def misses(self, k: int = 5) -> list[QuestionResult]:
        """Questions with no gold file in the top ``k`` files (strict)."""
        return [r for r in self.results if r.rank is None or r.rank > k]


def evaluate(retriever: Retriever | OverviewRetriever, questions: Sequence[EvalQuestion], name: str | None = None) -> EvalReport:
    """Run every question through ``retriever`` and score the ranking and the context it would hand to an LLM."""
    report = EvalReport(name=name or retriever.mode)
    for q in questions:
        started = time.perf_counter()
        result = retriever.retrieve(q.question)
        elapsed_ms = (time.perf_counter() - started) * 1000
        ranked = result.ranked_files
        context = set(result.context_files)
        report.results.append(
            QuestionResult(
                id=q.id,
                type=q.type,
                rank=first_gold_rank(ranked, q.gold_files),
                lenient_rank=first_gold_rank(ranked, q.lenient_files),
                in_context=bool(context & set(q.gold_files)),
                lenient_in_context=bool(context & q.lenient_files),
                symbol_in_context=(
                    symbol_in_sources(q.gold_symbols, [s.text for s in result.sources]) if q.gold_symbols else None
                ),
                area_coverage=area_coverage(result.sources, q.gold_dirs),
                overview=result.overview,
                tokens=result.tokens,
                sources=len(result.sources),
                milliseconds=elapsed_ms,
                top_files=tuple(ranked[:5]),
            )
        )
    return report


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def to_markdown(reports: Sequence[EvalReport], *, title: str | None = None) -> str:
    """A results table with one row per report (e.g. per retrieval mode)."""
    lines = [f"### {title}", ""] if title else []
    lines += [
        "| Config | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Lenient Hit@1 | Lenient MRR | In context | Symbols | Tokens | ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for report in reports:
        s = report.summary
        symbols = _pct(s["symbol_in_context"]) if "symbol_in_context" in s else "—"
        lines.append(
            f"| {report.name} | {_pct(s['strict_hit@1'])} | {_pct(s['strict_hit@3'])} | {_pct(s['strict_hit@5'])} | "
            f"{_pct(s['strict_hit@10'])} | {s['strict_mrr']:.3f} | {_pct(s['lenient_hit@1'])} | "
            f"{s['lenient_mrr']:.3f} | {_pct(s['context_recall'])} | {symbols} | {s['avg_tokens']:.0f} | "
            f"{s['avg_ms']:.0f} |"
        )
    return "\n".join(lines)
