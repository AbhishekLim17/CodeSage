"""Typer entry point: index, ask, serve, eval. Exposes `app` (see pyproject [project.scripts])."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.table import Table

from codebase_ai.config import Settings, get_settings

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Ask natural-language questions about a codebase and get answers that cite files and line ranges.",
)
console = Console()
err_console = Console(stderr=True)


def _make_output_safe() -> None:
    """Never crash on a character the terminal cannot show.

    Source code is full of emoji and other symbols, and a Windows console or a pipe often uses a legacy code page
    such as cp1252, where printing them raises ``UnicodeEncodeError``. Unencodable characters become ``?`` instead.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (OSError, ValueError):  # a stream that cannot be reconfigured is left as it is
                continue


def _show_version(value: bool) -> None:
    if value:
        from codebase_ai import __version__

        console.print(f"codebase-ai {__version__}", markup=False)
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_show_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:  # no docstring: the app-level help text above is the one to show
    _make_output_safe()


RepoArg = Annotated[
    str,
    typer.Argument(
        metavar="REPO",
        help="Path to the repository, or its git URL (https or ssh; `index` clones it shallowly first).",
    ),
]


class SearchMode(str, Enum):
    vector = "vector"
    keyword = "keyword"
    hybrid = "hybrid"


def _settings() -> Settings:
    try:
        return get_settings()
    except ValidationError as exc:
        err_console.print(f"[red]Invalid configuration:[/red]\n{exc}")
        raise typer.Exit(2) from exc


def _fail(message: str, code: int = 1) -> typer.Exit:
    err_console.print(f"[red]{message}[/red]")
    return typer.Exit(code)


def _repo(text: str, settings: Settings, *, fetch: bool = False, branch: str | None = None) -> Path:
    """The local folder for what the user typed: the folder itself, or the managed clone of a git URL.

    Only ``fetch=True`` (the ``index`` command) touches the network; every other command finds the clone that
    ``index`` made, so asking a question never silently downloads anything.
    """
    from codebase_ai.ingest.git_source import (
        GitSourceError,
        check_branch,
        clone_dir_for,
        looks_like_url,
        parse_git_url,
        sync_repo,
    )

    if not looks_like_url(text):
        if branch is not None:
            raise _fail("--branch only applies to a git URL.", 2)
        path = Path(text).expanduser()
        if not path.is_dir():
            raise _fail(f"'{text}' is not a directory.", 2)
        return path.resolve()
    try:
        remote = parse_git_url(text)
        check_branch(branch)
    except GitSourceError as exc:
        raise _fail(str(exc), 2) from exc  # a URL or branch that is not acceptable is a usage error
    if fetch:
        console.print(f"Fetching {remote.display} (shallow clone)...", markup=False)
        try:
            return sync_repo(remote, settings.index_dir, branch=branch).resolve()
        except GitSourceError as exc:
            raise _fail(str(exc)) from exc
    folder = clone_dir_for(remote, settings.index_dir)
    if not folder.is_dir():
        raise _fail(f"{remote.display} has not been cloned yet. Run: codebase-ai index {text}")
    return folder.resolve()


def _shown(text: str, repo: Path) -> str:
    """How to refer to a repository in a hint: the URL the user gave, or the resolved folder."""
    from codebase_ai.ingest.git_source import looks_like_url

    return text if looks_like_url(text) else str(repo)


@contextmanager
def _built_index(repo_arg: str, repo: Path, settings: Settings) -> Iterator:
    """Open a repository's index for one command, failing with a hint if it has not been built; always closed."""
    from codebase_ai.index.indexer import RepoIndex

    repo_index = RepoIndex(repo, settings.index_dir)
    try:
        if not repo_index.is_built():
            raise _fail(f"No index for {_shown(repo_arg, repo)}. Run: codebase-ai index {_shown(repo_arg, repo)}")
        yield repo_index
    finally:
        repo_index.close()


def _limits(settings: Settings):
    from codebase_ai.ingest.walker import WalkLimits

    return WalkLimits(max_file_bytes=settings.max_file_bytes, max_config_bytes=settings.max_config_bytes)


@app.command()
def index(
    repo_arg: RepoArg,
    full: Annotated[bool, typer.Option("--full", help="Discard the existing index and rebuild from scratch.")] = False,
    branch: Annotated[str | None, typer.Option(help="Branch to clone, for a git URL (default: the remote's default).")] = None,
) -> None:
    """Index a repository (incremental: only new or changed files are re-embedded).

    A git URL is cloned shallowly (latest commit only) into a folder under the index directory, and updated in place
    on later runs. Nothing is ever written into a repository you point at by path.
    """
    from codebase_ai.index.embedder import EmbeddingError, create_embedder
    from codebase_ai.index.indexer import Indexer, IndexMismatchError, IndexReport, RepoIndex

    settings = _settings()
    repo = _repo(repo_arg, settings, fetch=True, branch=branch)
    embedder = create_embedder(settings)
    repo_index = RepoIndex(repo, settings.index_dir)
    indexer = Indexer(
        repo_index,
        embedder,
        limits=_limits(settings),
        chunk_max_lines=settings.chunk_max_lines,
        window_lines=settings.window_lines,
        window_overlap=settings.window_overlap,
        batch_size=settings.embed_batch_size,
    )
    try:
        with console.status("Indexing...") as status:

            def progress(r: IndexReport) -> None:
                status.update(f"Indexing... {r.files_indexed} files changed/new, {r.chunks_added} chunks")

            report = indexer.run(full=full, progress=progress)
        searchable = repo_index.keyword.count()
    except (IndexMismatchError, EmbeddingError) as exc:
        raise _fail(str(exc)) from exc
    finally:
        repo_index.close()

    table = Table(title=f"Indexed {repo.name}", show_header=False)
    table.add_row("Embedding model", report.embedding_model_id)
    table.add_row("Files seen", str(report.files_seen))
    table.add_row("Files indexed (new / changed)", f"{report.files_indexed} ({report.files_new} / {report.files_changed})")
    table.add_row("Files unchanged", str(report.files_unchanged))
    table.add_row("Files removed", str(report.files_removed))
    table.add_row("Chunks added / removed", f"{report.chunks_added} / {report.chunks_removed}")
    table.add_row("Time", f"{report.seconds:.1f}s")
    table.add_row("Index location", str(repo_index.dir))
    console.print(table)
    if report.skipped:
        skipped = Table(title="Skipped", show_header=True)
        skipped.add_column("Reason")
        skipped.add_column("Count", justify="right")
        skipped.add_column("Examples")
        for reason, count in report.skipped.most_common():
            skipped.add_row(reason, str(count), ", ".join(report.skipped_examples.get(reason, [])[:3]))
        console.print(skipped)
    if settings.embedding_provider == "openai":
        console.print("[yellow]Note: chunk text was sent to OpenAI to compute embeddings.[/yellow]")
    if searchable == 0:
        raise _fail(f"Nothing to search: no source code or documentation was found in {repo}. Is it the right folder?")


@app.command()
def stats(repo_arg: RepoArg) -> None:
    """Show what is stored in a repository's index."""
    settings = _settings()
    repo = _repo(repo_arg, settings)
    with _built_index(repo_arg, repo, settings) as repo_index:
        info = repo_index.info()
        table = Table(title=f"Index for {repo.name}", show_header=False)
        table.add_row("Location", str(repo_index.dir))
        table.add_row("Indexed root", str(info["repo_root"]))
        table.add_row("Embedding model", str(info["embedding_model_id"]))
        table.add_row("Embedding dimension", str(info["embedding_dim"]))
        table.add_row("Files indexed", str(len(repo_index.manifest.files())))
        table.add_row("Chunks (keyword / vector)", f"{repo_index.keyword.count()} / {repo_index.vectors.count()}")
        console.print(table)


@app.command()
def chunks(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, resolve_path=True, help="File to chunk.")],
    text: Annotated[bool, typer.Option("--text", help="Also print each chunk's text.")] = False,
) -> None:
    """Show how one file would be chunked (debug aid; no index or model needed)."""
    from codebase_ai.chunking import chunk_file
    from codebase_ai.ingest.walker import load_source_file

    settings = _settings()
    source, reason = load_source_file(file, file.name, _limits(settings))
    if source is None:
        raise _fail(f"{file.name} would not be indexed (reason: {reason}).")
    found = chunk_file(
        source,
        "debug",
        max_lines=settings.chunk_max_lines,
        window_lines=settings.window_lines,
        window_overlap=settings.window_overlap,
    )
    console.print(f"{file.name}: {len(found)} chunks ({source.language})")
    for chunk in found:
        console.print(f"  {chunk.kind:<11} {chunk.start_line:>5}-{chunk.end_line:<5} {chunk.symbol or ''}")
        if text:
            console.print(chunk.text, markup=False, highlight=False)
            console.print("  " + "-" * 40)


@app.command()
def search(
    repo_arg: RepoArg,
    query: Annotated[str, typer.Argument(help="What to look for.")],
    k: Annotated[int, typer.Option("-k", help="Maximum number of sources to show.")] = 8,
    mode: Annotated[
        SearchMode | None,
        typer.Option(help="vector (embeddings), keyword (BM25) or hybrid. Default: the RETRIEVAL_MODE setting."),
    ] = None,
    text: Annotated[bool, typer.Option("--text", help="Print each source's full text.")] = False,
) -> None:
    """Retrieve the code that best answers a query: ranked, merged, cited as path:start-end.

    The sources are exactly what `ask` would give the model, including the repository map for whole-project questions.
    """
    from codebase_ai.index.embedder import EmbeddingError, create_embedder
    from codebase_ai.rag.answerer import create_retriever
    from codebase_ai.retrieval.reranker import create_reranker
    from codebase_ai.retrieval.retriever import RetrievalError

    settings = _settings()
    repo = _repo(repo_arg, settings)
    chosen = mode.value if mode is not None else settings.retrieval_mode
    with _built_index(repo_arg, repo, settings) as repo_index:
        try:
            embedder = None if chosen == "keyword" else create_embedder(settings)
            retriever = create_retriever(settings, repo_index, embedder, mode=chosen, reranker=create_reranker(settings))
            result = retriever.retrieve(query)
        except (EmbeddingError, RetrievalError) as exc:
            raise _fail(str(exc)) from exc

    if not result.sources:
        console.print("No results.")
        return
    if result.overview:
        console.print("A question about the whole project: the generated repository map is added.", markup=False)
    for number, source in enumerate(result.sources[:k], start=1):
        label = source.role if source.role in ("context", "map") else "/".join(source.kinds)
        console.print(f"{number:>2}. {source.location}  [{label}] {', '.join(source.symbols)}", markup=False, highlight=False)
        if text:
            console.print(source.text, markup=False, highlight=False)
            console.print("  " + "-" * 40)
        else:
            first = next((line.strip() for line in source.text.split("\n") if line.strip()), "")
            console.print(f"      {first[:100]}", markup=False, highlight=False)
    shown = min(k, len(result.sources))
    console.print(f"{shown} of {len(result.sources)} sources, about {result.tokens} tokens ({chosen})", markup=False)


class ProviderChoice(str, Enum):
    anthropic = "anthropic"
    openai = "openai"
    ollama = "ollama"


def _print_sources(answer) -> None:
    """The retrieved code, split into what the answer cited and what it did not."""
    from codebase_ai.rag.context import describe_source

    def line(number: int, source) -> None:
        label = describe_source(source)
        console.print(f"  [{number}] {source.location}  {label}", markup=False, highlight=False)

    if answer.cited_sources:
        console.print("\nSources cited", markup=False)
        for number, source in answer.cited_sources:
            line(number, source)
    if answer.uncited_sources:
        console.print("\nAlso retrieved (not cited)", markup=False)
        for number, source in answer.uncited_sources:
            line(number, source)


@app.command()
def ask(
    repo_arg: RepoArg,
    question: Annotated[str, typer.Argument(help="What you want to know about the code.")],
    provider: Annotated[
        ProviderChoice | None, typer.Option(help="Which LLM answers. Default: the LLM_PROVIDER setting.")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Model name for that provider. Default: its *_MODEL setting.")] = None,
    mode: Annotated[
        SearchMode | None, typer.Option(help="How code is retrieved. Default: the RETRIEVAL_MODE setting.")
    ] = None,
    sources: Annotated[bool, typer.Option("--sources/--no-sources", help="List the retrieved sources after the answer.")] = True,
) -> None:
    """Ask a question about an indexed repository; the answer cites the code it is based on by number."""
    from codebase_ai.index.embedder import EmbeddingError, create_embedder
    from codebase_ai.llm.base import ProviderError, create_provider
    from codebase_ai.rag.answerer import create_answerer
    from codebase_ai.retrieval.reranker import create_reranker
    from codebase_ai.retrieval.retriever import RetrievalError

    settings = _settings()
    repo = _repo(repo_arg, settings)
    if not question.strip():
        raise _fail("The question is empty.", 2)
    chosen = mode.value if mode is not None else settings.retrieval_mode
    with _built_index(repo_arg, repo, settings) as repo_index:
        try:
            llm = create_provider(
                settings, provider=provider.value if provider is not None else None, model=model
            )
            embedder = None if chosen == "keyword" else create_embedder(settings)
            answerer = create_answerer(settings, repo_index, embedder, llm, mode=chosen, reranker=create_reranker(settings))
            stream = answerer.stream(question)  # retrieval happens here, before anything is sent to the model
        except (ProviderError, EmbeddingError, RetrievalError) as exc:
            raise _fail(str(exc)) from exc

    if stream.sources and llm.sends_code_off_machine:
        err_console.print(
            f"[yellow]Note: your question and {len(stream.sources)} retrieved code excerpt(s) are sent to "
            f"{llm.name} ({llm.model}).[/yellow]"
        )
    try:
        for piece in stream:
            console.print(piece, end="", markup=False, highlight=False, soft_wrap=True)
    except ProviderError as exc:
        console.print()
        if not stream.text_so_far:  # no answer, but the search itself worked: show where the answer lives
            from codebase_ai.rag.context import describe_source

            console.print("No answer was generated. The code it would have been based on:", markup=False)
            for number, source in enumerate(stream.sources, start=1):
                console.print(f"  [{number}] {source.location}  {describe_source(source)}", markup=False, highlight=False)
        detail = "" if not stream.text_so_far else " The answer above is incomplete."
        raise _fail(f"{exc}{detail}") from exc
    console.print()

    answer = stream.answer
    for warning in answer.warnings:
        console.print(f"Warning: {warning}", style="yellow", markup=False, highlight=False)
    if sources and not answer.no_context:
        _print_sources(answer)
    usage = answer.usage
    tokens = ""
    if usage is not None and usage.input_tokens is not None and usage.output_tokens is not None:
        tokens = f", {usage.input_tokens} tokens in / {usage.output_tokens} out"
    if not answer.no_context:
        console.print(
            f"\n{answer.provider} {answer.model}{tokens}; retrieval {answer.retrieval_seconds:.1f}s, "
            f"answer {answer.generation_seconds:.1f}s",
            style="dim",
            markup=False,
            highlight=False,
        )


@app.command()
def serve(
    repo: Annotated[
        str | None,
        typer.Argument(
            metavar="[REPO]",
            help="Repository to open first: a path or a git URL (it can also be chosen in the UI).",
        ),
    ] = None,
    port: Annotated[int, typer.Option(help="Port to serve the UI on.")] = 8501,
    host: Annotated[
        str,
        typer.Option(help="Address to listen on. The default, localhost, keeps the UI reachable from this computer only."),
    ] = "localhost",
) -> None:
    """Launch the Streamlit chat UI in your browser."""
    import importlib.util
    import os
    import subprocess

    settings = _settings()  # a bad .env should stop here, not deep inside the UI process
    target = None
    if repo is not None:
        from codebase_ai.ingest.git_source import GitSourceError, looks_like_url, parse_git_url

        if looks_like_url(repo):
            try:
                target = parse_git_url(repo).display  # validated only: the UI clones when the user asks it to
            except GitSourceError as exc:
                raise _fail(str(exc), 2) from exc
        else:
            target = str(_repo(repo, settings))
    if importlib.util.find_spec("streamlit") is None:
        raise _fail("Streamlit is not installed. Install it with: pip install streamlit")
    script = Path(__file__).parent / "ui" / "streamlit_app.py"
    env = dict(os.environ)
    if target is not None:
        env["CODEBASE_AI_REPO"] = target
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--browser.gatherUsageStats",
        "false",
        # The watcher exists to reload on source edits, which users of the tool never make. Once the embedding model
        # is imported it also floods the log with tracebacks from transformers' lazily loaded modules.
        "--server.fileWatcherType",
        "none",
        # A local tool has nothing to deploy: hide Streamlit's Deploy button and its in-app promotions.
        "--client.toolbarMode",
        "minimal",
        "--logger.hideWelcomeMessage",
        "true",
    ]
    console.print(f"Codebase Q&A is starting at http://{host}:{port} (press Ctrl+C to stop).", markup=False)
    try:
        code = subprocess.call(command, env=env)
    except KeyboardInterrupt:
        code = 0
    raise typer.Exit(code)


def _eval_table(title: str, reports) -> Table:
    table = Table(title=title, box=box.SIMPLE)
    for column in ("Config", "Hit@1", "Hit@3", "Hit@5", "MRR", "MRR*", "In ctx", "Sym", "Tok", "ms"):
        table.add_column(column, justify="left" if column == "Config" else "right", no_wrap=True)
    for report in reports:
        s = report.summary
        symbols = f"{s['symbol_in_context']:.0%}" if "symbol_in_context" in s else "-"
        table.add_row(
            report.name,
            f"{s['strict_hit@1']:.0%}",
            f"{s['strict_hit@3']:.0%}",
            f"{s['strict_hit@5']:.0%}",
            f"{s['strict_mrr']:.3f}",
            f"{s['lenient_mrr']:.3f}",
            f"{s['context_recall']:.0%}",
            symbols,
            f"{s['avg_tokens']:.0f}",
            f"{s['avg_ms']:.0f}",
        )
    return table


@app.command("eval")
def eval_cmd(
    repo_arg: RepoArg,
    questions: Annotated[
        Path,
        typer.Option("--questions", "-q", exists=True, dir_okay=False, resolve_path=True, help="JSONL question file."),
    ],
    modes: Annotated[str, typer.Option(help="Comma-separated retrieval modes to compare.")] = "vector,keyword,hybrid",
    keyword_weight: Annotated[
        float | None, typer.Option(help="Weight of keyword results in hybrid fusion (default: KEYWORD_WEIGHT setting).")
    ] = None,
    test_penalty: Annotated[
        float | None,
        typer.Option(help="Score multiplier for test files, 1 = off (default: TEST_PENALTY setting)."),
    ] = None,
    symbol_lookups: Annotated[
        bool, typer.Option("--symbol-lookups", help="Also score identifier lookups derived from the gold symbols.")
    ] = False,
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write per-question results here.")] = None,
) -> None:
    """Score retrieval on a labelled question set (Hit@k, MRR, context recall) for each mode."""
    import json
    from dataclasses import asdict

    from codebase_ai.evaluation import evaluate, load_questions, symbol_lookup_questions
    from codebase_ai.index.embedder import EmbeddingError, create_embedder
    from codebase_ai.rag.answerer import create_retriever
    from codebase_ai.retrieval.reranker import create_reranker
    from codebase_ai.retrieval.retriever import MODES, RetrievalError

    settings = _settings()
    repo = _repo(repo_arg, settings)
    wanted = [m.strip() for m in modes.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in MODES]
    if unknown or not wanted:
        raise _fail(f"Unknown mode(s) {unknown or modes!r}; choose from {', '.join(MODES)}.", 2)
    try:
        loaded = load_questions(questions)
    except ValueError as exc:
        raise _fail(str(exc), 2) from exc

    weight = settings.keyword_weight if keyword_weight is None else keyword_weight
    penalty = settings.test_penalty if test_penalty is None else test_penalty
    lookups = symbol_lookup_questions(loaded) if symbol_lookups else []
    with _built_index(repo_arg, repo, settings) as repo_index:
        indexed = set(repo_index.manifest.files())
        missing = sorted({f for q in loaded for f in (*q.gold_files, *q.acceptable_files)} - indexed)
        if missing:
            err_console.print(
                f"[yellow]Warning: {len(missing)} labelled file(s) are not in the index and can never be found, "
                f"e.g. {', '.join(missing[:3])}[/yellow]"
            )
        try:
            embedder = create_embedder(settings) if any(m != "keyword" for m in wanted) else None
            reranker = create_reranker(settings)
            retrievers = {
                mode: create_retriever(
                    settings,
                    repo_index,
                    embedder,
                    mode=mode,
                    reranker=reranker,
                    keyword_weight=weight,
                    test_penalty=penalty,
                    overview=False,  # scores search alone; the map has its own evaluation
                )
                for mode in wanted
            }
            reports = [evaluate(r, loaded, name=mode) for mode, r in retrievers.items()]
            lookup_reports = [evaluate(r, lookups, name=mode) for mode, r in retrievers.items()] if lookups else []
        except (EmbeddingError, RetrievalError) as exc:
            raise _fail(str(exc)) from exc

    setting_note = f"keyword weight {weight:g} in hybrid, test demotion x{penalty:g}"
    console.print(_eval_table(f"{repo.name}: {len(loaded)} questions ({setting_note})", reports))
    if lookup_reports:
        console.print(_eval_table(f"{repo.name}: {len(lookups)} identifier lookups", lookup_reports))
    console.print(
        "Hit@k: a gold file is among the top k files. MRR*: lenient, also accepts acceptable_files. "
        "In ctx: a gold file is in the retrieved context. Sym: gold symbols found in context. Tok: avg context tokens.",
        markup=False,
    )

    primary = next((r for r in reports if r.name == settings.retrieval_mode), reports[-1])
    misses = primary.misses(5)
    if misses:
        console.print(f"\n{primary.name}: {len(misses)} question(s) with no gold file in the top 5:")
        by_id = {q.id: q for q in loaded}
        for miss in misses:
            console.print(f"  {miss.id}  {by_id[miss.id].question}", markup=False, highlight=False)
            console.print(f"      gold: {', '.join(by_id[miss.id].gold_files)}", markup=False, highlight=False)
            console.print(f"      got:  {', '.join(miss.top_files[:3]) or '-'}", markup=False, highlight=False)

    if json_out is not None:
        payload = {r.name: {"summary": r.summary, "questions": [asdict(x) for x in r.results]} for r in reports}
        if lookup_reports:
            payload["symbol_lookups"] = {
                r.name: {"summary": r.summary, "questions": [asdict(x) for x in r.results]} for r in lookup_reports
            }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
