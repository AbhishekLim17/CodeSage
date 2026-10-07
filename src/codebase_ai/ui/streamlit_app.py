"""Streamlit chat UI: repo selection, indexing progress, provider settings, cited answers with source viewer.

Run it with ``codebase-ai serve [REPO]``. The UI only ever talks to the library the CLI uses (``Indexer``,
``create_answerer``); it holds no retrieval or prompt logic of its own.

Design notes:

* A ``RepoIndex`` is opened for one operation and closed again. Only the embedding model, which is slow to load, is
  kept between reruns.
* Nothing is sent anywhere until a question is asked, and the sidebar says where questions go before that happens.
* A follow-up is rewritten into a standalone question using the last few exchanges (one short model call). The
  rewrite is shown, and the model that answers sees only that question and the retrieved code, never the chat.
* Model output is rendered as markdown with raw HTML disabled, so text that came from the code cannot inject markup.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from codebase_ai.config import Settings, get_settings
from codebase_ai.index.embedder import EmbeddingError
from codebase_ai.index.indexer import (
    Indexer,
    IndexMismatchError,
    IndexReport,
    IndexStatus,
    RepoIndex,
    index_status,
)
from codebase_ai.ingest.git_source import (
    GitSourceError,
    RemoteRepo,
    clone_dir_for,
    looks_like_url,
    parse_git_url,
    sync_repo,
)
from codebase_ai.ingest.walker import WalkLimits
from codebase_ai.llm.base import PROVIDER_NAMES, ProviderError, create_provider
from codebase_ai.rag.answerer import Answer, create_answerer
from codebase_ai.rag.context import describe_source
from codebase_ai.retrieval.query import Turn
from codebase_ai.retrieval.retriever import MODES, RetrievalError, Source

log = logging.getLogger(__name__)

REPO_ENV = "CODEBASE_AI_REPO"  # set by ``codebase-ai serve REPO``
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_KEY_HINTS = {
    "anthropic": "Add ANTHROPIC_API_KEY to .env, or log in with `ant auth login`.",
    "openai": "Add OPENAI_API_KEY to .env.",
}


def escape_math(text: str) -> str:
    """Stop ``$`` in prose (shell variables, prices) from being typeset as maths; code spans are left alone."""
    parts = _CODE.split(text)
    return "".join(part if index % 2 else part.replace("$", "\\$") for index, part in enumerate(parts))


def _label(source: Source, number: int) -> str:
    """``[2] path:10-40 - method Foo.bar`` as one code span, so a path with markdown characters stays literal."""
    return f"[{number}] {source.location} - {describe_source(source)}".replace("`", "'")


@st.cache_resource(show_spinner=False)
def load_embedder(model_id: str):
    """One embedding model per process. ``model_id`` is only the cache key: a changed setting loads a new model."""
    from codebase_ai.index import embedder

    return embedder.create_embedder(get_settings())


@st.cache_resource(show_spinner=False)
def load_reranker(model_name: str):
    """The reranker, if one is configured; like the embedder it is loaded once per process."""
    from codebase_ai.retrieval import reranker

    return reranker.CrossEncoderReranker(model_name)


# --- rendering ---------------------------------------------------------------------------------------------------

FOUND_TITLE = "No answer, but this is the code it would have been based on"


def render_sources(sources: list[tuple[int, Source]], title: str, *, expanded: bool = False) -> None:
    if not sources:
        return
    with st.expander(f"{title} ({len(sources)})", expanded=expanded):
        for number, source in sources:
            st.markdown(f"`{_label(source, number)}`")
            st.code(source.text, language=source.language or None)


def render_details(answer: Answer) -> None:
    """Everything under an answer's text: warnings, where it came from, and the code itself."""
    for warning in answer.warnings:
        st.warning(warning)
    if answer.condensed is not None and answer.condensed.changed:
        st.caption(f"Searched for: {answer.searched_for}")
    if answer.no_context:
        return
    if answer.retrieval.overview:
        st.caption("This looks like a question about the whole project, so a generated map of the repository was added.")
    if answer.cited_sources:
        st.caption("Cited: " + " · ".join(f"[{n}] {source.location}" for n, source in answer.cited_sources))
    render_sources(answer.cited_sources, "Cited sources", expanded=False)
    render_sources(answer.uncited_sources, "Also retrieved, not cited")
    usage = answer.usage
    tokens = ""
    if usage is not None and usage.input_tokens is not None and usage.output_tokens is not None:
        tokens = f" · {usage.input_tokens} tokens in, {usage.output_tokens} out"
    if not answer.no_context:
        st.caption(f"{answer.provider} · {answer.model}{tokens} · answered in {answer.generation_seconds:.1f}s")


def render_turn(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.markdown(escape_math(turn["text"]))
            return
        if turn["text"]:
            st.markdown(escape_math(turn["text"]))
        if turn["error"]:
            st.error(turn["error"])
        render_sources(turn.get("found", []), FOUND_TITLE)
        if turn["answer"] is not None:
            render_details(turn["answer"])


# --- asking ------------------------------------------------------------------------------------------------------


def _stopped_early(turn: dict, text: str, placeholder, reason: str) -> dict:
    """Keep the text that arrived, drop the cursor, and say plainly that the answer stopped."""
    turn["text"] = text
    placeholder.markdown(escape_math(text))
    turn["error"] = reason + (" The answer above is incomplete." if text else "")
    st.error(turn["error"])
    return turn


def history_of(turns: list[dict]) -> list[Turn]:
    """Earlier exchanges that got an answer, oldest first (a question that failed left nothing to refer back to)."""
    return [
        Turn(question["text"], answer["text"])
        for question, answer in zip(turns[::2], turns[1::2], strict=False)
        if answer["answer"] is not None
    ]


def ask_and_render(question: str, config: Config, settings: Settings, history: list[Turn]) -> dict:
    """Answer one question inside the current chat message, streaming as text arrives; returns the turn to keep."""
    turn: dict = {"role": "assistant", "text": "", "error": None, "answer": None}
    with st.chat_message("assistant"):
        try:
            provider = create_provider(settings, provider=config.provider, model=config.model or None)
            embedder = None if config.mode == "keyword" else load_embedder(settings.embedding_model_id())
            with closing(RepoIndex(config.repo, settings.index_dir)) as index:
                reranker = load_reranker(settings.reranker_model) if settings.reranker_model else None
                answerer = create_answerer(settings, index, embedder, provider, mode=config.mode, reranker=reranker)
                with st.spinner("Searching the code... (the first search loads the embedding model and can take a while)"):
                    stream = answerer.stream(question, history)
        except (ProviderError, EmbeddingError, RetrievalError) as exc:
            turn["error"] = str(exc)
            st.error(turn["error"])
            return turn
        except Exception as exc:  # anything unexpected is shown here, not as a stack trace
            log.exception("Unexpected error while retrieving")
            turn["error"] = f"Something went wrong while searching ({type(exc).__name__}): {exc}"
            st.error(turn["error"])
            return turn

        if stream.sources and provider.sends_code_off_machine:
            st.caption(
                f"Sending your question and {len(stream.sources)} code excerpt(s) to {provider.name} ({provider.model})."
            )
        placeholder = st.empty()
        try:
            for _ in stream:
                placeholder.markdown(escape_math(stream.text_so_far) + " ▌")
        except ProviderError as exc:
            turn = _stopped_early(turn, stream.text_so_far, placeholder, str(exc))
            if not stream.text_so_far:  # no answer, but the search worked: still show where the answer lives
                turn["found"] = list(enumerate(stream.sources, start=1))
                render_sources(turn["found"], FOUND_TITLE)
            return turn
        except Exception as exc:
            log.exception("Unexpected error while answering")
            return _stopped_early(turn, stream.text_so_far, placeholder, f"{type(exc).__name__}: {exc}")

        answer = stream.answer
        turn["text"], turn["answer"] = answer.text, answer
        placeholder.markdown(escape_math(answer.text))
        render_details(answer)
    return turn


# --- sidebar -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    repo: Path | None
    status: IndexStatus | None
    provider: str
    model: str
    mode: str
    blocker: str | None  # why questions cannot be asked yet, if they cannot
    remote: RemoteRepo | None = None  # set when the repository was given as a git URL


def _run_index(
    repo: Path | None, remote: RemoteRepo | None, settings: Settings, *, full: bool, fetch: bool
) -> tuple[str, str]:
    """Index (or re-index) and return ``(level, message)`` to show afterwards; the sidebar redraws on completion.

    For a git URL with ``fetch`` this first clones (or updates) the repository, which is the only place the UI
    reaches the network on the user's behalf.
    """
    with st.status("Indexing...", expanded=True) as status:
        if remote is not None and fetch:
            status.update(label=f"Fetching {remote.display} (latest commit only)...")
            try:
                repo = sync_repo(remote, settings.index_dir).resolve()
            except GitSourceError as exc:
                status.update(label="Could not fetch the repository", state="error")
                return "error", str(exc)
            status.update(label="Indexing...")
        if repo is None:
            status.update(label="Nothing to index", state="error")
            return "error", "The repository has not been fetched yet."
        try:
            embedder = load_embedder(settings.embedding_model_id())
            index = RepoIndex(repo, settings.index_dir)
        except EmbeddingError as exc:
            status.update(label="Indexing failed", state="error")
            return "error", str(exc)
        except Exception as exc:
            log.exception("Unexpected error while opening the index")
            status.update(label="Indexing failed", state="error")
            return "error", f"Indexing failed ({type(exc).__name__}): {exc}"
        try:
            indexer = Indexer(
                index,
                embedder,
                limits=WalkLimits(max_file_bytes=settings.max_file_bytes, max_config_bytes=settings.max_config_bytes),
                chunk_max_lines=settings.chunk_max_lines,
                window_lines=settings.window_lines,
                window_overlap=settings.window_overlap,
                batch_size=settings.embed_batch_size,
            )

            def progress(report: IndexReport) -> None:
                status.update(label=f"Indexing... {report.files_indexed} files changed or new, {report.chunks_added} chunks")

            status.update(label="Indexing... (the first run downloads the embedding model)")
            report = indexer.run(full=full, progress=progress)
            searchable = index.keyword.count()
        except IndexMismatchError as exc:
            status.update(label="Indexing failed", state="error")
            # The message ends with the CLI's advice; here the same thing is a button.
            return "error", str(exc).replace("Re-run with --full to rebuild it.", 'Use "Rebuild from scratch" below.')
        except EmbeddingError as exc:
            status.update(label="Indexing failed", state="error")
            return "error", str(exc)
        except Exception as exc:
            log.exception("Unexpected error while indexing")
            status.update(label="Indexing failed", state="error")
            return "error", f"Indexing failed ({type(exc).__name__}): {exc}"
        finally:
            index.close()
        if searchable == 0:
            status.update(label="Nothing to index", state="error")
            return "error", f"Nothing to search: no source code or documentation was found in {repo}. Is it the right folder?"
        status.update(label="Indexing finished", state="complete", expanded=False)
    return "success", (
        f"Indexed {report.files_indexed} new or changed file(s) ({report.files_unchanged} unchanged), "
        f"{report.chunks_added} chunks, in {report.seconds:.1f}s."
    )


def _key_status(settings: Settings, provider: str) -> None:
    """Whether a key is configured, never the key itself."""
    key = {"anthropic": settings.anthropic_api_key, "openai": settings.openai_api_key}.get(provider)
    if provider == "ollama":
        st.caption(f"Ollama at {settings.ollama_host}")
    elif key is not None:
        st.caption("API key: set")
    else:
        st.caption(f"API key: not set. {_KEY_HINTS[provider]}")


def _embedding_problem(settings: Settings, status: IndexStatus) -> str | None:
    """Why semantic search cannot work on this index, or ``None``. Compares with the embedder that will be used."""
    try:
        current = load_embedder(settings.embedding_model_id()).model_id
    except EmbeddingError as exc:
        return str(exc)
    except Exception as exc:
        log.exception("Unexpected error while setting up the embedding model")
        return f"The embedding model could not be set up ({type(exc).__name__}): {exc}"
    if status.embedding_model_id == current:
        return None
    return (
        f"This index was built with `{status.embedding_model_id}` but the current setting is `{current}`. "
        "Rebuild the index from scratch, or choose keyword retrieval."
    )


def sidebar(settings: Settings) -> Config:
    with st.sidebar:
        st.header("Repository")
        raw = st.text_input(
            "Path or git URL of the repository",
            value=os.environ.get(REPO_ENV, ""),
            key="repo_path",
            help="A folder on this computer, or an https or ssh git URL. A URL is only downloaded when you click the button.",
        )
        repo: Path | None = None
        remote: RemoteRepo | None = None
        if raw.strip():
            if looks_like_url(raw):
                try:
                    remote = parse_git_url(raw)
                except GitSourceError as exc:
                    st.error(str(exc))
                else:
                    folder = clone_dir_for(remote, settings.index_dir)
                    repo = folder.resolve() if folder.is_dir() else None
            else:
                candidate = Path(raw.strip()).expanduser()
                if candidate.is_dir():
                    repo = candidate.resolve()
                else:
                    st.error("That folder does not exist.")

        status: IndexStatus | None = None
        if repo is not None or remote is not None:
            status = index_status(repo, settings.index_dir) if repo is not None else None
            if status is not None:
                label = "Update index"
            else:
                label = "Clone and index" if remote is not None and repo is None else "Index repository"
            help_text = (
                "Downloads the latest commit only, then processes new or changed files."
                if remote is not None
                else "Only new or changed files are processed."
            )
            if st.button(label, help=help_text, key="index_button"):
                st.session_state["index_message"] = _run_index(repo, remote, settings, full=False, fetch=True)
                st.rerun()  # redraw everything from the new state: button label, sizes, whether questions are allowed
            if status is not None and repo is not None:
                with st.expander("Rebuild from scratch"):
                    st.caption("Discards this index and rebuilds it. Needed after changing the embedding model.")
                    if st.button("Rebuild index", key="rebuild_button"):
                        st.session_state["index_message"] = _run_index(repo, remote, settings, full=True, fetch=False)
                        st.rerun()
            message = st.session_state.get("index_message")
            if message:
                (st.success if message[0] == "success" else st.error)(message[1])
            if status is not None:
                st.caption(f"{status.files} files, {status.chunks} chunks. Embeddings: {status.embedding_model_id}")

        st.header("Model")
        default_provider = PROVIDER_NAMES.index(settings.llm_provider)
        provider = st.selectbox("Provider", PROVIDER_NAMES, index=default_provider, key="provider")
        default_models = {
            "anthropic": settings.anthropic_model,
            "openai": settings.openai_model,
            "ollama": settings.ollama_model,
        }
        model = st.text_input("Model", value=default_models[provider], key=f"model_{provider}").strip()
        _key_status(settings, provider)
        try:
            provider_object = create_provider(settings, provider=provider, model=model or None)
        except ProviderError as exc:
            st.error(str(exc))
        else:
            if provider_object.sends_code_off_machine:
                st.warning(f"Your questions and the retrieved code are sent to {provider_object.name}.")
            else:
                st.caption("Questions and code stay on this computer.")

        st.header("Retrieval")
        mode = st.selectbox(
            "How code is found",
            MODES,
            index=MODES.index(settings.retrieval_mode),
            key="mode",
            help="vector is the default: in the M2 evaluation hybrid search did not beat it.",
        )
        if st.session_state.get("turns") and st.button("Clear conversation", key="clear_button"):
            st.session_state["turns"] = []

    blocker = None
    if repo is None and remote is not None:
        blocker = "This repository has not been downloaded yet. Click **Clone and index** in the sidebar."
    elif repo is None:
        blocker = "Enter the path or git URL of a repository in the sidebar to get started."
    elif status is None:
        blocker = "This repository is not indexed yet. Click **Index repository** in the sidebar."
    elif mode != "keyword":
        blocker = _embedding_problem(settings, status)
    return Config(repo=repo, status=status, provider=provider, model=model, mode=mode, blocker=blocker, remote=remote)


# --- page --------------------------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Codebase Q&A", page_icon="🔎", layout="wide")
    try:
        settings = get_settings()
    except ValidationError as exc:
        st.title("Codebase Q&A")
        st.error(f"The configuration is invalid; fix your .env or environment and restart.\n\n```\n{exc}\n```")
        st.stop()

    config = sidebar(settings)
    if st.session_state.get("repo_shown") != str(config.repo):  # another repository: old answers no longer apply
        st.session_state["turns"] = []
        st.session_state["repo_shown"] = str(config.repo)
    turns: list[dict] = st.session_state.setdefault("turns", [])

    st.title("Codebase Q&A")
    if config.repo is not None:
        st.caption(
            f"Asking about `{config.remote.display if config.remote else config.repo}`. A follow-up is rewritten into a standalone question using your last "
            "few exchanges; the model that answers never sees the chat itself."
        )
    for turn in turns:
        render_turn(turn)
    if config.blocker:
        st.info(config.blocker)

    question = st.chat_input("Ask about this codebase", disabled=config.blocker is not None)
    if question and question.strip():
        user_turn = {"role": "user", "text": question.strip(), "error": None, "answer": None}
        earlier = history_of(turns)
        turns.append(user_turn)
        render_turn(user_turn)
        turns.append(ask_and_render(user_turn["text"], config, settings, earlier))


if __name__ == "__main__":  # Streamlit runs the script as __main__; importing the module (tests) does not
    main()
