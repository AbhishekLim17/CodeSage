# AI Codebase Understanding System

[![CI](https://github.com/AbhishekLim17/CodeSage/actions/workflows/ci.yml/badge.svg)](https://github.com/AbhishekLim17/CodeSage/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Ask questions about a codebase in plain English and get explanations that cite the exact files and line ranges they came from.

It combines NLP (identifier-aware search, follow-up rewriting), Large Language Models (Claude, OpenAI or a local Ollama model write the answer) and Retrieval-Augmented Generation (the answer is built only from code retrieved from your repository, and every citation is checked against what was retrieved). The goal is to shorten code comprehension and developer onboarding.

## Demo

Everything below is real output from running the tool on [`psf/requests`](https://github.com/psf/requests) (a shallow clone, 7.6 MB), on a laptop CPU with no GPU. The transcripts are lightly condensed (table borders removed, long paths shortened, the map abridged); no numbers were changed.

**Index a repository from its URL.** The clone is shallow (latest commit only), has no git hooks, and lives under the index folder; nothing is written into any repository you point at by path.

```text
$ codebase-ai index https://github.com/psf/requests
Fetching https://github.com/psf/requests (shallow clone)...
                         Indexed requests-4128e517
  Files seen                      130
  Files indexed (new / changed)   93 (93 / 0)
  Chunks added / removed          786 / 0
  Time                            100.7s
                                   Skipped
  unsupported_type   29   .coveragerc, .git-blame-ignore-revs, LICENSE
  secret_file         7   tests/certs/expired/ca/ca-private.key, .../server.key, .../server.pem
  denied_dir          1   .git
  empty               1   tests/testserver/__init__.py
```

The second run took 17 seconds and re-embedded nothing: unchanged files are skipped by content hash. The private-key test fixtures were kept out of the index as secrets.

**Find the code that answers a question.** Sources are exact line ranges, best first:

```text
$ codebase-ai search https://github.com/psf/requests "how are redirects followed and how is the redirect limit enforced?" -k 5
 1. HISTORY.md:441-560  [doc_section]
      - Redirect resolution should now only occur when
 2. src/requests/sessions.py:186-245  [method] SessionRedirectMixin.resolve_redirects
      def resolve_redirects(
 3. src/requests/sessions.py:127-130  [context] SessionRedirectMixin
      class SessionRedirectMixin:
 4. src/requests/sessions.py:286-307  [method] SessionRedirectMixin.resolve_redirects
      # Override the original request.
 5. src/requests/auth.py:268-271  [method] HTTPDigestAuth.handle_redirect
      def handle_redirect(self, r: Response, **kwargs: Any) -> None:
5 of 16 sources, about 3548 tokens (vector)
```

The right code is found, but notice result 1: the changelog ranks above the implementation. `CHANGELOG_PENALTY` can demote release notes, but on 109 labelled questions it changed one question by one rank, so it is off (see [Limitations](#limitations)).

**Whole-project questions get a map.** For "what does this project do?", the model is also given a generated map of the repository, built from the index with no model call, and the top of the README. This is the map for the same repository, abridged:

```text
Repository map (generated from the index; it is not a source file).
93 files: python 36, rst 16, markdown 13, yaml 12, makefile 8, gitignore 2.
Files at the top level: README.md, AUTHORS.rst, HISTORY.md, Makefile, pyproject.toml, ...

docs/  (22 files, rst, python)
  docs/community/  (7 files, rst)
  docs/dev/  (2 files, rst)
  docs/user/  (4 files, rst)
src/requests/  (19 files, python) - Requests HTTP Library  defines Request, MockRequest, CaseInsensitiveDict, Session
Tests: tests/ (23 files).

Documentation files (title or first heading):
  docs/api.rst - Developer Interface
  docs/index.rst - Requests: HTTP for Humans™
Code files that describe themselves (first line of their header comment or docstring):
  src/requests/adapters.py - This module contains the transport adapters that Requests uses to define
  src/requests/auth.py - This module contains the authentication handlers for Requests.
  ...
```

**Ask.** A real answer, produced fully offline by `qwen2.5-coder:7b` through Ollama on a laptop with a 4 GB GPU (unedited, except that the list of uncited sources is shortened):

```text
$ codebase-ai ask https://github.com/psf/requests "How does a redirect get followed, and what limits how many redirects are followed?" --provider ollama
A redirect is followed by the `resolve_redirects` method in the `SessionRedirectMixin` class [5]. This method is called
by the `send` method of the `Session` class when a redirect is detected [15].

The `resolve_redirects` method follows redirects by repeatedly calling `get_redirect_target` to get the new URL from the
`Location` header of the response [1]. It then creates a new `PreparedRequest` with the new URL and sends it back to the
server [6].

The number of redirects that can be followed is limited by the `max_redirects` attribute of the `SessionRedirectMixin`
class [2]. If the number of redirects exceeds this limit, a `TooManyRedirects` exception is raised [3].

This limit is checked in the `resolve_redirects` method before each redirect is followed [5]. If the limit is reached,
the method raises the `TooManyRedirects` exception [5].

Sources cited
  [5] src/requests/sessions.py:186-245  method SessionRedirectMixin.resolve_redirects
  [15] tests/test_lowlevel.py:308-339  function test_redirect_rfc1808_to_non_ascii_location
  [1] src/requests/sessions.py:134-152  method SessionRedirectMixin.get_redirect_target
  [6] src/requests/sessions.py:286-307  method SessionRedirectMixin.resolve_redirects
  [2] src/requests/sessions.py:127-130  class header SessionRedirectMixin
  [3] src/requests/exceptions.py:106-107  class TooManyRedirects

Also retrieved (not cited)
  [4] HISTORY.md:441-560  doc_section
  [7] src/requests/models.py:883-889  method Response.is_permanent_redirect
  ... 8 more

ollama qwen2.5-coder:7b, 3947 tokens in / 198 out; retrieval 9.9s, answer 98.1s
```

**Read it the way you should read any answer here.** The mechanism is right and every location is real. But citation `[15]` is a test, not `Session.send`, so it does not support its sentence; `[2]` is a four-line class header that may not show `max_redirects`; and the default limit (30) is not mentioned. Every `[n]` is checked against what was supplied, not against what the sentence claims: that is what clicking through to the cited code is for.

`ask` prints a warning when an answer cites nothing, cites a source that does not exist (shown as `[?]`), mentions a file or line range it was never shown, was cut off, or was declined. With a hosted provider it first says that code is about to be sent and where. `codebase-ai serve` is the same thing as a chat page that streams the answer and shows the cited code beside it.

## Status

| Milestone | State |
|---|---|
| **M0** skeleton, config, CLI | done |
| **M1** ingest, chunk, index; embedding-model benchmark | done |
| **M2** retrieval, token budget, evaluation harness | done |
| **M3** cited answers, LLM providers, Streamlit UI | done |
| **M4** quality: repo map, follow-ups, reranker, answer-quality harness | done; answer quality measured with a local model, follow-ups still untested live |
| **M5** polish: git URLs, demo README, final report | done |

The design is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The evidence behind every default, what worked, what did not and what was never measured, is in [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md).

**What was and was not tested.** About 980 automated tests pass, on Ubuntu (Python 3.11 and 3.13) and Windows (3.13) in CI, offline and without an API key. Retrieval, indexing, the repository map and git cloning have been run for real, and so has answering: **30 questions were answered fully offline by `qwen2.5-coder:7b` through Ollama** (results in [`docs/QUALITY_EVAL.md`](docs/QUALITY_EVAL.md) section 3). **Claude and OpenAI have not been run live**: their adapters are tested against the real vendor SDKs over mocked HTTP only, and follow-up rewriting has only been tested with scripted models. If a provider misbehaves, the error says which one and what kind of failure it was.

## Install

Requires Python 3.11+ and, for git URLs, `git`. The first `index` run downloads the embedding model from Hugging Face (about 130 MB for the default); it then runs on your machine.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[anthropic]"        # or ".[openai]", ".[ollama]"; ".[dev]" for all three plus the test tools
copy .env.example .env               # optional: everything has a default
```

## Use it

```bash
codebase-ai index path/to/repo                       # or a git URL: https://github.com/owner/repo
codebase-ai index https://github.com/owner/repo --branch main
codebase-ai stats path/to/repo
codebase-ai search path/to/repo "how are emails queued?"            # no model needed
codebase-ai ask path/to/repo "how are emails queued?"               # needs an LLM (below)
codebase-ai ask path/to/repo "..." --provider ollama --model qwen2.5-coder:7b
codebase-ai serve path/to/repo                       # chat UI at http://localhost:8501 (this computer only)
codebase-ai chunks path/to/repo/src/app.py --text    # see exactly how a file is split
```

`index` is the only command that touches the network for a git URL; `search`, `ask`, `stats` and `serve` use the clone `index` made (the chat UI's **Clone and index** button is the one other place, and only when you click it). `python -m codebase_ai` is the same as `codebase-ai`.

**Giving it a model.** Put a key in `.env` (or use Ollama, which needs none):

```
LLM_PROVIDER=anthropic              # anthropic | openai | ollama
ANTHROPIC_API_KEY=...               # or OPENAI_API_KEY; Ollama needs `ollama serve` and `ollama pull <model>`
```

The repository must be indexed first. Each answer is built from the retrieved code only; earlier chat turns are used just to rewrite a follow-up ("and what about retries?") into a standalone question, and the model that answers never sees the chat.

Indexes and clones live in `~/.codebase_ai/` (set `INDEX_DIR` to change it).

## What it does

- **Code-aware chunking** (tree-sitter): one chunk per function, class, method or type, with symbol names and exact line ranges. Python, JavaScript/JSX, TypeScript/TSX, Java and Go; Markdown by heading; everything else in overlapping line windows.
- **Safe ingestion:** honours nested `.gitignore` files, never follows symlinks, skips binaries, lockfiles and minified code, and keeps secret files and token-shaped strings out of the index. Says so when a folder has nothing to search.
- **Git URLs** (`https` and `ssh` only): shallow clone, no hooks, no submodules, no LFS, never prompts for a password, refuses URLs that contain credentials, and never deletes anything.
- **Incremental, crash-safe indexing** by file hash; refuses to mix embeddings from different models.
- **Retrieval you can cite:** vector search by default, test files demoted, merged into exact `path:start-end` sources that fit a token budget. Keyword and hybrid modes exist; the evaluation found neither beat plain vector search. `search` lists exactly what `ask` would give the model.
- **Answers you can check:** the model is told to answer only from numbered sources and cite them; every `[n]` is validated against what was retrieved, invented citations are shown as `[?]` and reported, and a `path:line` the model was never shown is flagged. A valid citation means the source was supplied, not that it proves the claim. If the model cannot be reached (no key, Ollama not running), you still get the code it would have answered from.
- **Provider-agnostic:** Claude, OpenAI or Ollama behind one streaming interface, with errors that never contain your key and a note before any code is sent to a hosted provider. Ollama on another machine counts as hosted.
- **Whole-project questions** get a generated repository map and the README; **follow-ups** are rewritten into standalone questions (and the rewrite is shown).
- **Streamlit chat** with streaming, cited code beside each answer, indexing progress, and an explicit state for every failure.
- **Measured, not assumed:** an evaluation harness (Hit@k, MRR, in-context recall, paired significance tests) over three repositories drives the defaults, including the ones that were decided *against* (hybrid search, a reranker).
- **Optional reranker**, off by default because the one tried made retrieval worse.

## Evidence in one table

| Question | Result | Where |
|---|---|---|
| Which embedding model? | `bge-small-en-v1.5`, within noise of larger ones | [`docs/EMBEDDING_BENCHMARK.md`](docs/EMBEDDING_BENCHMARK.md) |
| Does hybrid (keyword + vector) search beat vector? | No: -0.034 MRR, interval [-0.095, +0.026], 85 questions | [`docs/RETRIEVAL_EVAL.md`](docs/RETRIEVAL_EVAL.md) |
| Does demoting test files help? | Yes: +0.032 MRR, better on 10 questions, worse on none | same |
| Does a repository map help whole-project questions? | Areas shown rose from 63% to 93% on 11 questions; recognising such questions was the weak part (2 of 20 fresh phrasings at first, 18 of 20 after widening, which is not independent) | [`docs/QUALITY_EVAL.md`](docs/QUALITY_EVAL.md) |
| Does a cross-encoder reranker help? | No: -0.037 MRR, about 2 s slower per query (one small model tried) | same |
| Does demoting changelogs help? | No measurable effect (1 of 109 questions moved by one rank); off | same |
| Does it work on code it was never tuned on? | Yes: 36 questions written blind on `httpx` and `jinja2`, MRR 0.915 and 0.801, the right file in context every time | same |
| Does it scale? | Linear up to 4,000 files / 28,000 chunks (63 s, 85 MB, 10 ms search) without the embedding model; the real model needs about an hour for that size on a CPU (extrapolated) | same |
| Are the answers right, and do they cite their code? | Measured with a local 7B model, offline: no invented citation or location in 30 answers; a one-line reminder raised answers with citations from 40% to 100% on held-out questions; hand-graded, 14 of 20 correct, 4 partly, 2 wrong (both faithful to plausible but wrong code) | same |

## Limitations

- **Answer quality is measured on 30 questions with one small local model only.** Claude and OpenAI were not run live, follow-up rewriting has never met a real model, and the correctness grades were given by the AI assistant that built the system, not by an independent reader. A wrong answer can still be faithful: if retrieval supplies dead or unrelated code, the model describes it accurately (seen twice in 20).
- **Prose can outrank code.** On `psf/requests` the changelog ranked above `sessions.py` for a question about redirects. Demoting release notes made no measurable difference over 24 `requests` questions and 85 others, so it is off; long guides can still outrank code, which is sometimes the right answer.
- **Small, partly self-written question sets.** 85 retrieval questions and 11 overview questions over three repositories; two were labelled by the author of the system. Treat the direction of each result as credible and the exact percentages as rough.
- **Languages.** Tree-sitter chunking covers Python, JavaScript/TypeScript, Java and Go. Other languages are indexed in line windows, which works but cuts at arbitrary points.
- **Overview recognition is pattern based.** On fresh phrasings written before the patterns were widened it recognised only 2 of 20. After widening it recognises 18 of 20 of the same items, but those were written by the same person who wrote the patterns, so measure it on your own phrasing with `eval/check_classifier.py`. A miss just gets ordinary retrieval.
- **Git URLs:** `https` and `ssh` only, public repositories or ones your own git credentials can already reach. `index --branch` selects a branch on the first clone and on updates.

## License

MIT; see [`LICENSE`](LICENSE).

## Privacy

With a cloud LLM or cloud embeddings, retrieved code snippets are sent to that provider; `ask` and the chat UI say so before sending anything. For fully on-device operation use Ollama on this machine with local embeddings. The default (local embeddings) sends nothing anywhere except the one-time model download, and git clones talk only to the server you named. Secret files (`.env*`, keys, service-account JSON) and files containing well-known token formats are excluded from indexing; this is a best-effort safety net, so review `codebase-ai index` output (the "Skipped" table) for repositories with sensitive material. `serve` listens on localhost only, because the UI can index any folder you name.

## Development

```bash
pip install -e ".[dev]"
pytest                               # about 960 tests; needs no network and no API key
ruff check src tests eval
```

```
src/codebase_ai/   library: ingest, chunking, index, retrieval, llm, rag, ui, evaluation
tests/             unit and component tests, plus a 6-language fixture repo
eval/              benchmark and evaluation scripts, and the labelled question sets
docs/              architecture, and the evidence: benchmark, retrieval, quality, final report
```

Evaluation scripts (`eval/README.md` explains the question files): `bench_embeddings.py`, `run_eval.py` (retrieval), `run_overview_eval.py`, `run_rerank_eval.py`, and `run_answers.py` (answer quality with a judge; costs money with a hosted provider and refuses to start without `--yes`).
