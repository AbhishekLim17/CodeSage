# AI Codebase Understanding System

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

The right code is found, but notice result 1: the changelog ranks above the implementation. On repositories with a lot of prose documentation, prose can outrank code (see [Limitations](#limitations)).

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

**Ask.** `ask` needs a language model, so this README cannot show a real answer: no model was run while it was written. This is the *layout* of what `ask` prints (the answer text is deliberately left out):

```text
$ codebase-ai ask https://github.com/psf/requests "How does a redirect get followed?"
Note: your question and <n> retrieved code excerpt(s) are sent to <provider> (<model>).
<the answer streams here, citing the sources it used as [1], [2], ...>

Sources cited
  [1] <path>:<first line>-<last line>  <what it is>
Also retrieved (not cited)
  [2] ...
<provider> <model>, <n> tokens in / <n> out; retrieval <s>s, answer <s>s
```

It also prints a warning when an answer cites nothing, cites a source that does not exist (that citation is removed), mentions a file or line range it was never shown, was cut off, or was declined. `codebase-ai serve` is the same thing as a chat page that streams the answer and shows the cited code beside it.

## Status

| Milestone | State |
|---|---|
| **M0** skeleton, config, CLI | done |
| **M1** ingest, chunk, index; embedding-model benchmark | done |
| **M2** retrieval, token budget, evaluation harness | done |
| **M3** cited answers, LLM providers, Streamlit UI | done |
| **M4** quality: repo map, follow-ups, reranker, answer-quality harness | done, with caveats |
| **M5** polish: git URLs, demo README, final report | done |

The design is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The evidence behind every default, what worked, what did not and what was never measured, is in [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md).

**What was and was not tested.** About 900 automated tests pass. The three LLM provider adapters are tested against the real vendor SDKs over mocked HTTP, and the app is tested headless and by hand in a browser against real indexes. **No language model was ever run against this system**: no API key was used and no local model is installed. So the answers themselves, follow-up rewriting, and the answer-quality evaluation with its faithfulness judge are built and tested with scripted stand-ins only. Retrieval, indexing, the repository map and git cloning have been run for real. To see real answers you need your own key (below); if a provider misbehaves, the error says which one and what kind of failure it was.

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
- **Safe ingestion:** honours nested `.gitignore` files, never follows symlinks, skips binaries, lockfiles and minified files, and keeps secret files and token-shaped strings out of the index.
- **Git URLs** (`https` and `ssh` only): shallow clone, no hooks, no submodules, no LFS, never prompts for a password, refuses URLs that contain credentials, and never deletes anything.
- **Incremental, crash-safe indexing** by file hash; refuses to mix embeddings from different models.
- **Retrieval you can cite:** vector search by default, test files demoted, merged into exact `path:start-end` sources that fit a token budget. Keyword and hybrid modes exist; the evaluation found neither beat plain vector search.
- **Answers you can check:** the model is told to answer only from numbered sources and cite them; every `[n]` is validated against what was retrieved, invented citations are removed and reported, and a `path:line` the model was never shown is flagged. A valid citation means the source was supplied, not that it proves the claim.
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
| Does a repository map help whole-project questions? | Areas shown rose from 63% to 90% on 11 questions; the classifier recognises 64% of such questions | [`docs/QUALITY_EVAL.md`](docs/QUALITY_EVAL.md) |
| Does a cross-encoder reranker help? | No: -0.037 MRR, about 2 s slower per query (one small model tried) | same |
| Are the answers faithful to the cited code? | **Not measured**: needs a live model; the harness is built | same |

## Limitations

- **No live-model verification** (above). Answer quality, follow-up rewriting and the judge are untested against a real model.
- **Prose can outrank code.** On `psf/requests` the changelog and long guides ranked above `sessions.py` for a question about redirects. Test files are demoted; documentation is not. No fix was added without an evaluation, and this one has not been evaluated.
- **Small, partly self-written question sets.** 85 retrieval questions and 11 overview questions over three repositories; two were labelled by the author of the system. Treat the direction of each result as credible and the exact percentages as rough.
- **Languages.** Tree-sitter chunking covers Python, JavaScript/TypeScript, Java and Go. Other languages are indexed in line windows, which works but cuts at arbitrary points.
- **Overview recognition is pattern based** and misses about a third of the phrasings tried; a missed one just gets ordinary retrieval.
- **Git URLs:** `https` and `ssh` only, public repositories or ones your own git credentials can already reach. `index --branch` selects a branch on the first clone and on updates.
- **No license file yet.** One has to be chosen by whoever owns the code before it is shared.

## Privacy

With a cloud LLM or cloud embeddings, retrieved code snippets are sent to that provider; `ask` and the chat UI say so before sending anything. For fully on-device operation use Ollama on this machine with local embeddings. The default (local embeddings) sends nothing anywhere except the one-time model download, and git clones talk only to the server you named. Secret files (`.env*`, keys, service-account JSON) and files containing well-known token formats are excluded from indexing; this is a best-effort safety net, so review `codebase-ai index` output (the "Skipped" table) for repositories with sensitive material. `serve` listens on localhost only, because the UI can index any folder you name.

## Development

```bash
pip install -e ".[dev]"
pytest                               # about 900 tests; needs no network and no API key
ruff check src tests eval
```

```
src/codebase_ai/   library: ingest, chunking, index, retrieval, llm, rag, ui, evaluation
tests/             unit and component tests, plus a 6-language fixture repo
eval/              benchmark and evaluation scripts, and the labelled question sets
docs/              architecture, and the evidence: benchmark, retrieval, quality, final report
```

Evaluation scripts (`eval/README.md` explains the question files): `bench_embeddings.py`, `run_eval.py` (retrieval), `run_overview_eval.py`, `run_rerank_eval.py`, and `run_answers.py` (answer quality with a judge; costs money with a hosted provider and refuses to start without `--yes`).
