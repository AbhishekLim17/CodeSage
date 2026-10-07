# Final report

This is the closing summary of the project: what was built, what the evidence says about each decision, what was
never verified, and how to reproduce every number. It is written to be read without the rest of the documentation;
the details live in `ARCHITECTURE.md`, `EMBEDDING_BENCHMARK.md`, `RETRIEVAL_EVAL.md` and `QUALITY_EVAL.md`.

## 1. What was built

A tool that answers natural-language questions about a codebase with citations to files and line ranges.

```
repository (path or git URL) -> walk (.gitignore, secrets, binaries) -> tree-sitter chunks -> embeddings
   -> Chroma vectors + SQLite keyword index            (indexed once, updated by file hash)

question -> [follow-up rewritten to stand alone] -> vector search, test files demoted -> token-budgeted,
   merged sources [+ repository map and README for whole-project questions]
   -> Claude / OpenAI / Ollama, streaming -> every [n] checked against the sources -> answer + cited code
```

About 6,000 lines of library code in 42 modules, 7,000 lines of tests (about 960 tests that need no network and no
API key), and 1,100 lines of evaluation scripts, plus a Typer CLI (`index`, `stats`, `search`, `ask`, `serve`, `chunks`,
`eval`) and a Streamlit chat.

| Milestone | Delivered |
|---|---|
| M0 | Package, typed settings from `.env`, CLI skeleton |
| M1 | Safe ingestion, AST chunking (Python, JS/TS, Java, Go; Markdown by heading; line windows otherwise), incremental crash-safe indexing, embedding-model benchmark |
| M2 | Retriever with token budget, class context and merged exact sources; evaluation harness with paired statistics |
| M3 | Three provider adapters behind one streaming interface, cited answers with a citation validator, CLI `ask` and `serve`, Streamlit app |
| M4 | Overview classifier and repository map, follow-up rewriting, optional reranker, answer-quality harness with a faithfulness judge |
| M5 | Git URL support, demo README, this report, MIT license, CI (Linux and Windows), classifier rebuilt by intent and measured on held-out phrasing, release-note demotion evaluated, scale test |

## 2. Decisions and the evidence behind them

| Decision | Evidence | Verdict |
|---|---|---|
| Default embedding model `bge-small-en-v1.5` | Benchmark of several models on labelled questions (`EMBEDDING_BENCHMARK.md`): larger models were within noise | Kept: smallest that was not measurably worse |
| **Vector search, not hybrid** (reverses the original design) | 85 questions, 3 repositories: hybrid vs vector MRR -0.034, interval [-0.095, +0.026]; keyword alone -0.178 (significantly worse) | Hybrid stays available, not the default |
| **Demote test files** (x0.5 unless the query is about tests) | MRR +0.032, interval [+0.011, +0.057]; better on 10 questions, worse on none; concentrated in the repository with a large test suite | On by default |
| Repository map + README for whole-project questions | 11 overview questions: areas shown 63% -> 93%, at about 1,700 tokens. Recognising them is the weak part: 2 of 20 fresh phrasings at first, 18 of 20 after widening (not independent); 0 of 109 specific questions misclassified | On by default |
| Demote release notes (CHANGELOG, HISTORY) | 24 `requests` questions + 85 others: 108 of 109 tied, one moved by a single rank | Off (`CHANGELOG_PENALTY`) |
| **Reranker off** | One cross-encoder tried: MRR -0.037, interval [-0.095, +0.023], 9 questions better and 17 worse, roughly 2 s slower per query on a CPU | Code kept, off by default |
| **Citation reminder at the end of the prompt** | A local 7B model left a third of its answers uncited; on 10 held-out questions the reminder raised grounded answers from 40% to 100% (better on 6, worse on 0) with no measurable change in correctness | On |
| Follow-ups rewritten into standalone questions | Tests with scripted models only | On by default; **unmeasured** |
| Deterministic repository map, not model-written summaries | Free, cannot be wrong about what a directory holds | Revisit only if overview answers turn out weak |
| `tree-sitter` pinned below 0.26 | 0.26.0 returned garbage node positions and crashed the interpreter on ordinary files | Pin is the protection; a native crash cannot be caught |

Two of these reversed the plan that preceded them (hybrid, reranker), and one earlier hint did not survive a re-run
(that hybrid finds the gold *function* less often): all are recorded where they happened rather than edited out.

## 3. The ledger: measured, tested with stand-ins, still open

**Measured on real data (real embedding model, real repositories):** embedding choice; vector vs keyword vs hybrid;
test demotion; token budget and in-context recall; repository map coverage; the reranker; retrieval on two libraries
it had never seen.

**Measured with a real language model, offline** (2026-10-06, `qwen2.5-coder:7b` answering and `gemma3:4b` judging,
both through Ollama; `QUALITY_EVAL.md` section 3): 30 answers. No citation pointed at a source that was not supplied,
and no answer mentioned a location it had not been shown. With the citation reminder every answer cited its sources.
Graded by hand against the code, 14 of the first 20 answers were correct, 4 partly correct and 2 wrong; both wrong
answers were faithful to plausible but wrong code (one of them dead code), which no check here can catch. The 4B judge
called all 33 answers it judged "supported", so it cannot tell good from bad, and no faithfulness number is quoted.

**Tested, but only with stand-ins:**

- The Claude and OpenAI adapters, against the real vendor SDKs over mocked HTTP. Request shapes, streaming, usage, stop
  reasons, refusals, retries and error mapping are checked; nothing was sent to Claude or OpenAI. (The Ollama adapter has
  now been run for real.)
- Follow-up rewriting.

**Still open:** a hand check of the judge (10 answers with their cited code are in `eval/results/`), a stronger
judge, a hosted model, and correctness grades from someone other than the system's builder. The honest claim is now
"retrieval is measured; answers are measured on a small set with one small local model".

## 4. What real use turned up

Running on real repositories and real data found problems that unit tests written in advance had not:

| Found by | Problem | Fix |
|---|---|---|
| Indexing a real JS repository | tree-sitter 0.26.0 crashed the interpreter | Version pin; fallback for impossible positions |
| Printing source lines | Emoji in source raised `UnicodeEncodeError` on a cp1252 console | Output made encoding-safe, regression-tested |
| Probing the Anthropic adapter with no key | The SDK's bare `TypeError` reached the user | Translated to an `auth` error that says to set the key |
| Reading the provider code for remote hosts | Ollama on another machine was reported as staying on this computer | Locality is decided from the host |
| The real UI against a real index | Stale button label after indexing; log flooded with tracebacks from a file watcher | Page redraws from fresh state; watcher off in `serve` |
| Running the repository map on `psf/requests`, which it had not been tuned on | `requirements.txt` listed as documentation; `.rst` titles missed; modules whose docstring begins with their own name described as that name | Fixed, with regression tests |
| Writing held-out questions for the overview classifier | The 64% recall reported at first was optimistic: 2 of 20 fresh phrasings were recognised | Patterns rebuilt by intent; both numbers are reported, and the 18 of 20 afterwards is labelled as not independent |
| The first CI run, on Linux for the first time | One test of about 960 failed in all three environments: typer forces colour codes into help text on GitHub Actions, which landed mid-sentence | The test strips them and the test environment clears colour-forcing variables; CI now also reports failing tests as public annotations |
| The first answers from a real model (local, 2026-10-06) | A third cited nothing; the model named files in prose instead | A citation reminder at the end of the prompt: 40% to 100% grounded on held-out questions |
| Grading those answers by hand | The exported sample said "cited [14]" but not what [14] was, so it could not be graded | Each result keeps its cited code, and the sample shows it |
| Planning that run | The plan promised 16,000 output tokens from Ollama, which caps answers at 4,096 | The plan states the cap the provider really applies |
| Re-indexing MagnaFlow after it grew 74% (review of 2026-10-06) | Its main design document, `CLAUDE.md`, was skipped as "minified": one Markdown line was over 2,000 characters | The long-line rule now applies to code and config only; regression-tested, and re-indexing picked the file up |
| Debugging an overview question with `search` | `search` left out the repository map that `ask` adds, so it showed a different context from what the model gets | Both go through one retriever factory; tested |
| Feeding the citation checker bad output | Deleting an invalid `[42]` left broken prose ("is enforced in [2][3] and in.") | It shows as `[?]`, and simply goes when it sits next to a valid citation |
| Using the CLI without a key | A missing key gave only an error, although the search had already found the code | `ask` and the UI list the code it would have answered from |
| Indexing an empty or wrong folder | Reported success with 0 files | Says there is nothing to search and exits with 1 |
| Reading CLI output | Model-loading progress bars appeared mid-output; "Fetching..." was printed before a bad branch name was refused; Streamlit showed a Deploy button and promotions | Bars switched off, branch checked first, Streamlit chrome hidden |
| Capturing `--help` for the README | `[n]` in a help string was eaten as markup | Reworded, regression-tested |
| The full test suite, once it passed 500 tests | "Too many open files": closing an index left Chroma's handles open, which also blocks deleting the index folder on Windows | `close()` now releases them; reproduced and tested |

## 5. Threats to validity

- **Small question sets.** 85 retrieval questions and 11 overview questions over three repositories. One question moves a
  percentage by 1 to 4 points. Intervals are given; treat them as guides, not probabilities.
- **Labels written by the author.** Two of the three retrieval sets and both overview sets were labelled by the person
  who built the system. Only the `rich` set is independent code, and it has no prose and no tests.
- **One repository is this project**, which changes as it is built: the self-evaluation corpus grew from 74 to 93 files
  between the first and the final run, and its numbers moved with it. It is re-run for every report.
- **One embedding model and one reranker** were used in the comparisons that decided against hybrid search and the
  reranker. Other models might behave differently.
- **Coverage is generous by construction.** The overview metric counts a directory as shown when the map names it.
- **Classifier recall was measured once**, on questions written before the final patterns, and deliberately not tuned on
  them. It needs a fresh set to be measured again fairly.
- **No rare exact-string queries** (error messages, config keys), the case where keyword search traditionally helps.
- **Pooled statistics** treat questions as independent and mix repositories of different sizes.

## 6. Known limitations

- Prose can outrank code. On `psf/requests` the changelog ranked above `sessions.py` for a question about redirects. Demoting
  release notes was evaluated and made no measurable difference, so it is off; long guides can still outrank code, which is
  sometimes the right answer.
- Tree-sitter chunking covers four language families (Python, JavaScript/TypeScript, Java, Go); everything else is indexed in line windows.
- Overview recognition is pattern based. It recognised 2 of 20 fresh phrasings before it was widened and 18 of 20 after, but the
  widening was done by the same person who wrote those items, so that is a regression guard, not independent evidence.
- Git URLs: `https` and `ssh` only; private repositories only through the user's own git credentials.
- Scale was checked only up to 4,000 synthetic files (28,000 chunks) and without the real embedding model, whose cost was
  extrapolated from one measured run (about an hour for 4,000 files on a CPU).

## 7. Reproducing everything

| To reproduce | Command | Needs |
|---|---|---|
| The test suite | `pytest` | Nothing (no network, no key) |
| Embedding benchmark (M1) | `python eval/bench_embeddings.py ...` | The models it compares, downloaded once |
| Retrieval evaluation (M2) | `python eval/run_eval.py --suite ... --out-md docs/RETRIEVAL_EVAL.md` | The embedding model; the three repositories |
| Overview evaluation | `python eval/run_overview_eval.py --suite ... --others ...` | Same |
| Classifier on your own phrasing | `python eval/check_classifier.py my_questions.json` | Nothing |
| Release-note demotion | `python eval/run_changelog_eval.py --suite requests ... ` | The embedding model; a clone of `psf/requests` |
| Scale | `python eval/scale_test.py --sizes 200,1000,4000` | Nothing (synthetic repositories) |
| Reranker ablation | `python eval/run_rerank_eval.py --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --suite ...` | The reranker model (about 90 MB) |
| Answer quality | `python eval/run_answers.py REPO QUESTIONS --provider ... --judge-provider ... --yes` | A model and your key; costs money with a hosted provider |

The exact command lines for each are in the docstring of the script and in `eval/README.md`. The raw per-question results
of every run are written to `eval/results/` (not committed).

## 8. What to do next, in order of value

1. **Grade the 10 exported answers by hand** (`eval/results/answers_*_sample.md`, each with its cited code) and run
   `--check-sample`; then re-run with a stronger judge, since the 4B one called everything "supported". A hosted model
   (Claude, OpenAI) is the other half: its adapters have never sent a real request.
2. **Measure the overview classifier on questions you write**: `python eval/check_classifier.py my_questions.json` takes seconds,
   and it is the only genuinely independent measurement left for that component.
3. **Try other rerankers** (larger or code-tuned) with `eval/run_rerank_eval.py`; nothing else needs to change.
4. **Index a really large repository** with the real model to replace the extrapolation, and look at memory use.
5. Tree-sitter grammars for more languages, and model-written directory summaries only if overview answers prove weak.
