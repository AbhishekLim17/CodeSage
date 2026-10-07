# Evaluation

Retrieval quality is measured on labelled questions, not guessed. Question files live in `questions/`, one per
repository, as JSONL:

```json
{"id": "q01", "type": "explain", "question": "How does login refuse suspended organizations?",
 "gold_files": ["src/contexts/AuthContext.jsx"],
 "acceptable_files": ["CLAUDE.md"],
 "gold_symbols": ["orgIsSuspended"]}
```

- `gold_files`: the code that is the answer; any one counts. **Strict** metrics use only these.
- `acceptable_files` (optional): prose (README, design notes) that also legitimately answers the question.
  **Lenient** metrics accept gold or acceptable. Both views are always reported.
- `gold_symbols` (optional): functions/classes whose code should be in the retrieved context.
- `gold_dirs` (optional, overview questions): directories an answer to a whole-repository question needs; scored as *area coverage*.
- `type`: `locate`, `explain` or `doc`, or `overview` in the `*_overview.jsonl` files. Paths are relative to the repo root, with forward slashes.

| File | Repository | Language | Questions | Labelled by |
|---|---|---|---|---|
| `questions/magnaflow.jsonl` | MagnaFlow (React/Firebase app) | JavaScript | 36 | the person building this tool, who had read the repo's overview documents |
| `questions/codebase_ai.jsonl` | this project | Python | 26 | the author of the code |
| `questions/rich.jsonl` | the `rich` library, as installed in `.venv` | Python | 23 | independent code, not written by us |
| `questions/httpx.jsonl` | `httpx` 0.28.1, as installed in `.venv` | Python | 18 | written blind, before the library was ever indexed (2026-10-06) |
| `questions/jinja2.jsonl` | `jinja2` 3.1.6, as installed in `.venv` | Python | 18 | written blind, before the library was ever indexed (2026-10-06) |

For `magnaflow.jsonl`, `acceptable_files` follows a **mechanical rule** applied after the first results and stated up
front: a prose file counts only if it names the specific code artifact (file, function or setting) the question is
about; no per-question judgement, and the strict numbers are unchanged. For `codebase_ai.jsonl` the author judged by
reading the docs. `rich.jsonl` has none because the installed package contains no prose.

## Retrieval evaluation (M2)

```bash
# one repo, from the CLI (the repo must be indexed first)
codebase-ai eval path/to/repo --questions eval/questions/magnaflow.jsonl

# several repos at once, with a pooled table and a markdown report
python eval/run_eval.py \
    --suite magnaflow path/to/MagnaFlow eval/questions/magnaflow.jsonl \
    --suite codebase_ai . eval/questions/codebase_ai.jsonl \
    --suite rich .venv/Lib/site-packages/rich eval/questions/rich.jsonl \
    --out-md docs/RETRIEVAL_EVAL.md
```

It compares `vector`, `keyword` and `hybrid` (plus hybrid with other keyword weights) and reports Hit@k, MRR, lenient
scores, in-context recall, symbol recall, mean context tokens and latency. Results: `docs/RETRIEVAL_EVAL.md`; raw
per-question output goes to `eval/results/` (gitignored). The `## Reading the numbers` section of the report is
written by hand and kept when the tables are regenerated.

## Embedding benchmark (M1)

`bench_embeddings.py` compares embedding models on a question file (default: `questions/magnaflow.jsonl`), using
vector search only:

```bash
python eval/bench_embeddings.py path/to/repo --out-md docs/EMBEDDING_BENCHMARK.md
python eval/bench_embeddings.py path/to/repo --models BAAI/bge-small-en-v1.5,intfloat/e5-small-v2
```

## Overview questions and the repository map (M4)

`questions/magnaflow_overview.jsonl` and `questions/codebase_ai_overview.jsonl` hold questions about the whole
repository ("what does this project do?"). Their `gold_dirs` are the areas a good answer needs; the score is *area
coverage* (how many of them the context shows, as code or as a line in the repository map), with and without the map.
They were written before the classifier was run on them, and the classifier was not tuned on them.

```bash
python eval/run_overview_eval.py \
    --suite magnaflow path/to/MagnaFlow eval/questions/magnaflow_overview.jsonl \
    --suite codebase_ai . eval/questions/codebase_ai_overview.jsonl \
    --others eval/questions/magnaflow.jsonl eval/questions/codebase_ai.jsonl eval/questions/rich.jsonl
```

`--others` also checks the classifier the other way: how many specific-code questions it wrongly treats as overviews.

## Reranking ablation (M4)

```bash
python eval/run_rerank_eval.py --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --suite magnaflow path/to/MagnaFlow eval/questions/magnaflow.jsonl \
    --suite codebase_ai . eval/questions/codebase_ai.jsonl \
    --suite rich .venv/Lib/site-packages/rich eval/questions/rich.jsonl
```

Compares the default retrieval with the same retrieval followed by each `--reranker` (repeatable), question by question,
and reports the latency cost. The first run of a model downloads it from Hugging Face.

## Classifier phrasing check, release notes, scale (M5)

```bash
python eval/check_classifier.py my_questions.json     # does the overview classifier recognise YOUR phrasing?
python eval/run_changelog_eval.py --suite requests path/to/requests eval/questions/requests.jsonl ...
python eval/scale_test.py --sizes 200,1000,4000        # synthetic repositories, no embedding model
```

`classifier_heldout.json` holds the held-out classifier cases; it was written before the patterns were widened and by the
same person, so it guards against regressions rather than proving generalisation. `questions/requests.jsonl` is 24
questions about `psf/requests` (20 about code, 4 about releases) used by the changelog evaluation.

## Answer-quality evaluation (M4; needs a model and your key)

`run_answers.py` answers the labelled questions with a real model and scores grounding, citation validity and
precision, and (with a judge) faithfulness. It calls a language model once per question and once more per answer for
the judge, so it costs money with a hosted provider and sends retrieved code to it: it refuses to start without `--yes`
in that case and prints exactly what it would do first.

```bash
python eval/run_answers.py path/to/repo eval/questions/magnaflow.jsonl \
    --provider anthropic --model <answering model> \
    --judge-provider openai --judge-model <a different model> \
    --limit 20 --export-sample 10 --yes
python eval/run_answers.py --check-sample my_verdicts.json --results eval/results/answer_eval.json
```

**Long runs and fair comparisons.** Every result is also appended to `<out-json>.partial.jsonl` as soon as it is
scored; after a crash or Ctrl+C, the same command with `--resume` carries on (failed answers are retried, and a
checkpoint from a different model, judge, question file or frozen file is refused). To give every model and prompt
exactly the same code, store the retrieval once and replay it:

```bash
python eval/run_answers.py REPO QUESTIONS --provider ollama --freeze-retrieval frozen.json   # no model is called
python eval/run_answers.py REPO QUESTIONS --provider ollama --model qwen2.5-coder:7b --frozen frozen.json
```

**The experiment matrix** (research plan, step 3.3). `run_matrix.py` runs every model x repository x condition from a
TOML config (example: `../research/matrix.dev.toml`). It checks that Ollama is running and every model is pulled,
freezes retrieval once per repository, runs each job with `--resume`, and appends a provenance record per job to
`runs.jsonl` (code commit and uncommitted changes, Ollama version, model digest and quantisation, the configured settings,
and under `applied` what the model actually got: code budget, output cap and temperature, which Ollama narrows):

```bash
python eval/run_matrix.py research/matrix.dev.toml --dry-run      # the job list; nothing runs
python eval/run_matrix.py research/matrix.dev.toml --limit 2      # a smoke run; drop --limit to continue to all
```

Running the same command again resumes. Each condition is a prompt and a context. The prompts are P0 to P4 of the
protocol (`answer_eval.apply_prompt`; `run_answers.py --prompt P0` runs one by hand, and the default P1 is the tool's
own prompt); the exact prompt goes into each result file. The context is `frozen` (the retrieved code) or `oracle`:
the same retrieval limited to each question's gold files, frozen once per repository into `frozen_oracle.json`
(`run_answers.py --freeze-retrieval FILE --oracle` does it by hand). A condition can list `models` to run on only some
of them, as P2 to P4 and the oracle do in the dev config.

**Human grading** (research plan, step 3.6). `export_annotation.py` turns finished result files into blinded grading
batches: `main/` with every answer and `second/` with a random 30% for the agreement check, each holding a reading packet
(question, key facts, answer, cited code), `claims.csv` (one row per sentence to label for support) and `answers.csv`
(correctness, flag, minutes), plus `key.json`, which says which model and prompt wrote each answer and stays with you.
Key facts are an optional `key_facts` list in the question file. The guidelines are `../research/ANNOTATION_GUIDELINES.md`.

```bash
python eval/export_annotation.py RESULTS.json [RESULTS.json ...] --out grading/HG-A
python eval/export_annotation.py results/*/P1-frozen/*.json --sample 50 --out grading/HG-B
```

**Claim-level judge** (research plan, step 3.7). `judge_claims.py` labels each claim (the same sentence rows the graders
get) as supported, unsupported or contradicted against only the code that claim cites, one model call per cited claim;
a claim that cites nothing is unsupported by rule, without a call. One JSON line per claim; re-running resumes.

```bash
python eval/judge_claims.py RESULTS.json [RESULTS.json ...] --judge-model gemma3:4b --out labels/gemma3_4b.jsonl
```

**Analysis** (research plan, step 3.8). `analyze.py` computes the protocol's primary comparisons (C1 to C5, Holm-adjusted,
with failed answers counted as failures and again without them) and the exploratory tables, from the matrix results,
the graded folders and the judges' labels. `power_simulation.py` gives the power of H1 and H2 for a range of sample
sizes and disagreement rates (`../research/POWER_SIMULATION.md`). Both need numpy, scipy and pandas, which the project's
own dependencies already bring in (sentence-transformers and Streamlit); nothing extra is installed.

```bash
python eval/analyze.py eval/results/matrix --grading grading/HG-A grading/HG-C --judges labels/*.jsonl --out analysis
python eval/power_simulation.py --out-md research/POWER_SIMULATION.md
```

**Determinism** (research plan, step 3.9). Answer the same questions twice with the same frozen retrieval, then compare:
`check_determinism.py` says per question whether the answer is identical, how similar it is, where the two first part,
and whether any score changed, and warns if the two runs were not set up the same way. With Ollama, a run started
straight after another can differ from a fresh one; `run_answers.py --fresh-model` reloads the model for every answer
so a re-run repeats exactly (`run_matrix.py` always uses it, and `judge_claims.py` always reloads its judge). Findings:
`../research/DETERMINISM_CHECK.md`.

```bash
python eval/run_answers.py REPO QUESTIONS --provider ollama --model qwen2.5-coder:7b --frozen F --limit 20 --out-json det/first.json
python eval/run_answers.py REPO QUESTIONS --provider ollama --model qwen2.5-coder:7b --frozen F --limit 20 --out-json det/second.json
python eval/check_determinism.py det/first.json det/second.json --out-md det/report.md
```

The second command of the first block in the answer-quality section compares your hand-graded verdicts with the judge's. See `../docs/QUALITY_EVAL.md` section 3 for what
is scored and why the judge needs checking. The first results, from local models through Ollama (no key, nothing sent
anywhere), are in section 3 of that document.
