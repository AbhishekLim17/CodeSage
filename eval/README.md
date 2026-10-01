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

The second command compares your hand-graded verdicts with the judge's. See `../docs/QUALITY_EVAL.md` section 3 for what
is scored and why the judge needs checking. No results are recorded yet: no model was run while this was built.
