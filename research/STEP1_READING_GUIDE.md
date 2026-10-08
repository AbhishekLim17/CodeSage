# Step 1: literature review, a reading guide

**Owner: you.** The review has to be your reading and your writing. This guide only tells you what to read first, what
to look for, and how to turn notes into a gap statement. I have not written any part of the review for you, and I will
not.

**What the descriptions below are, and are not.** Each one is my paraphrase of the paper's *abstract* on arXiv
(checked on 2026-10-06). They are to help you decide what to read closely. They are not a substitute for reading the
paper, and the abstracts rarely state limitations, so finding those is your job.

**Time budget (an estimate):** about 26 hours of reading over 4 weeks, plus note-taking and the writing itself.
Tier 1: about 3 hours per paper; tier 2: about 2 hours; tier 3: about 1 hour.

---

## 1. Reading order

### Tier 1: read closely (these define the problem)

**1. Gao, Yen, Yu et al. (2023). *Enabling Large Language Models to Generate Text with Citations* (ALCE).** EMNLP 2023.
arXiv:2305.14627
- *What it does:* a benchmark and automatic metrics for answers that cite their sources, covering fluency, correctness
  and citation quality; reports that even the strongest models leave the answer without full citation support about half
  the time on some datasets.
- *Why it matters here:* the closest definition of "citation quality". Your RQ1 and RQ2.
- *Extract:* how citation recall and precision are defined; how citation support is decided; which prompting strategies
  they tried; whether model size changes the result; how they validated the automatic metrics against people.

**2. Min, Krishna, Lyu et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text
Generation.* EMNLP 2023.** arXiv:2305.14251
- *What it does:* splits generated text into atomic facts and scores the share a knowledge source supports; reports
  surprisingly low scores for well-known models on biographies.
- *Why it matters here:* the model for claim-level faithfulness in your human evaluation, and for RQ3 (can an automatic
  estimator replace people?).
- *Extract:* what counts as an atomic fact and who decides; how the human annotation was organised and what it cost;
  how the automatic estimator was checked against humans.

**3. Liu, Zhang, Liang et al. (2023). *Evaluating Verifiability in Generative Search Engines.* Findings of EMNLP 2023.**
arXiv:2304.09848
- *What it does:* a human audit of four commercial answer engines, measuring how often statements are supported by their
  citations; finds frequent unsupported statements and inaccurate citations.
- *Why it matters here:* a worked example of human grading of "does the citation support the claim", the very thing
  your graders will do. Also the trade-off between a fluent answer and a checkable one.
- *Extract:* the annotation guidelines and the support scale; how many annotators; how agreement was measured; how
  queries were sampled.

**4. Zheng, Chiang, Sheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023
Datasets and Benchmarks.** arXiv:2306.05685
- *What it does:* tests whether strong language models can judge other models' answers; GPT-4 agreed with human
  preferences about 80% of the time; documents biases (position, verbosity, preferring its own style, weak reasoning).
- *Why it matters here:* RQ3. Note carefully that their judges are **large**; yours are 3 to 8B. That difference is
  your gap.
- *Extract:* the agreement measure; the bias experiments; the mitigations; what "agreement" means when humans disagree
  with each other.

**5. Liu, Lin, Hewitt et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL.**
arXiv:2307.03172
- *What it does:* tests how models use information placed at different positions in a long input; performance drops
  when the relevant text is in the middle, compared with the start or end.
- *Why it matters here:* your hypothesis H4 (source order) and the question of why a reminder placed at the *end* of
  the prompt works.
- *Extract:* how position was varied; which models and context lengths; whether small models were included; whether
  they test an instruction placed at different positions (they test documents, not instructions).

### Tier 2: read for design

**6. Niu, Wu, Zhu et al. (2024). *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented
Language Models.* arXiv:2401.00396**
- *What it does:* a corpus of roughly 18,000 retrieval-augmented responses with human labels for hallucinated spans;
  reports that small models fine-tuned on good hallucination data can detect hallucinations about as well as a very
  large one.
- *Why it matters here:* RQ3 and H5: small judges can work *if trained for it*; yours are not.
- *Extract:* the annotation scheme and agreement; the detectors and their sizes; the tasks covered (none is code).

**7. Wang, Asai, Yu et al. (2024). *CodeRAG-Bench: Can Retrieval Augment Code Generation?* arXiv:2406.14497**
- *What it does:* a benchmark for whether retrieved documents and code help *code generation*, including
  repository-level tasks; finds retrieval often helps but retrievers struggle when query and document share few words,
  and generators often fail to use what is retrieved.
- *Why it matters here:* the nearest work on "models do not use retrieved context well" (RQ1, RQ4), for generation, not
  question answering. Your difference: explanation with citations.
- *Extract:* which retrievers and models; how retrieval quality and generation quality are separated; their
  repository-level setup.

**8. Liu, Tian, Daita et al. (2024). *RepoQA: Evaluating Long Context Code Understanding.* arXiv:2406.06025**
- *What it does:* 500 tasks from 50 repositories in 5 languages, finding a function from a natural-language
  description; evaluates 26 models and reports gaps between open and proprietary models and across languages.
- *Why it matters here:* a precedent for repository selection across languages, and for comparing open models.
- *Extract:* how repositories were selected; how tasks were created and checked; which languages; the trend with model
  size; how they handled contamination.

**9. Es, James, Espinosa-Anke et al. (2023). *Ragas: Automated Evaluation of Retrieval Augmented Generation.*
arXiv:2309.15217**
- *What it does:* a framework to evaluate retrieval-augmented systems **without human-written references**, with metrics
  for retrieval quality and generation faithfulness.
- *Why it matters here:* the automatic route you are testing in RQ3; compare how they define faithfulness with yours.
- *Extract:* how faithfulness is computed (claim extraction, then checking); how they validated it against people (check
  the paper, the abstract does not say).

### Tier 3: skim for positioning and background

**10. Lewis, Perez, Piktus et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS
2020.** arXiv:2005.11401 - the original formulation: a language model combined with a dense retriever over Wikipedia; note
the authors' own remark that provenance for a model's decisions remains an open problem, which is your motivation.

**11. Jimenez, Yang, Wettig et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.**
arXiv:2310.06770 - 2,294 real issues from 12 Python repositories; success rates are low. A different task (fixing, not
understanding), useful for how they select repositories and report contamination.

**12. Huang, Tang, Shou et al. (2021). *CoSQA: 20,000+ Web Queries for Code Search and Question Answering.* ACL 2021.**
arXiv:2105.13239 - a dataset of web queries paired with code, for matching natural-language questions to code. Useful
for how real questions about code look and how pairs were annotated.

---

## 2. A note template (one per paper; half a page)

```
Paper:                       (authors, year, venue, arXiv id)
Read on:                     (date)   Time spent:
The question it asks:
Method (what they did):
Data (size, source, who labelled it):
Models and sizes:
Metrics (define them in your own words):
Main result (a number, not "it works"):
Limitation (from the paper's own limitations or discussion section):
What it does NOT cover:
What I will reuse (a definition, a protocol, a measure):
What I disagree with or doubt:
Where it fits (which RQ):
```

The two lines that do the real work are **"What it does NOT cover"** and **"What I disagree with or doubt"**. If you
cannot fill them in, you have not finished reading.

## 3. Self-check after each paper

Without looking at your notes, answer aloud or in writing: (a) what did they measure and how; (b) what was the single
most important number; (c) what could make that number misleading; (d) would the result hold for a 4B model on code?
If you cannot answer (c) and (d), read the evaluation section again.

## 4. What the abstracts do not cover: search for these yourself

Use arXiv, the ACL Anthology, Semantic Scholar and Google Scholar. Suggested search phrases:

- "small language models" with "retrieval-augmented generation" and "faithfulness"
- "quantization" with "hallucination" or "faithfulness"
- "abstention" or "unanswerable questions" with "retrieval-augmented"
- "attribution" or "citation" with "code question answering"
- "repository-level question answering" or "codebase question answering"
- "LLM judge" with "small models" or "agreement with human annotators"
- inter-annotator agreement methodology: Cohen's kappa, Krippendorff's alpha (find the original sources and cite them)
- paired significance testing in evaluation: McNemar's test, bootstrap confidence intervals, multiple-comparison
  correction (find the original sources)
- data contamination or memorisation in code benchmarks

Keep a list of what you searched and what you found, so the review can say how the literature was surveyed.

## 5. Turning notes into the gap statement (you write this)

Fill in the five blanks, in your own words, with specifics from your notes. Then give it to your supervisor.

1. **What is known:** (two or three findings with numbers, each with a citation)
2. **What is known only for a different setting:** (for example: large models, generation instead of question answering,
   English text instead of code)
3. **What is not known:** (the specific thing none of the papers measure)
4. **Why that matters:** (one concrete consequence for people building on-device tools)
5. **What this study does about it:** (research questions 1 to 4, one sentence each)

**A test of the gap:** a supervisor should be able to read it and say "I agree nobody has shown that", or point at a
paper that did. If they point at a paper, add it to your notes and revise the gap. That is the review working.

## 6. Suggested review outline (about 8 to 10 pages)

| Section | Content | Pages |
|---|---|---|
| Background | retrieval-augmented generation; why answers need provenance | 1 |
| Citing and attribution | ALCE, the verifiability audit, citation measures | 2 |
| Measuring faithfulness | atomic claims (FActScore), hallucination data (RAGTruth), automatic frameworks (Ragas) | 2 |
| Language models as judges | Zheng et al.; bias; the gap for small judges | 1.5 |
| Context use | Lost in the Middle; how models use supplied context | 1 |
| Code and repositories | CodeRAG-Bench, RepoQA, SWE-bench, CoSQA | 1.5 |
| Gap and research questions | your gap statement | 1 |

**Step 1 is done when** your supervisor has read the gap statement and agrees it is real. Tell me when you have drafts
and I will read them as a critical reader: finding claims a paper does not support, missing comparisons, and
inconsistencies. I will not rewrite them.
