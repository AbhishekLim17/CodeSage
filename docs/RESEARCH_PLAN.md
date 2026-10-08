# Research plan: grounding and faithfulness of small local models in repository question answering

Status: **draft v0.2, 2026-10-06.** It turns the existing system (CodeSage) from an engineering project into the test
bed for a research study. It assumes the research question recommended in the assessment ("option A"); section 12 says
how to switch to another one.

**Progress** (updated at the end of each step):

| Step | State | Where |
|---|---|---|
| 0. Ground rules and supervisor | **prepared; waiting for you** (the meeting and the rule-checking are yours) | `research/STEP0_SUPERVISOR_BRIEF.md` |
| 1. Literature review | **prepared; waiting for you** (reading and writing are yours) | `research/STEP1_READING_GUIDE.md` |
| 2. Protocol | **drafted; waiting for your decisions and your supervisor** | `research/STEP2_PROTOCOL_DRAFT.md`, `research/ANNOTATION_GUIDELINES.md` |
| 3. Test bed (coding) | **Step 3 complete** (2026-10-07). **Audit fixes (2026-10-08):** unanswerable questions can now be written (type `unanswerable`, no gold files; retrieval evaluation leaves them out, answers to them get no precision score, and they have no oracle), and the matrix runner records the variant Ollama actually runs when it keeps a model as several (gemma3:4b since Ollama 0.40). **Since then:** the D11 fix is built and checked (`--fresh-model`: two runs straight after each other gave 4 of 4 identical answers, against 2 of 8 without it), and `eval/figures.py` draws the analysis as figures (matplotlib, PNG and PDF); the H2 power figure is in `research/figures/`. **3.9 done:** `eval/check_determinism.py`; the 7B and 8B models re-run straight after themselves matched on only 2 of 8 answers (one *grounded* flipped), but a run from a freshly loaded model reproduced exactly (4 of 4, and 8 of 8 against yesterday's run on the old Ollama version): `research/DETERMINISM_CHECK.md`, **decision D11** (unload the model before every answer; recommended, not yet built). **3.8 done** (2026-10-07): `eval/analyze.py` computes C1 to C5 with Holm, with and without failed answers, plus the exploratory tables (RQ4 error sources, grader agreement, each judge's kappa over all, cited and same-family claims), checked against known values and a small study worked out by hand, and run on the real smoke results. `eval/power_simulation.py` (`research/POWER_SIMULATION.md`): H2 has 78% power at best with 150 questions and 20% disagreement, so **decisions D9 (C1 method) and D10 (H2 power) are added to the protocol**. No figures yet: matplotlib is not installed. **3.7 done** (2026-10-07): `eval/judge_claims.py` labels the graders' sentence rows (supported / unsupported / contradicted) against only the code each claim cites; uncited claims are unsupported by rule, without a call. Tried with the three protocol judges on the six real 7B smoke answers (40 claims, 13 cited, about 12 to 14 s per cited claim): gemma3:4b said supported to all 13; llama3.1:8b and qwen2.5-coder:7b each caught a claim whose cited excerpt only defined the exception it said was raised. At that speed the three judges on the human-graded sets need roughly 15 hours, and one judge on every answer about as much again. **3.6 done** (2026-10-07): `eval/export_annotation.py` writes blinded grading batches (main grader: every answer; second grader: a random 30%), each a reading packet plus `claims.csv` (one row per sentence) and `answers.csv`, and a `key.json` for you; questions may carry `key_facts`. Tried on the real 7B smoke answers: 6 answers, 40 claim rows, the split followed sentences and kept each citation with its sentence. **3.4 and 3.5 done** (2026-10-07): the prompts P0 to P4 (`--prompt`, a test keeps them word for word the protocol's, the exact prompt is stored with every result) and the oracle context (`--freeze-retrieval ... --oracle`, a search limited to the gold files before ranking; all 36 dev questions' gold files found in the index). Checked with the 7B model: each prompt reached the model as specified (P3 cited only [1] to [3]; P4's numbers mirrored P1's), and oracle answers ran. **Open decision D8** (oracle per-file cap) added to the protocol. **3.1 to 3.3 done** (2026-10-06): checkpoint and `--resume`; `--freeze-retrieval` / `--frozen` (checked with the real model, deterministic); the matrix runner `eval/run_matrix.py` with per-job provenance, including the values the model actually got (code budget 7,705 tokens, output cap 4,096, temperature 0). The four answering models are pulled. Smoke run (4 models x httpx and jinja2 x 2 questions, P1, frozen): 8 of 8 jobs finished, re-running resumed with no model calls. Seconds per answer on this laptop: about 30 to 45 (3B), 70 (4B), 80 to 110 (7B), 125 to 245 (8B), so the 12 conditions on about 150 questions need roughly **40 to 60 hours** of machine time before any judge, run overnight in resumable pieces. | section 9; `eval/README.md` |
| 4 to 10 | not started | |

**Two corrections made while drafting the protocol (they supersede what this plan said earlier):** (1) the
non-inferiority margin for H2 is **-10 points**, not -5: with about 150 questions a 5-point margin cannot be
demonstrated; (2) the human-grading design is now the four sets of `STEP2_PROTOCOL_DRAFT.md` section 5 (about 560 answers,
one grader each and 30% by two), not "300 answers, each by two": the old design broke the pairs that H2 and H3 need.

**How to use it.** Work through the steps in section 9 in order. Each step says what to do, who does it, what it
produces and when it is done. Steps 0 and 1 can start today. Every number marked *estimate* is an extrapolation, not a
measurement, and the pilot (Step 4) exists to replace it.

---

## 1. Where this starts from

What is measured today, and how strong it is:

| Fact | Evidence | Limit |
|---|---|---|
| Retrieval finds the right file: MRR 0.80 to 0.92 on code it was not tuned on | 36 blind questions, 2 Python libraries | small; labels written by the builder |
| Hybrid search does not beat vector search; test-file demotion helps; one reranker hurts | 85 to 145 questions, paired tests | one embedding model, small sets |
| A local 7B model never invents a citation or a location | 0 of 30 answers | one model, one run |
| A third of its answers cited nothing; one line at the end of the prompt fixed it (40% to 100% grounded, 6 better, 0 worse) | 10 held-out questions, p = 0.03 | **10 questions, one model** |
| 14 of 20 answers fully correct, 4 partly, 2 wrong; both wrong ones faithful to wrong code | hand-graded by the system's builder | **not independent**, 20 answers |
| A 4B judge called all 33 answers "supported" | one judge | the judge's weakness is the finding, but it is one judge |

Everything that follows is about replacing "10 questions, one model, graded by the builder" with a study that could
survive examination. The pilot findings are **hypotheses to test**, not results to cite.

## 2. Research questions and hypotheses

**Working title:** *When small local language models answer questions about code, do they cite faithfully, and what
makes them better?*

| | Question | Why it matters |
|---|---|---|
| **RQ1** | How often do small open models (about 3 to 8 billion parameters) give answers about a code repository that are cited, faithful to the code they cite, and correct? | On-device code assistants are a live concern (privacy, cost), and nobody reports these numbers for repository questions. |
| **RQ2** | Which cheap interventions (where the citation instruction sits, how many sources are shown, in what order) improve grounding without hurting correctness? | Pilot: one line moved grounding from 40% to 100%. |
| **RQ3** | Can a small local model act as a faithfulness judge, compared with human annotators? | The pilot's 4B judge said "supported" for everything. Automated evaluation is only useful if it can disagree. |
| **RQ4** | When an answer is wrong, is it because retrieval supplied the wrong code or because the model misused the right code? | Pilot: both wrong answers were faithful to wrong code. This separates the two failure sources. |

**Hypotheses** (to be fixed in the protocol, Step 2, *before* the test data is collected):

- **H1.** A citation reminder at the end of the prompt raises the share of answers that cite a source for every model
  tested, and the gain is larger for smaller models.
- **H2.** The reminder does not reduce correctness. Tested as *non-inferiority*: the lower bound of the 95% interval of
  the correctness difference stays above -10 percentage points (a -5 margin cannot be shown with about 150 questions;
  see `research/STEP2_PROTOCOL_DRAFT.md` section 8).
- **H3.** Showing fewer sources (3 instead of 12) raises citation precision but lowers correctness on questions whose
  right file is not among the top 3.
- **H4 (exploratory).** Putting the best-ranked source last instead of first changes grounding or correctness (cf. Liu
  et al., *Lost in the Middle*). No direction is predicted.
- **H5.** Judges of 4B parameters or fewer agree with human annotators at Cohen's kappa below 0.4; 7 to 8B judges do
  better, but not enough to replace humans.
- **H6.** More than half of the *incorrect* answers are faithful to the code they cite, and giving the model the
  correct files directly (an oracle context) removes most of them.

## 3. What would be claimed, and what would not

If the study works: a benchmark of repository questions with independent labels; measured grounding, faithfulness and
correctness for several small open models; the effect of three cheap interventions; evidence on whether small judges
can be trusted; and a decomposition of errors into retrieval and generation failures.

**Not claimed:** that the system is better than other tools; results for models above 8B or for hosted models, unless
they are run (Claude as an upper bound is optional); results for languages not in the data; that any intervention
transfers to other prompts or tasks; **any effect on developer productivity or onboarding time** (no user study is
planned: see below).

**Novelty, honestly.** None of the ingredients is new. The contribution is an *independent, controlled measurement*
in a setting (repository question answering, small local models) where existing work either uses large models,
generates code instead of answering questions, or evaluates retrieval without looking at what the model does with
it. Related work to position against is in section 13.

### Fit with the submitted project title and description

The registered title is **"AI Codebase Understanding System"**, described as a tool that uses NLP, LLMs and RAG so
that developers can ask natural-language questions about large codebases and get accurate explanations with
references, in order to simplify comprehension, improve productivity and reduce onboarding time. This plan keeps that
system as the thing being built and studied; what it adds is a rigorous evaluation of the part the description calls
"accurate explanations along with references".

| Part of the description | In this plan |
|---|---|
| Understand large, complex codebases | The system is the test bed. "Large" is shown only up to a few hundred real files (4,000 synthetic); the test repositories are mid-sized (5,000 to 100,000 lines). Say "mid-sized to large" and do not claim more, or add one genuinely large repository. |
| NLP + LLM + RAG | RAG and the LLM are studied (RQ1 to RQ4). The NLP components (identifier-aware keyword search, follow-up rewriting, overview-question classification) are described in the system chapter and evaluated in `docs/QUALITY_EVAL.md`. |
| Source code, documentation, project structure | All three appear in the test questions (the *overview* type was added for the structure part). |
| Accurate explanations with references to files and code | RQ1 (how often) and RQ4 (why wrong). The core of the study. |
| **Improve productivity, reduce onboarding time** | **Decision (2026-10-06): option (a), chosen by you.** It is the *motivation*, not a measured outcome: this study does not test whether the tool makes anyone faster, and nothing in the dissertation may say it does. The registered objective is reworded accordingly (below), **subject to your supervisor's approval**. The alternative, a small user study (option b: about 8 to 12 programmers, with and without the tool, time and correctness measured; ethics approval; 3 to 4 more weeks; a real chance of a negative result, since a 7B model takes about 100 seconds per answer on this laptop), is **future work**, to be named in the limitations. |

**Proposed rewording of the project description** (to take to your supervisor; the changed sentences are the last two):

> The AI Codebase Understanding System is an intelligent software tool that aims to help developers understand large
> and complex codebases. It uses Natural Language Processing (NLP), Large Language Models (LLMs), and
> Retrieval-Augmented Generation (RAG) to analyze source code, documentation, and project structure. Users can ask
> questions in natural language about the codebase, and the system provides explanations along with references to the
> relevant files and code sections. **This project evaluates how accurate, faithful and well-referenced such
> explanations are when produced by small, locally run language models, and which techniques improve them. The
> system is designed to support code comprehension and developer onboarding; measuring its effect on developer
> productivity is outside the scope of this study and is left to future work.**

Two things changed from what you submitted: "provides accurate explanations" became "provides explanations" (accuracy
is something the study *measures*, not something it can promise), and the productivity claim became a stated scope
limit.

If the supervisor wants the registered title kept exactly, the research questions go in the dissertation's aims, not
the title. If a subtitle is allowed, something like *"... : evaluating grounded question answering with small local
models"* tells the reader what the research is.

## 4. Experimental design

**The key control: freeze retrieval.** For each question the retrieved sources are computed once and stored. Every
model and every prompt then sees *exactly the same sources*, so any difference is due to the model or the prompt, not to
search. (Retrieval is deterministic here, but storing it also records which index and model produced it.)

| Factor | Levels |
|---|---|
| **Answering model** | `qwen2.5-coder:3b`, `gemma3:4b`, `qwen2.5-coder:7b`, `llama3.1:8b` (Ollama, default 4-bit builds). Optional upper bound: one hosted model (Claude), if budget allows. |
| **Prompt, all four models** | **P0** current system prompt only (the pilot baseline); **P1** plus the end-of-prompt reminder (current default). |
| **Interventions, on the 7B model only** | **P2** P1 plus one worked example of a cited answer; **P3** only the top 3 sources; **P4** sources in reverse order (best last). |
| **Context** | the frozen retrieval, and an **oracle** context (the gold files only) for answerable questions. |
| **Fixed everywhere** | temperature 0, `num_ctx` 16,384, same index, same embedding model, same token budget. Record the model digest, code commit and seed. |

That is 4 models x 2 prompts + 3 interventions + 1 oracle condition = **12 conditions** on the same questions.

## 5. Data

Two sets, kept strictly apart.

- **Development set (exists):** the 145 questions in `eval/questions/` plus the pilot's questions. Used to tune and to
  debug. Written by the builder, so never used for a claim.
- **Test set (to build):** about **150 questions** over **8 repositories** in **at least 3 languages**
  (Python, JavaScript/TypeScript and one of Java or Go), written and labelled by **people other than the builder** and
  frozen (git tag plus a hash) *before any model is run on it*. Rule: no prompt, retrieval or code change after the
  freeze is allowed to depend on test-set results.

**Repository selection** is by a written rule decided in advance (permissive licence, 5,000 to 100,000 lines, active in
the last two years, a mix of sizes, at least two with a substantial test suite), with pinned commit hashes, not by what
looks easy.

**Question mix:** 15% *locate*, 30% *explain*, 20% *cross-file* (the answer needs two or more files), 20%
*overview* (what the project does, how it is organised, where to start: answered from the README, the documentation
and the repository map, so the "project structure" part of the registered description is evaluated too), 15%
*unanswerable* (the repository does not contain the answer; the right behaviour is to say so). Each question has gold
files (or gold directories, for overview questions) and a short list of **key facts** an answer must contain.

**People.** At least three: you plus two peers with programming experience. Each repository's questions are written
by one person and the gold files and key facts are checked by a second.

## 6. Measures

| Measure | How | Who |
|---|---|---|
| Grounded, citation validity, precision, locations never shown | automatic (exists) | code |
| **Faithfulness**, claim level | each answer is split into claims (sentences to start with, merged or split by the annotator); each claim is labelled *supported*, *unsupported* or *contradicted* by the sources it cites. Faithfulness = share of supported claims. (After Min et al., FActScore; Gao et al., ALCE.) | 2 humans on a sample |
| **Correctness** | the key facts: *correct* (all present, none contradicted), *partly*, *wrong* | 2 humans on the same sample |
| **Abstention** | on unanswerable questions: did it say it could not find the answer? | human, 1 |
| Cost | seconds and tokens per answer | code |
| Judge verdict | the LLM judge's label on the same claims | judges |

**Human grading is a designed sample, not everything:** about **560 answers** in four sets that keep the pairs the
hypotheses need (the 7B model with both prompts on **all** questions; the other three models on 50 questions; the
fewer-sources condition on 60; the oracle on the failures). One grader grades each answer and a second grades a random
30% independently, so agreement (Cohen's kappa, Krippendorff's alpha) is measured and disagreements adjudicated. The
rest get the automatic measures and one chosen judge's verdicts. Details: `research/STEP2_PROTOCOL_DRAFT.md`, section 5.

## 7. Statistics (to be fixed in the protocol)

- **Unit of analysis:** the question. Conditions are compared on the *same* questions (paired).
- **Binary outcomes** (grounded, correct): exact McNemar test; effect reported as a difference in proportions with a
  95% confidence interval.
- **Continuous outcomes** (faithfulness share, citation precision): paired bootstrap interval of the mean difference.
- **Non-inferiority (H2):** the lower bound of the correctness difference's interval against the -10 point margin.
- **Many comparisons:** the primary comparisons are listed in the protocol and corrected together (Holm); everything
  else is labelled exploratory.
- **Model x prompt interaction (H1):** mixed-effects logistic regression with random intercepts for question and
  repository.
- **Judges (H5):** accuracy, Cohen's kappa and a confusion matrix against the human labels, plus checks for a bias toward
  "supported" and for judging answers from its own model family.
- **Sample size:** a **power simulation** (Step 3) decides the final number of questions. Rough guide, to be replaced
  by the simulation: with about 150 paired questions, a difference of 10 to 15 points is detectable if a quarter to a
  third of questions change outcome between conditions; smaller effects are not.
- **Randomness:** temperature 0 should be deterministic, but CPU/GPU splits can change outputs slightly. Step 3
  measures this by running a question set twice, and reports it.

## 8. Compute, effort and cost (all estimates)

Anchored on one measurement: **about 100 s per answer for the 7B model on this laptop** (GTX 1650 4 GB, 41% of the
model on the GPU, the rest on a Ryzen 5 5600H CPU). The 3B model should be faster and the 8B slower; neither is measured.

| Work | Volume | Estimate |
|---|---|---|
| 4 models x 2 prompts x 150 questions | 1,200 answers | about 30 hours |
| 7B interventions + oracle (4 x 150) | 600 answers | about 17 hours |
| Judge study: 3 judges on the 560 human-graded answers | about 1,700 judgements | about 14 hours |
| One chosen judge on all answers | about 1,800 judgements | about 15 hours |
| **Total machine time** | | **about 75 hours, i.e. 3 days non-stop, in practice a few weeks of nights** |
| Question writing and checking (150 questions) | about 40 + 12 person-hours | spread over 3 people |
| Human grading (560 answers once, 170 of them again; about 6 minutes per answer) | about 70 person-hours in all | spread over 3 people |
| Pilot grading (60 answers, two people) | about 6 hours each | |
| **Total human effort** | **about 120 person-hours** | |

Disk: four models about 15 GB (you have 46 GB free on C:, where Ollama keeps them). A hosted upper-bound model costs
real money: check the provider's current price before deciding; a one-model, 150-question run is small. If the
laptop proves too slow, university compute or a rented GPU would shorten the 75 hours a lot, and the plan does not
depend on it.

## 9. The steps

Owner key: **YOU** = must be you (a research project's ideas, decisions and writing have to be yours); **ME** = I do
it, you review; **BOTH**.

### Step 0. Fix the ground rules (week 0)

**Owner: YOU.** *Before anything else, because they change what I am allowed to do.*

1. Read your university's rules on **AI assistance** in a dissertation: what must be disclosed, and what is not
   allowed. This project has been built with an AI assistant throughout (code, tests, questions, documents), so
   whatever the rule says has to be followed honestly and stated in the thesis.
2. Check whether **recruiting peers as annotators** needs **ethics approval** at your institution.
3. Take this plan, `docs/PAPER_REFERENCE.md` and `docs/REVIEW_2026-10-06.md` to your **supervisor**. Ask: is this
   research question acceptable? How many pages and what structure? What counts as your contribution?
4. Decide: topic A (this plan), B or C (section 12).
5. **Decided: option (a).** Bring the proposed rewording of the project description (section 3, "Fit with the
   submitted project title and description") to your supervisor and get it approved in writing. Also ask whether a
   subtitle may be added to the registered title. If the supervisor insists on measuring productivity, the user study
   (option b) comes back as RQ5 and the timeline grows by 3 to 4 weeks plus ethics approval.

**Done when:** a supervisor has agreed to a research question in writing (an email is enough), has approved the
reworded objective (or asked for the user study instead), and you know the AI-use and ethics rules.

### Step 1. Literature review (weeks 1 to 4)

**Owner: YOU** (I can find papers, quiz you, and criticise drafts, but not write the review).

Read in this order and write a **half-page note per paper** (question, method, data, metric, main result, limitation,
what it means for this study): first the five that define the problem (ALCE, FActScore, Verifiability in generative
search engines, Lost in the Middle, LLM-as-a-judge), then code-specific work (CodeRAG-Bench, RepoQA, CoSQA, SWE-bench),
then general RAG (Lewis et al., Ragas, RAGTruth). Section 13 lists them. Then search for what it misses: small models
in RAG, effects of quantisation, abstention, inter-annotator methodology.

**Output:** a literature review of about 8 to 10 pages ending in a **gap statement**: one paragraph saying what is not
known and why RQ1 to RQ4 fill it. **Done when** your supervisor says the gap is real.

### Step 2. Research protocol (weeks 3 to 5)

**Owner: BOTH** (you decide, I draft the mechanics and check them for holes).

A short document fixing everything that must not change after seeing data: the four RQs, hypotheses H1 to H6, the
model list, the prompts (exact text), the repository selection rule, the question mix, the measures, the annotation
guidelines, the primary comparisons and the statistical tests. Optionally **pre-register** it (the Open Science
Framework is free) so the timing of decisions is on the record.

**Done when:** your supervisor has read it and a dated copy is committed to the repository.

### Step 3. Upgrade the test bed (weeks 3 to 6)

**Owner: ME**, in small reviewable steps, each with tests (this is engineering I can do well and quickly).

| # | Change | Why it is needed |
|---|---|---|
| 3.1 | **Resumable runner:** write each result the moment it is produced; `--resume` skips finished questions | today the results file is written only at the end of a run, so a crash at question 149 loses everything |
| 3.2 | **Frozen retrieval:** store each question's sources; replay them for every model and prompt | the controlled comparison in section 4 |
| 3.3 | **Experiment-matrix runner:** one config file lists models x prompts x contexts x questions; runs detached; records model digest, code commit, seed and settings | 12 conditions cannot be run by hand |
| 3.4 | **Named prompt variants** P0 to P4 (default unchanged) | the interventions |
| 3.5 | **Oracle context** condition | RQ4 |
| 3.6 | **Annotation export:** answers split into sentence-level claims with their cited code, in a spreadsheet-friendly format, judge verdicts hidden | human grading without a custom tool |
| 3.7 | **Claim-level judge** (one verdict per claim) | RQ3 |
| 3.8 | **Analysis package:** paired tests, Holm correction, mixed model, kappa and alpha, figures, and the **power simulation** | section 7 |
| 3.9 | **Determinism check:** the same questions twice, report any difference | section 7 |

**Done when:** a 5-question smoke run of the whole matrix works end to end and can be stopped and resumed.

### Step 4. Pilot (weeks 5 to 6)

**Owner: BOTH.** 24 questions (3 repositories x 8), written and labelled by you and a peer, answered under all 12
conditions (288 answers), of which a random 60 are graded by two people. Purpose: **find out whether the protocol works**, not to get results. Measure how
long writing, checking and grading really take, whether the annotation guidelines are clear, and annotator agreement.

**Gate G1:** kappa on the pilot is at least 0.6 and the effects are large enough to detect with the planned number of
questions. If not, **revise the guidelines or the question count** before spending weeks on the full set.

### Step 5. Build and freeze the test set (weeks 6 to 10)

**Owner: YOU + peers**, me for tooling and consistency checks (broken paths, duplicate questions, answers that name the
file in the question).

Write, cross-check and fix about 150 questions; record agreement on the gold labels; then **freeze** (tag, hash,
`docs/DATASET.md` describing how it was built).

**Gate G2:** after the freeze, the dataset does not change.

### Step 6. Run the experiments (weeks 10 to 13)

**Owner: ME**, running detached and unattended; you check progress and decide on scope cuts (section 11).

Index the 8 repositories at their pinned commits, freeze retrieval, run the 12 conditions, run the judges. Results go
to `eval/results/` with the full configuration of each run.

**Done when:** every condition has an answer for every question, or a recorded reason why not.

### Step 7. Human evaluation (weeks 12 to 15)

**Owner: YOU + peers**, using the export from 3.6 and `research/ANNOTATION_GUIDELINES.md`. About 560 answers in the
four sets of the protocol, one grader each and a random 30% graded again independently by someone who did not write the
question. I compute agreement and prepare the next batch; I do not grade, and nobody uses a language model to help.

**Done when:** agreement is reported and disagreements have been resolved by the rule fixed in the protocol (a third
person, or discussion).

### Step 8. Analysis (weeks 15 to 17)

**Owner: ME** to run the pre-specified analyses and produce figures; **YOU** to interpret them. Report all primary
comparisons, including the ones that did not work. Add the error decomposition for RQ4: for each wrong answer, was the
gold file in the context (generation failure) or not (retrieval failure)?

### Step 9. Write (weeks 14 to 20; start the methods chapter in week 6)

**Owner: YOU.** I can check numbers against the results files, find unsupported claims and inconsistencies, and
explain statistics. The prose and the argument should be yours, within your university's AI rules. Suggested
structure: introduction and gap; related work; the system (short; most of it is already in
`docs/ARCHITECTURE.md`); method; results by RQ; error analysis; limitations; conclusion. **The limitations must say
that no user study was done**, so the effect on productivity and onboarding is untested, and name it as future work.

### Step 10. Release and submit (weeks 19 to 20)

**Owner: BOTH.** Release the questions, labels, answers, human grades and analysis code with exact versions (not the
repositories' code, only their names and commit hashes), under a licence that respects the repositories' own. Check
the dissertation's numbers against the released files one last time.

## 10. Timeline

| Weeks | Work | Gate |
|---|---|---|
| 0 | Step 0 | supervisor agrees RQ |
| 1 to 4 | Step 1 literature | gap statement accepted |
| 3 to 5 | Step 2 protocol | protocol read by supervisor |
| 3 to 6 | Step 3 test bed (parallel) | smoke run passes |
| 5 to 6 | Step 4 pilot | **G1** |
| 6 to 10 | Step 5 dataset | **G2** freeze |
| 10 to 13 | Step 6 experiments | all conditions complete |
| 12 to 15 | Step 7 human grading | agreement reported |
| 15 to 17 | Step 8 analysis | primary results fixed |
| 14 to 20 | Step 9 writing | draft to supervisor by week 18 |
| 19 to 20 | Step 10 release | |

About **20 weeks (5 months)**. If your deadline is shorter, section 11 says what to cut, in order.

## 11. Risks and scope cuts

| Risk | What to do |
|---|---|
| Peers are not available to annotate | Reduce to 100 questions and one extra annotator; accept lower power and say so. Never fall back to the builder grading alone. |
| Effects are too small to detect | Decided at G1 from the pilot; if so, report estimates with intervals and no significance claims, or enlarge the set. |
| Machine too slow | Drop the 8B model first, then P4, then reduce interventions to P3 only; consider rented GPU time. |
| Ollama or model builds change mid-study | Record model digests; do not update Ollama or re-pull models during Steps 6 to 8. |
| The 4B judge remains useless | That is itself the RQ3 finding; report it, and add a larger judge (a hosted one) as the reference. |
| Annotators disagree too much | Revise guidelines at G1; report agreement honestly; use adjudication. |
| Scope creep | The protocol is the contract; changes after the freeze are logged and labelled. |

**Cut in this order if time runs short:** (1) the hosted upper-bound model, (2) the third language, (3) interventions
P2 and P4, (4) RQ3 down to two judges, (5) 150 down to 100 questions. Do **not** cut: the independent annotators, the
frozen retrieval, the pre-specified comparisons, or the pilot.

## 12. Alternatives to this research question

- **Option B, "faithful but wrong".** Narrow to RQ4: how often code answers faithfully describe dead or unrelated
  code, and whether retrieval that prefers code reachable from entry points reduces it. More novel, more work (needs a
  reachability analysis per language). Steps 0 to 2 and 5 to 10 stay; Step 3 adds the reachability retrieval.
- **Option C, "does syntax-aware chunking matter?"** Compare tree-sitter chunking with fixed and recursive
  splitting across languages and question types. The cleanest experiment and the least novel; roughly 12 weeks. Steps
  4 and 6 change; the model matrix disappears.

## 13. Reading list (all checked against arXiv on 2026-10-06)

| Topic | Paper |
|---|---|
| RAG | Lewis, Perez, Piktus et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. arXiv:2005.11401 |
| Citations in answers | Gao, Yen, Yu et al. (2023). *Enabling Large Language Models to Generate Text with Citations* (ALCE). EMNLP 2023. arXiv:2305.14627 |
| Claim-level factuality | Min, Krishna, Lyu et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. arXiv:2305.14251 |
| Do citations support claims? | Liu, Zhang, Liang et al. (2023). *Evaluating Verifiability in Generative Search Engines.* Findings of EMNLP 2023. arXiv:2304.09848 |
| Context position | Liu, Lin, Hewitt et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL. arXiv:2307.03172 |
| LLM judges | Zheng, Chiang, Sheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.05685 |
| RAG evaluation | Es, James, Espinosa-Anke et al. (2023). *Ragas: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217 |
| RAG hallucination data | Niu, Wu, Zhu et al. (2024). *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.* arXiv:2401.00396 |
| Retrieval for code | Wang, Asai, Yu et al. (2024). *CodeRAG-Bench: Can Retrieval Augment Code Generation?* arXiv:2406.14497 |
| Repository understanding | Liu, Tian, Daita et al. (2024). *RepoQA: Evaluating Long Context Code Understanding.* arXiv:2406.06025 |
| Repository tasks | Jimenez, Yang, Wettig et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. arXiv:2310.06770 |
| Code question answering | Huang, Tang, Shou et al. (2021). *CoSQA: 20,000+ Web Queries for Code Search and Question Answering.* ACL 2021. arXiv:2105.13239 |

Venue and author lists were taken from the arXiv pages (first three authors shown); venues I could not see there are
left out. **Still to find and read:** work on small models in retrieval-augmented settings, the effect of quantisation
on faithfulness, abstention and "I don't know" behaviour, and the methodology references for the statistics (kappa,
Krippendorff's alpha, McNemar, Holm); cite those from their original sources.

## 14. Ethics, licences, reproducibility

- **Humans:** if annotators are people other than you, the ethics rule from Step 0 applies. Give them written
  guidelines, say how long it takes, and record that they agreed. No personal data is needed.
- **Licences:** only questions, labels, grades and answers are released; repositories are referred to by name and
  commit. Check each repository's licence before quoting code in the thesis.
- **Privacy:** everything runs locally; no code leaves the machine unless a hosted upper-bound model is added, in
  which case say so and use only open-source repositories.
- **Reproducibility:** every result records the code commit, the model digest, the index and embedding model, and the
  settings; the matrix runner writes them automatically (Step 3.3).
- **AI use:** keep a log of what the assistant did (it is largely in the repository history and the session notes) so
  the disclosure in the thesis is accurate.

## 15. What happens next

1. **You:** do Step 0 (this week) with `research/STEP0_SUPERVISOR_BRIEF.md`, then start Step 1 reading with the five
   core papers in `research/STEP1_READING_GUIDE.md`, and read the protocol draft with your supervisor.
2. **Me, only when you say "start Step 3":** Steps 3.1 and 3.2 (the resumable runner and the frozen retrieval). They
   are the first code in this plan. They are needed under every option and do not depend on the decisions in Steps 0 to
   2, so they could start while you talk to your supervisor, but that is your call.
3. **Together, at the end of each step:** a short review: what was done, what changed, what is the next gate.

One thing I need from you before Step 3 gets far: **which models you are willing to download** (about 15 GB in all),
and whether a hosted upper-bound model is possible for you (an API key and a small budget). Both can wait for
Step 0.
