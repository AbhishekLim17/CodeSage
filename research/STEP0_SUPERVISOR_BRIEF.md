# Step 0: ground rules and the supervisor meeting

**Owner: you.** This file prepares you for the meeting and the rule-checking. It does not replace either. Nothing here
has been sent to anyone.

Status of Step 0 when this was written (2026-10-06): **not started**. Everything after Step 2 depends on its answers.

---

## 1. The proposal in one page (what to hand over)

**Registered title:** AI Codebase Understanding System.

**What exists:** a working system (about 6,000 lines of library code, 980 automated tests) that answers natural-language
questions about a code repository and cites the files and line ranges it used. It runs fully offline with small local
language models. It has been evaluated for search quality on about 145 questions and for answer quality on 30 questions
with one local model.

**What is proposed:** turn it into a research study of the part the description calls "accurate explanations along with
references". Working title: *When small local language models answer questions about code, do they cite faithfully, and
what makes them better?*

**Research questions**

1. How often do small open models (about 3 to 8 billion parameters) give answers about a repository that are cited,
   faithful to the code they cite, and correct?
2. Which cheap interventions (where the citation instruction sits, how many sources are shown, in what order) improve
   grounding without hurting correctness?
3. Can a small local model act as a faithfulness judge, compared with human annotators?
4. When an answer is wrong, is it because retrieval supplied the wrong code or because the model misused the right code?

**Why it is worth asking:** on-device code assistants are a live concern (privacy, cost). A pilot found that one line at
the end of a prompt raised the share of cited answers from 40% to 100% on a 7B model (10 questions, so a hypothesis, not a
result), that a 4B judge called every answer "supported", and that two of the 20 answers graded by hand were wrong
*because* they described the wrong code faithfully.

**Method in brief:** about 150 new questions over 8 repositories in 3 languages, written and labelled by people other than
the builder and frozen before any model is run; 4 small models x 2 prompts plus 3 interventions and an oracle-context
condition; about 560 answers graded by humans (one grader each, 30% by two); paired statistics with pre-stated
comparisons.

**Effort and time:** about 20 weeks; about 75 hours of laptop compute (an estimate); about 120 person-hours of human
effort across three people (an estimate).

**What it does not claim:** any effect on developer productivity or onboarding time (no user study; future work).

**Documents to bring** (all in `docs/` and `research/`): `RESEARCH_PLAN.md`, `REVIEW_2026-10-06.md`,
`PAPER_REFERENCE.md`, `STEP2_PROTOCOL_DRAFT.md`.

---

## 2. Questions to ask your supervisor

Bring a copy of this list and write the answers down; section 5 has a form for that.

| # | Question | Why it matters |
|---|---|---|
| 1 | Is this research question acceptable, and is the scope right for the programme's credits and deadline? | Everything is built on it. |
| 2 | Is the dissertation marked as "system plus evaluation", and what is the split between implementation and research? | The system already exists; examiners may weigh it differently from a research contribution. |
| 3 | **May I reword the registered project description?** (Proposed text: `RESEARCH_PLAN.md`, section 3.) May I add a subtitle to the title? Is there a formal change procedure? | Productivity and onboarding claims cannot be supported without a user study. |
| 4 | **What is the AI-assistance policy for dissertations?** What must I disclose, in what format, and what is not allowed (code, analysis scripts, question writing, text)? | This project was built with an AI assistant throughout. See section 4. |
| 5 | **Does recruiting two peers to write questions and grade answers need ethics approval?** Which form, which committee, how many weeks? | They would read public code and rate model answers; no personal data. Approval can take weeks, so start early if needed. |
| 6 | May the benchmark (questions, labels, grades) be published? Any data-management rules? | Planned release is part of Step 10. |
| 7 | Is there university compute (GPU, cluster) I can use? Is there a budget for one hosted model as an upper bound? | The 75 hours of compute fit a laptop; a GPU shortens them a lot. The hosted model is optional. |
| 8 | What are the milestone dates (proposal, interim report, final), and does 20 weeks fit? | The timeline assumes weeks counted from the first approval. |
| 9 | Can you suggest peers who could annotate, or a second reader? | At least three people are needed (you and two others). |
| 10 | What do examiners reward most: novelty, rigour, depth of analysis? | Decides what to cut first if time runs short. |
| 11 | Should the protocol be pre-registered (for example on the Open Science Framework) or only approved by you? | It makes the order of decisions verifiable. |

---

## 3. Rules to look up yourself

Find the official documents, do not rely on memory or on what anyone says they usually are:

- [ ] Policy on **generative AI and AI coding assistants** in assessed work (disclosure wording, permitted uses).
- [ ] **Ethics review** requirements for studies in which people produce or rate data, even without personal data.
- [ ] **Academic integrity** rules about authorship and what counts as your own work.
- [ ] **Data management** and open-data rules.
- [ ] **Dissertation format**: length, structure, referencing style, submission date.
- [ ] Procedure for **changing a registered title or description**.

Write the document names and dates next to each, so you can cite them in the thesis.

---

## 4. Draft statement of what the AI assistant did (edit and check before using)

This is a factual draft from the project's history, **for you to verify against your own memory and edit to your
university's required format**. It is not a finished disclosure.

**Done by an AI coding assistant (Claude Code), directed by you:**

- wrote the system's code (all of `src/`), the 980 automated tests and the evaluation scripts;
- wrote the project documentation, including the architecture and evaluation reports;
- wrote the development question sets (about 145 questions) and the blind sets on `httpx` and `jinja2`, and the labels;
- ran the evaluations and the first local-model run, and **graded the pilot answers by hand** (14 of 20 correct, 4 partly,
  2 wrong), so those grades are the assistant's, not an independent person's;
- reviewed the project, found bugs, and fixed them;
- wrote the research plan and the drafts in this folder.

**Done by you:** the idea and requirements; the choices of language (Python), interface (Streamlit) and providers
(Claude, OpenAI, Ollama); approval of every download and of publishing to GitHub; the decision to pursue research on
small local models; the decision to treat productivity and onboarding as motivation only.

**What this means for the thesis:** the research design, the independent questions, the human grading, the
interpretation and the writing should be yours, within the rules you find in section 3. State the above plainly.

---

## 5. Email you could adapt and send (written for you to change; I have not sent anything)

> Subject: Project proposal: AI Codebase Understanding System, a request to agree the research question
>
> Dear [Supervisor],
>
> My project is the AI Codebase Understanding System. A working version exists: it answers natural-language
> questions about a code repository and cites the files and line ranges it used, and it runs offline with small
> local language models. I would like to turn it into a research study, and I would value 30 minutes to agree the
> direction. The proposed question: *when small local language models answer questions about code, do they cite
> faithfully, and what makes them better?* A one-page summary and a draft plan are attached.
>
> There are three things I need your guidance on before going further:
>
> 1. The registered description says the tool will "improve developer productivity and reduce onboarding time". I have
>    no user study and propose to present this as the motivation, with a reworded description (attached). Is that
>    acceptable, and may I add a subtitle to the title?
> 2. The system was developed with substantial help from an AI coding assistant. I would like to know the
>    university's disclosure requirements and what is permitted, so that I describe my own contribution correctly.
> 3. I would like two peers to write questions and grade answers (public code, no personal data). Does this need
>    ethics approval?
>
> Kind regards,
> [Your name]

Check the tone and facts before you send it, and attach the one-page summary (section 1) rather than the whole plan if
you prefer.

---

## 6. Meeting notes form (fill in during or straight after the meeting)

| Item | Answer | Date and form (email, minutes) |
|---|---|---|
| Research question agreed (yes / changed to) | | |
| Reworded description approved (yes / changes) | | |
| Subtitle allowed (yes / no) | | |
| AI-assistance rules (summary and document name) | | |
| Ethics approval needed (yes / no; form; lead time) | | |
| Dataset release allowed | | |
| Compute or budget available | | |
| Milestone dates | | |
| Pre-registration (yes / no) | | |
| Annotators suggested | | |

**Step 0 is done when** the first four rows are answered in writing and you know the ethics rule. Tell me the answers
and I will update the plan (and the protocol draft) to match.
