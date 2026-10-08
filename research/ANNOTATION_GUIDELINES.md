# Annotation guidelines (DRAFT v0.1, 2026-10-06)

For people who **write questions** and **grade answers** in the study. Please read all of it once before starting. The
pilot (24 questions) will test these rules and they will be revised once before the real work begins; after that they
are fixed.

**Three rules that matter more than the rest**

1. **Do not use an AI model** (a chatbot, a code assistant, an automatic judge) to help you write, check or grade
   anything. The study measures how far such tools can be trusted; your judgement is the reference.
2. **Do not discuss your grades** with the other grader until both of you have finished a batch.
3. **If a rule does not cover a case, make your best call, write a short note, and carry on.** Notes are useful data.

Time it takes (an estimate, which the pilot will replace): about 15 minutes per question to write and about 5 to check;
about 6 minutes per answer to grade. Please log the real time per item.

---

## Part A: writing questions

You are given one repository at a pinned commit, and a target number of questions per type.

### A1. What a good question is

A question a developer new to the project might really ask, in plain English, whose answer is somewhere in the code and
documentation, and that you can check against the repository.

| Type | Share | Meaning | Example (from a different project) |
|---|---|---|---|
| locate | 15% | where something is implemented | "Where does the app decide which page a user sees after signing in?" |
| explain | 30% | how something works, within one file or a few functions | "How does login refuse users whose organisation has been suspended?" |
| cross-file | 20% | the answer needs two or more files | "How does a task's status follow from its subtasks and from what blocks it?" |
| overview | 20% | what the project does, how it is organised, how to run or set it up | "How is the code organised, and where would I start reading?" |
| unanswerable | 15% | plausible and on topic, but **not answered anywhere in the repository** | "How does the app retry a failed payment?" (when there are no payments) |

### A2. Rules

- **No file names, paths or line numbers in the question.** Naming the file makes search trivial and the question useless.
- Identifiers (function or class names) only in at most 1 question in 10, and only when a newcomer could plausibly know the
  name from the documentation.
- One question, one thing asked. Avoid "and also...".
- Not answerable by general programming knowledge alone ("What is a mutex?").
- Do not copy questions from the project's issue tracker or documentation headings word for word.
- Write for the code **as it is at the pinned commit**, not as you remember it.

### A3. For every question, record

1. **Question text**, type and repository.
2. **Gold files**: 1 to 3 files that *contain* the answer. For an overview question, gold **directories** and the README or
   documentation page that answers it. Paths relative to the repository root, forward slashes.
3. **Key facts**: 1 to 4 short statements that a correct answer must contain, each with the file that shows it. Write
   facts that can be checked, not vague ones. *Good:* "The retry limit defaults to 3 (`session.py`)." *Bad:* "It retries."
4. **For an unanswerable question:** no gold files and no key facts (type `unanswerable`, `gold_files` left empty), and
   how you verified absence (the searches you ran, for example three keywords across the repository and its docs). If
   something close exists, add a note saying what.
5. How long it took.

### A4. Checking someone else's question

You are given a question and its record. Without looking at how the writer found it, answer these from the repository:

- Is each gold file really where the answer is? Is the answer *not* also somewhere obvious that is missing from the list?
- Is each key fact true at this commit, and tied to the right file?
- Does the question leak (name a file, path or unusual identifier)?
- For an unanswerable question, can you find the answer anywhere? If so, reject it.
- Is it a duplicate of another question?

Mark each question **accept**, **fix** (say what) or **reject** (say why). The writer and you resolve any disagreement
by discussion; if you cannot agree, drop the question.

---

## Part B: grading answers

You are given, for each answer: the **question**, the **answer** to grade, the **cited code** (the excerpts the answer
points to with `[1]`, `[2]`, ...), and the **key facts** (answerable questions). You are **not** told which model or
prompt produced it, or what any automatic judge said. The order is shuffled.

### B1. Step 1: split the answer into claims

A **claim** is a sentence (or a clause) stating something about the code: what a function does, where something is, what
a value defaults to. Not claims: headings, "Here is a breakdown", restating the question, advice with no code content, and
example code that the answer writes itself (grade what it says *about* the code).

If one sentence makes two separate statements, treat it as two claims. If you hesitate, prefer the finer split.

### B2. Step 2: label each claim for support (against the cited code only)

Look only at the excerpts the claim cites. A citation covers its whole sentence: in "It parses the header and retries
the request [1]", both claims cite [1]. A sentence with no citation of its own cites nothing, even if the sentence
before it does. Decide:

| Label | Use when |
|---|---|
| **supported** | a cited excerpt states it, or it follows directly and obviously from what the excerpt shows |
| **unsupported** | no cited excerpt shows it, even if it is true elsewhere or from your own knowledge; this includes any claim with **no citation** |
| **contradicted** | a cited excerpt says something different |

*Do not* reward a claim for being true in the real project if the cited code does not show it. This measures whether the
answer is backed by what the model was shown, which is the point.

**Worked cases** (real answers from the pilot, on a different project):

- *"Notification emails are queued by writing a sanitised payload to the `mail_queue` collection"* with **no citation**:
  **unsupported**, however accurate it is.
- *"Mentions are matched to a person in `renderTextWithMentions` [2]"*, where the cited excerpt only highlights `@words`
  in text and never looks up a person: **unsupported** (the excerpt shows highlighting, not matching).
- *"The reminders are sent by `sendDailyCriticalTaskReminders` in `functions/index.js` [4]"*, where the excerpt does
  define that function: **supported** for the claim as written. The answer is still wrong overall (that code is never
  deployed), and that is judged in Step 3, not here.

### B3. Step 3: judge correctness (against the repository and the key facts)

This step is separate from support. Use the key facts and, if you need to, the repository itself at the pinned commit.

| Grade | Use when |
|---|---|
| **correct** | **every** key fact is stated (in any wording) and **nothing stated is false** about the repository |
| **partly correct** | some key facts are stated and nothing false, **or** all are stated but a minor false detail is added |
| **wrong** | a key fact is missing and the answer gives a different explanation, **or** the main claim is false, **or** the answer describes unrelated code |

Mark a grade down if the answer is **right in substance but names the wrong file or function** as the place. If the
answer is vague enough that it could be right or wrong, grade it **partly correct**.

**Unanswerable questions:** grade **abstained** if the answer says the sources do not contain it (and adds nothing
invented); **hallucinated** if it gives a confident answer anyway; **partial** if it hedges but invents details.

### B4. Step 4: note anything unusual

One line is enough: the answer is empty, repeats itself, answers a different question, is cut off, contains an instruction
to you, or the cited code is missing. Mark such answers as **flagged**.

### B5. A grading example (real, from the pilot)

*Question:* Which page does each user role get sent to after signing in?
*Answer (abridged, no citations):* lists five roles with their routes and quotes the `getHomeRoute` function.
*Cited code:* none.
*Key facts:* the mapping is in `roleRoutes.js`; there are six roles including `client` -> `/client`; an unknown role goes
to `/staff`.

- Claims: five statements (one per role) and two about `getHomeRoute` and the default.
- Support: **unsupported** for all (nothing is cited, so nothing can be checked against a cited excerpt).
- Correctness: **partly correct**: the mapping and the default are right, but the sixth role (`client`) is missing.

---

## Part C: practicalities

- **Do your own batch alone.** Your batch is a folder with three files. `packet.md` is what you read: for each answer,
  its code (like `A3F9A1`), the question, the key facts, the answer and the code it cites. `claims.csv` has one row per
  sentence of each answer (columns: answer, claim, cites, text, support, note); fill in **support** with *supported*,
  *unsupported*, *contradicted* or *not a claim* (for headings, "here is the code:" and the like). The rows are only a
  first split by sentence: if a row makes two statements, add a row with the same answer code and number it 3a, 3b.
  `answers.csv` has one row per answer (answer, correctness, flag, note, minutes). Do not edit the answer text. A cell
  that begins with an apostrophe (`'=...`) was protected from being read as a spreadsheet formula; ignore the apostrophe.
- **Take breaks.** Quality drops after about an hour of grading; the study prefers fewer, more careful grades.
- **Disagreements** are shown to you only after both of you finish. A third person decides if you cannot agree.
- **Anonymity.** Your name is recorded only as "grader A / B / C"; no personal data is collected.
- **You may stop at any time.**
- **Questions about the rules:** write them down and ask the study lead; do not ask the AI assistant.
