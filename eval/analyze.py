r"""The study's analysis, exactly as the protocol fixes it (research plan, step 3.8; protocol section 10).

    python eval/analyze.py RESULTS_DIR --grading grading/HG-A grading/HG-B grading/HG-C grading/HG-D \
        --judges labels/gemma3_4b.jsonl labels/llama3.1_8b.jsonl labels/qwen2.5-coder_7b.jsonl --out analysis/

Inputs: the matrix results folder (``run_matrix.py``), the grading folders (``export_annotation.py``, after grading),
and the judges' labels (``judge_claims.py``). Writes ``report.md`` and the tables behind it as CSV files.

Primary comparisons (Holm-corrected together):

* **C1** P1 vs P0, all models, grounded: the paired difference in grounded rate, averaged over models, with a two-stage
  cluster bootstrap (repositories, then questions within them). The protocol names a mixed-effects logistic regression;
  see decision D9 in the protocol.
* **C2** P1 vs P0, 7B, fully correct: paired difference, non-inferior if the margin (``--margin``, D2) is excluded.
* **C3** P3 vs P1, 7B, on the questions with a graded P3 answer: citation precision and fully correct.
* **C4** each judge vs the main grader, claim labels: Cohen's kappa, tested against the H5 threshold for its size.
* **C5** 7B P1 answers graded not fully correct: share whose every claim is supported, tested against 50%.

Failed answers count as not grounded and not correct (intention to treat); every primary comparison is repeated
without them. Intervals are 95%; bootstraps use 10,000 resamples and ``--seed``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from codebase_ai.evaluation import load_questions

RESAMPLES = 10_000
SUPPORT = ("supported", "unsupported", "contradicted", "not a claim")
CORRECTNESS = ("correct", "partly correct", "wrong", "abstained", "hallucinated", "partial")
FULLY_CORRECT = {"correct", "abstained"}  # an abstention is the right answer to an unanswerable question
_WORST = {"supported": 0, "unsupported": 1, "contradicted": 2}


# --- statistics (pure; each has a check in tests/test_eval_scripts.py) ------------------------------------------------


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (family-wise error), in the input's keys."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return {key: adjusted[key] for key in pvalues}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs (b one way, c the other)."""
    return 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)


def cohen_kappa(first: list[str], second: list[str]) -> float:
    labels = sorted(set(first) | set(second))
    index = {label: i for i, label in enumerate(labels)}
    table = np.zeros((len(labels), len(labels)))
    for a, b in zip(first, second, strict=True):
        table[index[a], index[b]] += 1
    return kappa_from_table(table)


def kappa_from_table(table: np.ndarray) -> float:
    n = table.sum()
    observed = np.trace(table) / n
    expected = (table.sum(axis=0) * table.sum(axis=1)).sum() / n**2
    return float("nan") if expected == 1 else float((observed - expected) / (1 - expected))


def krippendorff_alpha_nominal(units: list[list[str]]) -> float:
    """Krippendorff's alpha for nominal labels; ``units`` lists the labels each item got (missing ones left out)."""
    units = [u for u in units if len(u) >= 2]
    labels = sorted({v for u in units for v in u})
    if len(labels) < 2:
        return float("nan")
    index = {label: i for i, label in enumerate(labels)}
    coincidences = np.zeros((len(labels), len(labels)))
    for unit in units:
        counts = np.bincount([index[v] for v in unit], minlength=len(labels))
        coincidences += (np.outer(counts, counts) - np.diag(counts)) / (len(unit) - 1)
    totals = coincidences.sum(axis=1)
    n = totals.sum()
    observed = coincidences.sum() - np.trace(coincidences)
    expected = n**2 - (totals**2).sum()
    return float(1 - (n - 1) * observed / expected)


def two_stage_bootstrap(values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap means: resample the clusters (repositories), then the items (questions) within each one drawn."""
    groups = [values[clusters == c] for c in np.unique(clusters)]
    means = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        drawn = [groups[g] for g in rng.integers(len(groups), size=len(groups))]
        means[i] = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in drawn]).mean()
    return means


def bootstrap_p(estimate: float, boot: np.ndarray, null: float, alternative: str) -> float:
    """p-value from the bootstrap distribution shifted to the null; ``greater`` and ``less`` are the alternative."""
    shift = boot - estimate
    if alternative == "two-sided":
        extreme = np.abs(shift) >= abs(estimate - null)
    elif alternative == "greater":
        extreme = shift >= estimate - null
    else:
        extreme = shift <= estimate - null
    return float((1 + extreme.sum()) / (len(boot) + 1))


def interval(boot: np.ndarray) -> tuple[float, float]:
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return float(low), float(high)


# --- loading ------------------------------------------------------------------------------------------------------


def load_answers(results_dir: Path) -> pd.DataFrame:
    """One row per answer in a ``run_matrix.py`` results folder (``<repo>/<prompt>-<context>/<model>.json``)."""
    rows = []
    for path in sorted(results_dir.glob("*/*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        env = payload["env"]
        prompt, context = path.parent.name.split("-", 1)
        questions = {q.id: q for q in load_questions(env["questions_file"])}
        retrieval = env.get("retrieval", "live")
        frozen = {}
        if retrieval.startswith("frozen: "):
            frozen = json.loads(Path(retrieval.removeprefix("frozen: ")).read_text(encoding="utf-8"))["questions"]
        for row in payload["results"]:
            question = questions[row["id"]]
            supplied = [s["path"] for s in frozen.get(row["question"], {}).get("sources", [])]
            supplied = supplied[:3] if prompt == "P3" else supplied  # what P3 actually showed
            knowable = bool(frozen) and question.type != "unanswerable"
            rows.append({
                "results_file": str(path.resolve()), "repo": path.parent.parent.name, "model": env["model"],
                "prompt": prompt, "context": context, "question_id": row["id"], "type": question.type,
                "failed": row.get("error") is not None, "grounded": bool(row.get("grounded")),
                "precision": row.get("precision"), "gold_cited": bool(row.get("gold_cited")),
                "unverified_location": bool(row.get("unverified_locations")),
                "gold_in_context": any(p in question.gold_files for p in supplied) if knowable else None,
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["precision"] = pd.to_numeric(table["precision"])  # None (nothing cited) becomes NaN
    return table


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def load_grades(grading_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Answer grades and sentence-level claim labels from graded batches, joined to their runs through ``key.json``.

    Rows a grader split (3a, 3b) are folded back into their sentence: *not a claim* parts are dropped, and the sentence
    takes its worst remaining label (one unsupported part makes the sentence unsupported), so it can be compared with a
    judge's label for the same sentence. Unknown labels stop the analysis with the cells to fix.
    """
    answers, claims, problems = [], [], []
    for folder in grading_dirs:
        key = json.loads((folder / "key.json").read_text(encoding="utf-8"))["answers"]
        for grader in ("main", "second"):
            if not (folder / grader).exists():
                continue
            for line, row in enumerate(_read_csv(folder / grader / "answers.csv"), start=2):
                grade = row["correctness"].lower()
                if grade and grade not in CORRECTNESS:
                    problems.append(f"{folder / grader / 'answers.csv'}:{line}: correctness {row['correctness']!r}")
                elif grade:
                    answers.append({"grader": grader, "code": row["answer"], "correctness": grade,
                                    "flagged": bool(row["flag"]), **key[row["answer"]]})
            parts: dict[tuple[str, int], list[str]] = {}
            for line, row in enumerate(_read_csv(folder / grader / "claims.csv"), start=2):
                label = row["support"].lower()
                if label and label not in SUPPORT:
                    problems.append(f"{folder / grader / 'claims.csv'}:{line}: support {row['support']!r}")
                elif label:
                    parts.setdefault((row["answer"], int(re.match(r"\d+", row["claim"]).group())), []).append(label)
            for (code, sentence), labels in parts.items():
                real = [label for label in labels if label != "not a claim"]
                label = max(real, key=_WORST.get) if real else "not a claim"
                claims.append({"grader": grader, "code": code, "claim": sentence, "label": label, **key[code]})
    if problems:
        raise ValueError("Unknown labels in the grading sheets; fix these cells:\n  " + "\n  ".join(problems))
    columns = ["grader", "code", "results_file", "question_id"]
    return pd.DataFrame(answers, columns=[*columns, "correctness", "flagged"]), pd.DataFrame(claims, columns=[*columns, "claim", "label"])


def load_judges(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        judge = json.loads(lines[0])["_run"]["judge"].split("/", 1)[1]
        rows += [{"judge": judge, **row} for row in map(json.loads, lines[1:]) if row["error"] is None]
    return pd.DataFrame(rows, columns=["judge", "results_file", "question_id", "claim", "label", "by"])


# --- the comparisons -----------------------------------------------------------------------------------------------


def size_of(model: str) -> float:
    """Billions of parameters from an Ollama tag (``qwen2.5-coder:7b`` -> 7)."""
    found = re.search(r"(\d+(?:\.\d+)?)b", model.split(":")[-1])
    return float(found.group(1)) if found else float("nan")


def family_of(model: str) -> str:
    return re.match(r"[a-z]+", model.lower()).group()


def paired(table: pd.DataFrame, outcome: str, first: str, second: str, by: str = "prompt") -> pd.DataFrame:
    """One row per (model, question) with ``outcome`` under both conditions; failed answers count as 0 (ITT)."""
    if table.empty:
        return pd.DataFrame()
    wide = table.pivot_table(index=["model", "repo", "question_id"], columns=by, values=outcome, aggfunc="first")
    if not {first, second} <= set(wide.columns):  # one condition has no answers (or no grades) yet
        return pd.DataFrame()
    return wide.dropna(subset=[first, second]).reset_index()


def primary(answers: pd.DataFrame, grades: pd.DataFrame, claims: pd.DataFrame, judges: pd.DataFrame,
            margin: float, seed: int, drop_failed: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    frozen = answers[answers["context"] == "frozen"].copy()
    if drop_failed:
        failed = frozen[frozen["failed"]][["model", "repo", "question_id"]].drop_duplicates()
        frozen = frozen.merge(failed, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    frozen["grounded"] = frozen["grounded"] & ~frozen["failed"]

    # C1: grounded, P1 - P0, every model; per question, the mean over models; clusters = repositories.
    pairs = paired(frozen, "grounded", "P0", "P1")
    if len(pairs):
        pairs["diff"] = pairs["P1"].astype(float) - pairs["P0"].astype(float)
        per_question = pairs.groupby(["repo", "question_id"])["diff"].mean().reset_index()
        estimate = per_question["diff"].mean()
        boot = two_stage_bootstrap(per_question["diff"].to_numpy(), per_question["repo"].to_numpy(), rng)
        rows.append({"id": "C1", "comparison": "P1 vs P0, every model: grounded (difference in rate)",
                     "n": f"{len(per_question)} questions x {pairs['model'].nunique()} models", "estimate": estimate,
                     "ci": interval(boot), "p": bootstrap_p(estimate, boot, 0.0, "two-sided"), "test": "two-sided, = 0"})

    main = grades[grades["grader"] == "main"].merge(answers, on=["results_file", "question_id"])
    main["correct"] = main["correctness"].isin(FULLY_CORRECT)
    if not drop_failed:  # intention to treat: a failed answer was never graded, and is not correct
        failed = answers[answers["failed"]].assign(correct=False)
        main = pd.concat([main, failed[failed["question_id"].isin(main["question_id"])]], ignore_index=True)
    seven = main[(main["model"].map(size_of) == 7) & (main["context"] == "frozen")]

    # C2: fully correct, P1 - P0, 7B; non-inferior if the difference is above the margin.
    pairs = paired(seven, "correct", "P0", "P1")
    if len(pairs):
        diff = pairs["P1"].astype(float) - pairs["P0"].astype(float)
        boot = two_stage_bootstrap(diff.to_numpy(), pairs["repo"].to_numpy(), rng)
        rows.append({"id": "C2", "comparison": f"P1 vs P0, 7B: fully correct (non-inferior if above {margin:+.0%})",
                     "n": f"{len(pairs)} questions", "estimate": diff.mean(), "ci": interval(boot),
                     "p": bootstrap_p(diff.mean(), boot, margin, "greater"), "test": f"one-sided, > {margin:+.2f}"})

    # C3: P3 vs P1, 7B, on the questions with a graded P3 answer: citation precision, and fully correct.
    graded_p3 = seven[seven["prompt"] == "P3"]["question_id"]
    precision = paired(frozen[(frozen["model"].map(size_of) == 7) & frozen["question_id"].isin(graded_p3)],
                       "precision", "P1", "P3")
    if len(precision):
        diff = precision["P3"].astype(float) - precision["P1"].astype(float)
        boot = two_stage_bootstrap(diff.to_numpy(), precision["repo"].to_numpy(), rng)
        rows.append({"id": "C3a", "comparison": "P3 vs P1, 7B: citation precision (difference in mean)",
                     "n": f"{len(precision)} questions", "estimate": diff.mean(), "ci": interval(boot),
                     "p": bootstrap_p(diff.mean(), boot, 0.0, "two-sided"), "test": "two-sided, = 0"})
    pairs = paired(seven, "correct", "P1", "P3")
    if len(pairs):
        p1, p3 = pairs["P1"].astype(bool), pairs["P3"].astype(bool)  # a pivot leaves objects, where ~True is -2
        better, worse = int((p3 & ~p1).sum()), int((~p3 & p1).sum())
        diff = pairs["P3"].astype(float) - pairs["P1"].astype(float)
        boot = two_stage_bootstrap(diff.to_numpy(), pairs["repo"].to_numpy(), rng)
        rows.append({"id": "C3b", "comparison": "P3 vs P1, 7B: fully correct (difference in rate)",
                     "n": f"{len(pairs)} questions", "estimate": diff.mean(), "ci": interval(boot),
                     "p": mcnemar_exact(better, worse), "test": "exact McNemar"})

    # C4: each judge against the main grader's sentence labels (rows the grader marked "not a claim" left out).
    human = claims[(claims["grader"] == "main") & (claims["label"] != "not a claim")]
    for judge, labels in judges.groupby("judge"):
        joined = human.merge(labels, on=["results_file", "question_id", "claim"], suffixes=("_human", "_judge"))
        if joined.empty:
            continue
        kappa, boot = kappa_with_bootstrap(joined, rng)
        threshold = 0.4 if size_of(judge) <= 4 else 0.7
        rows.append({"id": f"C4 {judge}", "comparison": f"judge {judge} vs main grader: claim labels (Cohen's kappa)",
                     "n": f"{len(joined)} claims, {joined['code'].nunique()} answers", "estimate": kappa,
                     "ci": interval(boot), "p": bootstrap_p(kappa, boot, threshold, "less"),
                     "test": f"one-sided, < {threshold} (H5)"})

    # C5: 7B P1 answers graded not fully correct: share whose every claim is supported (faithful), against 50%.
    wrong = seven[(seven["prompt"] == "P1") & ~seven["correct"].astype(bool) & seven["code"].notna()]
    faithful = human.groupby("code")["label"].apply(lambda s: bool((s == "supported").all()))
    found = faithful.reindex(wrong["code"]).dropna()
    if len(found):
        k, n = int(found.sum()), len(found)
        test = stats.binomtest(k, n, 0.5, alternative="greater")
        ci = test.proportion_ci(method="exact")
        rows.append({"id": "C5", "comparison": "7B P1, not fully correct: share faithful to the cited code",
                     "n": f"{n} answers", "estimate": k / n, "ci": (ci.low, ci.high), "p": float(test.pvalue),
                     "test": "exact binomial, > 0.5"})

    adjusted = holm({row["id"]: row["p"] for row in rows})
    for row in rows:
        row["p_holm"] = adjusted[row["id"]]
    return rows


def kappa_with_bootstrap(joined: pd.DataFrame, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Cohen's kappa over claims, with a bootstrap over answers (claims of one answer are not independent)."""
    labels = list(_WORST)
    index = {label: i for i, label in enumerate(labels)}
    tables = {}
    for code, group in joined.groupby("code"):
        table = np.zeros((len(labels), len(labels)))
        for a, b in zip(group["label_human"], group["label_judge"], strict=True):
            table[index[a], index[b]] += 1
        tables[code] = table
    stacked = np.stack(list(tables.values()))
    boot = np.array([kappa_from_table(stacked[rng.integers(len(stacked), size=len(stacked))].sum(axis=0))
                     for _ in range(RESAMPLES)])
    return kappa_from_table(stacked.sum(axis=0)), boot


# --- exploratory tables ---------------------------------------------------------------------------------------------


def by_condition(answers: pd.DataFrame) -> pd.DataFrame:
    """Automatic measures for every model and condition (all answers; failed ones count as not grounded)."""
    table = answers.assign(grounded=answers["grounded"] & ~answers["failed"]).groupby(["model", "prompt", "context"])
    return table.agg(answers=("question_id", "size"), failed=("failed", "sum"), grounded=("grounded", "mean"),
                     precision=("precision", "mean"), gold_cited=("gold_cited", "mean"),
                     invented_location=("unverified_location", "mean")).reset_index()


def per_model_gain(answers: pd.DataFrame) -> pd.DataFrame:
    """H1 per model: grounded under P0 and P1, the gain, and an exact McNemar test (exploratory)."""
    frozen = answers[answers["context"] == "frozen"].assign(grounded=lambda t: t["grounded"] & ~t["failed"])
    rows = []
    for model, group in paired(frozen, "grounded", "P0", "P1").groupby("model"):
        p0, p1 = group["P0"].astype(bool), group["P1"].astype(bool)
        rows.append({"model": model, "size_b": size_of(model), "questions": len(group), "P0": p0.mean(), "P1": p1.mean(),
                     "gain": p1.mean() - p0.mean(), "p_mcnemar": mcnemar_exact(int((p1 & ~p0).sum()), int((p0 & ~p1).sum()))})
    return pd.DataFrame(rows).sort_values("size_b") if rows else pd.DataFrame()


def error_sources(answers: pd.DataFrame, grades: pd.DataFrame) -> pd.DataFrame:
    """RQ4: of the answers graded not fully correct, how many had a gold file in the code they were given."""
    main = grades[grades["grader"] == "main"].merge(answers, on=["results_file", "question_id"])
    wrong = main[~main["correctness"].isin(FULLY_CORRECT) & main["gold_in_context"].notna()]
    table = wrong.groupby(["model", "prompt", "context"])["gold_in_context"]
    return table.agg(not_fully_correct="size", generation_failures="sum").reset_index().assign(
        retrieval_failures=lambda t: t["not_fully_correct"] - t["generation_failures"])


def grader_agreement(grades: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, table, unit, value in (("correctness", grades, ["code"], "correctness"),
                                     ("claim labels", claims, ["code", "claim"], "label")):
        both = table.pivot_table(index=unit, columns="grader", values=value, aggfunc="first").dropna()
        if {"main", "second"} <= set(both.columns) and len(both):
            rows.append({"what": name, "items": len(both),
                         "cohen_kappa": cohen_kappa(list(both["main"]), list(both["second"])),
                         "krippendorff_alpha": krippendorff_alpha_nominal(both[["main", "second"]].values.tolist())})
    return pd.DataFrame(rows)


def judge_details(claims: pd.DataFrame, judges: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    """Per judge: agreement over all claims and over cited claims only, the share it calls supported, same family."""
    human = claims[(claims["grader"] == "main") & (claims["label"] != "not a claim")]
    joined = human.merge(judges, on=["results_file", "question_id", "claim"], suffixes=("_human", "_judge"))
    joined = joined.merge(answers[["results_file", "question_id", "model"]], on=["results_file", "question_id"])
    rows = []
    for judge, group in joined.groupby("judge"):
        for subset, part in (("all claims", group), ("cited claims", group[group["by"] == "judge"]),
                             ("same family as the answerer", group[group["model"].map(family_of) == family_of(judge)])):
            if len(part):
                rows.append({"judge": judge, "subset": subset, "claims": len(part),
                             "accuracy": float((part["label_human"] == part["label_judge"]).mean()),
                             "kappa": cohen_kappa(list(part["label_human"]), list(part["label_judge"])),
                             "judge_says_supported": float((part["label_judge"] == "supported").mean()),
                             "human_says_supported": float((part["label_human"] == "supported").mean())})
    return pd.DataFrame(rows)


# --- report ----------------------------------------------------------------------------------------------------------


def primary_table(rows: list[dict]) -> str:
    lines = ["| | Comparison | n | Estimate | 95% interval | Test | p | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['comparison']} | {r['n']} | {r['estimate']:+.3f} | "
                     f"{r['ci'][0]:+.3f} to {r['ci'][1]:+.3f} | {r['test']} | {r['p']:.4f} | {r['p_holm']:.4f} |")
    return "\n".join(lines)


def markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No data yet._"
    head = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for row in table.itertuples(index=False)]
    return "\n".join([head, *body])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--grading", nargs="*", type=Path, default=[], help="graded folders written by export_annotation.py")
    parser.add_argument("--judges", nargs="*", type=Path, default=[], help="label files written by judge_claims.py")
    parser.add_argument("--margin", type=float, default=-0.10, help="non-inferiority margin for C2 (D2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    answers = load_answers(args.results_dir)
    if answers.empty:
        print(f"error: no result files under {args.results_dir}", file=sys.stderr)
        return 2
    try:
        grades, claims = load_grades(args.grading)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    judges = load_judges(args.judges)

    itt = primary(answers, grades, claims, judges, args.margin, args.seed, drop_failed=False)
    complete = primary(answers, grades, claims, judges, args.margin, args.seed, drop_failed=True)
    tables = {
        "by_condition": by_condition(answers), "per_model_gain": per_model_gain(answers),
        "error_sources": error_sources(answers, grades), "grader_agreement": grader_agreement(grades, claims),
        "judges": judge_details(claims, judges, answers),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
    pd.DataFrame([{**r, "ci_low": r["ci"][0], "ci_high": r["ci"][1]} for r in itt]).drop(columns="ci", errors="ignore").to_csv(
        args.out / "primary.csv", index=False)

    report = [
        "# Analysis", "",
        (f"Generated by `eval/analyze.py` from {len(answers)} answers, {len(grades)} answer grades, {len(claims)} "
         f"graded sentences and {len(judges)} judge labels (seed {args.seed}, {RESAMPLES} bootstrap resamples)."), "",
        "## Primary comparisons (failed answers count as failures)", "", primary_table(itt) if itt else "_No data yet._", "",
        "## The same, without failed answers", "", primary_table(complete) if complete else "_No data yet._", "",
        "## Exploratory", "",
        "### Automatic measures by model and condition", "", markdown(tables["by_condition"]), "",
        "### Grounded rate, P0 vs P1, per model (H1)", "", markdown(tables["per_model_gain"]), "",
        "### Where the not-fully-correct answers went wrong (RQ4)", "", markdown(tables["error_sources"]), "",
        "### Agreement between the two graders", "", markdown(tables["grader_agreement"]), "",
        "### Judges against the main grader", "", markdown(tables["judges"]), "",
    ]
    (args.out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {args.out / 'report.md'} and {len(tables) + 1} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
