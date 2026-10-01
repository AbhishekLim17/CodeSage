"""The scripts in eval/ are not a package; load them by path so their logic can be tested."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from codebase_ai.evaluation import EvalReport, QuestionResult
from codebase_ai.retrieval.retriever import Retriever

EVAL_DIR = Path(__file__).parent.parent / "eval"


def load_script(name: str):
    sys.path.insert(0, str(EVAL_DIR))  # run_eval imports bench_embeddings by plain name
    try:
        spec = importlib.util.spec_from_file_location(name, EVAL_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EVAL_DIR))


@pytest.fixture(scope="module")
def run_eval():
    return load_script("run_eval")


@pytest.fixture(scope="module")
def bench():
    return load_script("bench_embeddings")


def test_every_config_states_its_test_penalty_explicitly(run_eval):
    """Regression: the baselines silently picked up the library's new default penalty and matched the demoted rows."""
    configs = run_eval.configurations([0.5, 2.0], 0.5)
    assert all("test_penalty" in kwargs for _, kwargs in configs)
    by_name = dict(configs)
    for baseline in ("vector", "keyword", "hybrid", "hybrid kw=0.5", "hybrid kw=2"):
        assert by_name[baseline]["test_penalty"] == 1.0, baseline
    for demoted in ("vector + demote tests", "hybrid + demote tests", "hybrid kw=0.5 + demote tests"):
        assert by_name[demoted]["test_penalty"] == 0.5, demoted


def test_baselines_do_not_depend_on_the_library_default(run_eval):
    default_penalty = inspect.signature(Retriever.__init__).parameters["test_penalty"].default
    assert default_penalty < 1.0, "this test documents why configs must be explicit: the default demotes tests"
    baseline = dict(run_eval.configurations([0.5], 0.5))["vector"]
    assert baseline["test_penalty"] != default_penalty


def test_configurations_are_valid_retriever_arguments(run_eval):
    accepted = set(inspect.signature(Retriever.__init__).parameters) - {"self", "index", "embedder"}
    for name, kwargs in run_eval.configurations([0.5, 2.0], 0.5):
        assert set(kwargs) <= accepted, name
        assert kwargs["mode"] in ("vector", "keyword", "hybrid"), name


def test_demotion_configs_can_be_skipped(run_eval):
    names = [n for n, _ in run_eval.configurations([0.5], 1.0)]
    assert not any("demote" in n for n in names)


def test_paired_pairs_only_reference_existing_configs(run_eval):
    names = {n for n, _ in run_eval.configurations([0.5, 2.0], 0.5)}
    for a, b in run_eval.PAIRS:
        assert a in names and b in names, (a, b)


def report(name, ranks):
    return EvalReport(name, [QuestionResult(f"q{i}", "explain", r, r, True, True, None, 1, 1, 1.0, ()) for i, r in enumerate(ranks)])


def test_the_paired_table_flags_only_real_differences(run_eval):
    vector = report("vector", [1] * 12)
    worse = report("hybrid", [3] * 12)
    same = report("keyword", [1] * 12)
    table = run_eval.paired_table([vector, worse, same])
    lines = {line.split("|")[1].strip(): line for line in table.splitlines() if line.startswith("| ") and " vs " in line[:40]}
    assert "(*)" in lines["hybrid vs vector"] and "-0.667" in lines["hybrid vs vector"]
    assert "(*)" not in lines["keyword vs vector"] and "+0.000" in lines["keyword vs vector"]
    assert "12 / 0 / 0" not in table and "0 / 12 / 0" in table


def test_regenerating_a_report_keeps_the_hand_written_analysis(bench):
    old = "# T\n\nold table\n\n## Reading the numbers\n\nMY HAND WRITTEN NOTES\n"
    new = "# T\n\nNEW table\n\n## Reading the numbers\n\n_(placeholder)_\n"
    merged = bench.keep_analysis(old, new)
    assert "NEW table" in merged and "MY HAND WRITTEN NOTES" in merged
    assert "old table" not in merged and "placeholder" not in merged
    assert bench.keep_analysis("no analysis section", new) == new


@pytest.fixture(scope="module")
def overview_eval():
    return load_script("run_overview_eval")


def _result(qid, *, coverage, overview, in_context=True, tokens=100):
    return QuestionResult(qid, "overview", 1, 1, in_context, in_context, None, tokens, 1, 1.0, (), area_coverage=coverage, overview=overview)


def test_the_overview_summary_averages_only_what_was_measured(overview_eval):
    summary = overview_eval.summarize(
        EvalReport("x", [_result("a", coverage=1.0, overview=True), _result("b", coverage=None, overview=False, in_context=False)])
    )
    assert summary["questions"] == 2 and summary["overview_rate"] == 0.5
    assert summary["area_coverage"] == 1.0  # the question without gold_dirs is left out, not counted as zero
    assert summary["in_context"] == 0.5


def test_the_overview_report_names_the_questions_and_the_classifier_false_positives(overview_eval):
    from codebase_ai.evaluation import EvalQuestion

    questions = [EvalQuestion("o1", "What does this do?", "overview", ("README.md",))]
    suite = {
        "name": "demo",
        "questions": questions,
        "plain": EvalReport("plain retrieval", [_result("o1", coverage=0.5, overview=False)]),
        "with_map": EvalReport("with repository map", [_result("o1", coverage=1.0, overview=True, tokens=300)]),
    }
    env = {"date": "2026-01-01", "machine": "test", "embedding_model": "fake"}
    text = overview_eval.render([suite], ["q7: How does X work?"], 85, env)
    assert "| demo | plain retrieval | 1 | 0% | 50% |" in text
    assert "| demo | with repository map | 1 | 100% | 100% |" in text
    assert "demo/o1: What does this do? | yes | 100%" in text
    assert "1 of 85 labelled non-overview questions" in text and "- q7: How does X work?" in text


@pytest.fixture(scope="module")
def rerank_eval():
    return load_script("run_rerank_eval")


def test_the_rerank_configs_are_the_baseline_first_then_one_per_model(rerank_eval):
    names = rerank_eval.config_names(["org/small", "org/big"], 20)
    assert names[0] == rerank_eval.BASELINE == "vector + demote tests"
    assert names[1:] == [
        "vector + demote tests + rerank (org/small, top 20)",
        "vector + demote tests + rerank (org/big, top 20)",
    ]


def test_the_rerank_report_compares_each_config_with_the_baseline(rerank_eval):
    base = report("vector + demote tests", [3, 2, 5, 1, 4, 2])
    better = report("vector + demote tests + rerank (m, top 30)", [1, 1, 2, 1, 1, 1])
    rows = rerank_eval.paired_rows([base, better])
    assert "| vector + demote tests + rerank (m, top 30) | 6 | +" in rows
    assert "5 / 0 / 1" in rows  # five questions better, none worse, one tied
    assert rows.count("\n| ") == 1  # a single comparison row (the baseline is not compared with itself)


def test_the_pooled_rerank_reports_keep_every_question_once(rerank_eval):
    suites = [
        {"name": "a", "reports": [report("base", [1, 2]), report("rr", [1, 1])]},
        {"name": "b", "reports": [report("base", [3]), report("rr", [2])]},
    ]
    combined = rerank_eval.pooled(suites)
    assert [r.name for r in combined] == ["base", "rr"]
    assert [len(r.results) for r in combined] == [3, 3]


# --- eval/run_answers.py --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_answers():
    return load_script("run_answers")


@pytest.fixture
def answers_setup(run_answers, monkeypatch, sample_repo, tmp_path, fake_embedder):
    """A built index of the sample repo, a two-question file, and scripted providers in place of real ones."""
    import json

    from codebase_ai.index.indexer import Indexer, RepoIndex
    from helpers import ScriptedProvider

    index_root = tmp_path / "answers-index"
    index = RepoIndex(sample_repo, index_root)
    try:
        Indexer(index, fake_embedder).run()
    finally:
        index.close()
    questions = tmp_path / "questions.jsonl"
    rows = [
        {"id": "a1", "type": "explain", "question": "shopping cart total price", "gold_files": ["web/cart.js"]},
        {"id": "a2", "type": "locate", "question": "start the server", "gold_files": ["go/server.go"]},
    ]
    questions.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    class Setup:
        generator = ScriptedProvider("It is computed here [1].", name="answerer", model="gen-1")
        judge = ScriptedProvider(json.dumps({"verdict": "supported", "unsupported_claims": [], "reason": "ok"}), name="judge", model="j-1")
        repo = sample_repo
        questions_file = questions
        root = index_root
        out = tmp_path / "out" / "answers.json"

        def args(self, *extra):
            return [str(self.repo), str(self.questions_file), "--index-root", str(self.root), "--out-json", str(self.out), *extra]

    setup = Setup()
    providers = {"answerer": setup.generator, "judge": setup.judge}

    def create(settings, *, provider=None, model=None):
        return providers[provider or "answerer"]

    monkeypatch.setattr(run_answers, "create_provider", create)
    monkeypatch.setattr(run_answers, "create_embedder", lambda settings: fake_embedder)
    return setup


def test_a_run_answers_scores_writes_results_and_a_hand_check_sample(run_answers, answers_setup, capsys):
    import json

    code = run_answers.main(answers_setup.args("--judge-provider", "judge", "--export-sample", "2", "--yes"))
    output = capsys.readouterr().out
    assert code == 0
    assert "Grounded: cites at least one supplied source | 100%" in output and "a1: shopping cart total price" in output
    payload = json.loads(answers_setup.out.read_text(encoding="utf-8"))
    assert payload["summary"]["questions"] == 2 and payload["summary"]["judged"] == 2
    assert payload["env"]["provider"] == "answerer" and payload["env"]["judge"] == "judge/j-1"
    sample = answers_setup.out.with_name("answers_sample.md").read_text(encoding="utf-8")
    assert "## a1" in sample and "supported" not in sample.split("## a1", 1)[1]  # the judge's verdicts stay hidden


def test_hosted_providers_need_an_explicit_yes_and_nothing_is_called_without_it(run_answers, answers_setup, capsys):
    answers_setup.generator.sends_code_off_machine = True
    code = run_answers.main(answers_setup.args())
    captured = capsys.readouterr()
    assert code == 2 and "Not started: re-run with --yes" in captured.err
    assert "Retrieved code from the repository will be sent to: answerer" in captured.out
    assert answers_setup.generator.calls == [] and not answers_setup.out.exists()


def test_a_local_run_needs_no_confirmation(run_answers, answers_setup):
    assert run_answers.main(answers_setup.args()) == 0
    assert len(answers_setup.generator.calls) == 2


def test_the_plan_warns_when_the_judge_is_the_answering_model(run_answers, answers_setup, capsys):
    answers_setup.judge.name, answers_setup.judge.model = "answerer", "gen-1"
    run_answers.main(answers_setup.args("--judge-provider", "judge"))
    assert "same model as the answerer" in capsys.readouterr().out


def test_limit_and_types_narrow_the_questions(run_answers, answers_setup):
    assert run_answers.main(answers_setup.args("--limit", "1")) == 0
    assert len(answers_setup.generator.calls) == 1
    answers_setup.generator.calls.clear()
    assert run_answers.main(answers_setup.args("--types", "locate")) == 0
    assert answers_setup.generator.calls[0]["prompt"].endswith("Question: start the server")
    assert run_answers.main(answers_setup.args("--types", "doc")) == 2  # nothing selected


def test_a_missing_index_or_arguments_are_reported(run_answers, answers_setup, tmp_path, capsys):
    assert run_answers.main([str(answers_setup.repo), str(answers_setup.questions_file), "--index-root", str(tmp_path / "none")]) == 1
    assert "codebase-ai index" in capsys.readouterr().err
    assert run_answers.main([]) == 2


def test_a_rejected_key_is_reported_and_stops_the_run(run_answers, answers_setup, capsys):
    from codebase_ai.llm.base import ProviderError

    answers_setup.generator.error = ProviderError("Key rejected.", "auth", provider="answerer")
    assert run_answers.main(answers_setup.args()) == 1
    assert "Key rejected." in capsys.readouterr().err
    assert len(answers_setup.generator.calls) == 1


def test_the_judge_can_be_checked_against_a_human(run_answers, answers_setup, tmp_path, capsys):
    import json

    run_answers.main(answers_setup.args("--judge-provider", "judge"))
    capsys.readouterr()
    human = tmp_path / "human.json"
    human.write_text(json.dumps({"a1": "supported", "a2": "unsupported"}), encoding="utf-8")
    assert run_answers.main(["--check-sample", str(human), "--results", str(answers_setup.out)]) == 0
    assert "Compared 2 answer(s): the judge and the human gave the same verdict on 1 (50%), and were within one step on 1" in capsys.readouterr().out
    empty = tmp_path / "none.json"
    empty.write_text(json.dumps({"zzz": "supported"}), encoding="utf-8")
    run_answers.main(["--check-sample", str(empty), "--results", str(answers_setup.out)])
    assert "No overlap" in capsys.readouterr().out
