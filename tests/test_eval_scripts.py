"""The scripts in eval/ are not a package; load them by path so their logic can be tested."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from codebase_ai.evaluation import EvalReport, QuestionResult
from codebase_ai.rag.prompts import CITATION_REMINDER
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


def test_the_plan_states_the_output_cap_the_provider_really_applies(run_answers, answers_setup, capsys):
    # Seen for real: the plan promised 16000 output tokens from Ollama, which caps every answer at 4096.
    answers_setup.generator.max_output_tokens = 4096
    run_answers.main(answers_setup.args("--limit", "1"))
    assert "up to 4096 tokens of output" in capsys.readouterr().out


def test_limit_and_types_narrow_the_questions(run_answers, answers_setup):
    assert run_answers.main(answers_setup.args("--limit", "1")) == 0
    assert len(answers_setup.generator.calls) == 1
    answers_setup.generator.calls.clear()
    assert run_answers.main(answers_setup.args("--types", "locate")) == 0
    assert answers_setup.generator.calls[0]["prompt"].endswith(f"Question: start the server\n\n{CITATION_REMINDER}")
    assert run_answers.main(answers_setup.args("--types", "doc")) == 2  # nothing selected


def test_a_missing_index_or_arguments_are_reported(run_answers, answers_setup, tmp_path, capsys):
    assert run_answers.main([str(answers_setup.repo), str(answers_setup.questions_file), "--index-root", str(tmp_path / "none")]) == 1
    assert "codebase-ai index" in capsys.readouterr().err
    assert run_answers.main([]) == 2


# --- long runs: checkpoint and resume; frozen retrieval (research plan, steps 3.1 and 3.2) ---------------------------


def test_answers_survive_a_crash_and_the_run_resumes_where_it_stopped(run_answers, answers_setup, capsys):
    import json

    from codebase_ai.llm.base import ProviderError

    real_stream = answers_setup.generator.stream
    calls = []

    def stream(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ProviderError("The key was revoked.", "auth", provider="answerer")
        return real_stream(*args, **kwargs)

    answers_setup.generator.stream = stream
    assert run_answers.main(answers_setup.args()) == 1
    assert "--resume" in capsys.readouterr().err
    lines = answers_setup.out.with_suffix(".partial.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["id"] == "a1"  # the run's header, then the answer that finished

    answers_setup.generator.stream = real_stream
    answers_setup.generator.calls.clear()
    assert run_answers.main(answers_setup.args("--resume")) == 0
    assert len(answers_setup.generator.calls) == 1  # only the question that was not answered yet
    payload = json.loads(answers_setup.out.read_text(encoding="utf-8"))
    assert [r["id"] for r in payload["results"]] == ["a1", "a2"] and payload["summary"]["errors"] == 0


def test_resuming_retries_answers_that_failed(run_answers, answers_setup):
    import json

    from codebase_ai.llm.base import ProviderError

    answers_setup.generator.error = ProviderError("Ollama is not running.", "connection", provider="answerer", retryable=True)
    assert run_answers.main(answers_setup.args()) == 0  # a transient failure is recorded and the run goes on
    answers_setup.generator.error = None
    assert run_answers.main(answers_setup.args("--resume")) == 0
    summary = json.loads(answers_setup.out.read_text(encoding="utf-8"))["summary"]
    assert summary["errors"] == 0 and summary["grounded_rate"] == 1.0


def test_resuming_refuses_a_checkpoint_from_another_model(run_answers, answers_setup, capsys):
    assert run_answers.main(answers_setup.args("--limit", "1")) == 0
    answers_setup.generator.model = "gen-2"
    assert run_answers.main(answers_setup.args("--resume")) == 2  # mixing two models' answers would be silent and wrong
    assert "different run" in capsys.readouterr().err


def test_a_prompt_condition_is_applied_recorded_and_kept_apart_on_resume(run_answers, answers_setup, capsys):
    import json

    assert run_answers.main(answers_setup.args("--limit", "1", "--prompt", "P0")) == 0
    assert answers_setup.generator.calls[0]["prompt"].endswith("Question: shopping cart total price")  # no reminder
    prompt = json.loads(answers_setup.out.read_text(encoding="utf-8"))["env"]["prompt"]
    assert prompt["name"] == "P0" and prompt["reminder"] is None and prompt["sources"] == "as retrieved"
    assert run_answers.main(answers_setup.args("--resume")) == 2  # a P1 run must not pick up P0 answers
    assert "different run" in capsys.readouterr().err


def test_fresh_model_reloads_for_every_answer_and_is_part_of_the_checkpoint(run_answers, answers_setup, capsys):
    import json

    answers_setup.generator.keep_alive = None  # as an Ollama provider has
    assert run_answers.main(answers_setup.args("--limit", "1", "--fresh-model")) == 0
    assert answers_setup.generator.keep_alive == 0
    assert json.loads(answers_setup.out.read_text(encoding="utf-8"))["env"]["fresh_model"] is True
    assert run_answers.main(answers_setup.args("--resume")) == 2  # answers from both modes must not be mixed
    assert "--fresh-model" in capsys.readouterr().err


def test_frozen_retrieval_gives_every_run_exactly_the_stored_sources(run_answers, answers_setup, tmp_path):
    import json

    frozen = tmp_path / "frozen.json"
    assert run_answers.main(answers_setup.args("--freeze-retrieval", str(frozen))) == 0
    assert answers_setup.generator.calls == []  # freezing never calls a model
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    stored = payload["questions"]["shopping cart total price"]
    assert stored["id"] == "a1" and stored["sources"][0]["path"] == "web/cart.js"
    assert payload["meta"]["mode"] == "vector" and payload["meta"]["budget_tokens"] > 0

    stored["sources"][0]["text"] = "FROZEN MARKER"  # the replay must use what was stored, not what the index says now
    frozen.write_text(json.dumps(payload), encoding="utf-8")
    assert run_answers.main(answers_setup.args("--frozen", str(frozen))) == 0
    assert "FROZEN MARKER" in answers_setup.generator.calls[0]["prompt"]
    assert json.loads(answers_setup.out.read_text(encoding="utf-8"))["env"]["retrieval"] == f"frozen: {frozen}"


@pytest.fixture
def matrix(answers_setup, monkeypatch, tmp_path, fake_embedder):
    """The experiment-matrix runner over the sample repo, with two scripted 'Ollama' models and a fake Ollama server."""
    from helpers import ScriptedProvider

    run_matrix = load_script("run_matrix")
    models = {name: ScriptedProvider("It is computed here [1].", name="ollama", model=name) for name in ("m-a", "m-b")}
    monkeypatch.setattr(run_matrix.run_answers, "create_provider", lambda settings, *, provider=None, model=None: models[model or "m-a"])
    monkeypatch.setattr(run_matrix.run_answers, "create_embedder", lambda settings: fake_embedder)
    pulled = {name: {"digest": f"sha-{name}", "parameter_size": "1B", "quantization_level": "Q4_K_M"} for name in models}
    monkeypatch.setattr(run_matrix, "ollama_state", lambda host: ("0.35.1", pulled))
    monkeypatch.setattr(run_matrix, "code_version", lambda: {"commit": "abc123", "uncommitted_changes": False})

    def config(condition='prompt = "P1"\ncontext = "frozen"'):
        path = tmp_path / "matrix.toml"
        path.write_text(
            f"results_dir = '{(tmp_path / 'matrix').as_posix()}'\n"
            f"index_root = '{answers_setup.root.as_posix()}'\n"
            'models = ["m-a", "m-b"]\n\n'
            f"[[repos]]\nname = \"sample\"\npath = '{answers_setup.repo.as_posix()}'\n"
            f"questions = '{answers_setup.questions_file.as_posix()}'\n\n"
            f"[[conditions]]\n{condition}\n",
            encoding="utf-8",
        )
        return str(path)

    class Matrix:
        pass

    m = Matrix()
    m.run, m.models, m.pulled, m.config, m.results = run_matrix, models, pulled, config, tmp_path / "matrix"
    return m


def test_a_dry_run_lists_the_jobs_by_model_and_names_missing_models(matrix, capsys):
    assert matrix.run.main([matrix.config(), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.index("m-a") < out.index("m-b") and "P1-frozen" in out and not matrix.results.exists()
    del matrix.pulled["m-b"]
    assert matrix.run.main([matrix.config(), "--dry-run"]) == 2
    assert "ollama pull m-b" in capsys.readouterr().err


def test_the_matrix_freezes_once_runs_every_job_logs_it_and_resumes(matrix):
    import json

    assert matrix.run.main([matrix.config()]) == 0
    assert (matrix.results / "sample" / "frozen_retrieval.json").exists()
    for name, provider in matrix.models.items():
        assert len(provider.calls) == 2  # both questions
        assert (matrix.results / "sample" / "P1-frozen" / f"{name}.json").exists()
    # The controlled comparison: both models were shown exactly the same code.
    assert matrix.models["m-a"].calls[0]["prompt"] == matrix.models["m-b"].calls[0]["prompt"]
    runs = [json.loads(line) for line in (matrix.results / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["model"], r["exit_code"]) for r in runs] == [("m-a", 0), ("m-b", 0)]
    assert runs[0]["model_info"]["digest"] == "sha-m-a" and runs[0]["code"]["commit"] == "abc123"
    assert runs[0]["ollama_version"] == "0.35.1" and "ollama_num_ctx" in runs[0]["settings"]
    # The values the model really got, not just the configured ones (Ollama narrows the code budget and output).
    frozen = json.loads((matrix.results / "sample" / "frozen_retrieval.json").read_text(encoding="utf-8"))
    assert runs[0]["applied"] == {"code_budget_tokens": frozen["meta"]["budget_tokens"], "max_output_tokens": 4096,
                                  "temperature": 0.0, "fresh_model": True}
    assert "--fresh-model" in runs[0]["argv"]  # D11: every answer from a freshly loaded model

    assert matrix.run.main([matrix.config()]) == 0  # running it again answers nothing twice
    assert [len(p.calls) for p in matrix.models.values()] == [2, 2]
    assert len((matrix.results / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_unknown_conditions_are_refused(matrix, capsys):
    assert matrix.run.main([matrix.config('prompt = "P9"\ncontext = "live"')]) == 2
    err = capsys.readouterr().err
    assert "unknown prompt 'P9'" in err and "unknown context 'live'" in err and not matrix.results.exists()


def test_the_oracle_context_is_frozen_separately_and_replayed(matrix):
    import json

    assert matrix.run.main([matrix.config('prompt = "P1"\ncontext = "oracle"\nmodels = ["m-b"]')]) == 0
    oracle = json.loads((matrix.results / "sample" / "frozen_oracle.json").read_text(encoding="utf-8"))
    assert oracle["meta"]["oracle"] is True and not (matrix.results / "sample" / "frozen_retrieval.json").exists()
    # "start the server" is labelled with go/server.go, so that is the only code the model was shown.
    assert "go/server.go" in matrix.models["m-b"].calls[1]["prompt"] and "web/cart.js" not in matrix.models["m-b"].calls[1]["prompt"]
    assert matrix.models["m-a"].calls == [] and (matrix.results / "sample" / "P1-oracle" / "m-b.json").exists()


def test_a_condition_can_be_limited_to_some_models_and_each_job_gets_its_prompt(matrix):
    p3_on_b_only = 'prompt = "P0"\ncontext = "frozen"\n\n[[conditions]]\nprompt = "P3"\ncontext = "frozen"\nmodels = ["m-b"]'
    assert matrix.run.main([matrix.config(p3_on_b_only)]) == 0
    assert [len(p.calls) for p in matrix.models.values()] == [2, 4]  # m-a: P0 only; m-b: P0 and P3
    assert not matrix.models["m-a"].calls[0]["prompt"].endswith(CITATION_REMINDER)  # P0 has no reminder
    assert matrix.models["m-b"].calls[2]["prompt"].endswith(CITATION_REMINDER)  # P3 is P1 with fewer sources
    p3 = matrix.results / "sample" / "P3-frozen"
    assert (p3 / "m-b.json").exists() and not (p3 / "m-a.json").exists()


def test_the_oracle_is_only_frozen_and_records_what_it_is(run_answers, answers_setup, tmp_path, capsys):
    import json

    assert run_answers.main(answers_setup.args("--oracle")) == 2  # answering goes through --frozen, never --oracle
    assert "--freeze-retrieval" in capsys.readouterr().err
    oracle = tmp_path / "oracle.json"
    assert run_answers.main(answers_setup.args("--freeze-retrieval", str(oracle), "--oracle")) == 0
    stored = json.loads(oracle.read_text(encoding="utf-8"))["questions"]["start the server"]["sources"]
    assert {s["path"] for s in stored} == {"go/server.go"}  # no other file, and no repository map
    assert run_answers.main(answers_setup.args("--frozen", str(oracle))) == 0
    assert "oracle: only each question's gold files" in capsys.readouterr().out


def test_a_question_without_frozen_retrieval_is_refused_before_anything_runs(run_answers, answers_setup, tmp_path, capsys):
    frozen = tmp_path / "frozen.json"
    assert run_answers.main(answers_setup.args("--limit", "1", "--freeze-retrieval", str(frozen))) == 0
    assert run_answers.main(answers_setup.args("--frozen", str(frozen))) == 2
    assert "no frozen retrieval" in capsys.readouterr().err and answers_setup.generator.calls == []


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


# --- eval/check_classifier.py ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def check_classifier():
    return load_script("check_classifier")


def test_the_classifier_checker_reports_recall_and_false_positives(check_classifier):
    result = check_classifier.measure(
        {
            "overview": ["What does this project do?", "Tell me something unusual about it."],
            "specific": ["Give me an overview of the cart component.", "Give me an overview of the codebase."],
        }
    )
    assert (result["recognised"], result["overview"]) == (1, 2)
    assert result["missed"] == ["Tell me something unusual about it."]
    assert [q for q, _ in result["flagged"]] == ["Give me an overview of the codebase."]  # a mislabelled case is shown
    text = check_classifier.report(result)
    assert "Recognised 1 of 2 overview questions (50%)." in text and "Wrongly flagged 1 of 2" in text
    assert "Tell me something unusual" in text


def test_the_classifier_checker_reads_files_and_rejects_bad_ones(check_classifier, tmp_path, capsys):
    good = tmp_path / "mine.json"
    good.write_text('{"overview": ["What does this project do?"], "specific": []}', encoding="utf-8")
    assert check_classifier.main([str(good)]) == 0
    assert "Recognised 1 of 1" in capsys.readouterr().out
    assert check_classifier.main([str(tmp_path / "missing.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text('["not", "an", "object"]', encoding="utf-8")
    assert check_classifier.main([str(bad)]) == 2
    assert check_classifier.main([]) == 0  # the shipped held-out set


# --- eval/scale_test.py ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scale_test():
    return load_script("scale_test")


def test_the_scale_test_generates_a_repository_and_measures_it(scale_test):
    row = scale_test.measure(12, real_rate=10.0)
    assert row["files"] == 12 and row["files_indexed"] == 12
    assert row["chunks"] >= 12 * scale_test.FUNCTIONS_PER_FILE
    assert row["rerun_seconds"] < row["index_seconds"]  # nothing changed, so nothing is re-embedded
    assert row["embedding_estimate_minutes"] == pytest.approx(row["chunks"] / 10.0 / 60)
    text = scale_test.render([row], 10.0)
    assert "| 12 |" in text and "extrapolated" in text


def test_the_scale_test_without_a_real_rate_makes_no_estimate(scale_test):
    row = scale_test.measure(3, real_rate=None)
    assert "embedding_estimate_minutes" not in row
    assert "extrapolated" not in scale_test.render([row], None)


def test_the_synthetic_repository_spreads_files_over_folders(scale_test, tmp_path):
    scale_test.make_repo(tmp_path, 120)
    assert len(list(tmp_path.rglob("*.py"))) == 120
    assert len({p.parent for p in tmp_path.rglob("*.py")}) > 10


# --- the annotation export (research plan, step 3.6) -----------------------------------------------------------------


@pytest.fixture(scope="module")
def export():
    return load_script("export_annotation")


@pytest.mark.parametrize(
    ("answer", "claims"),
    [
        ("It adds the prices [1]. Then it adds tax [2].", ["It adds the prices [1].", "Then it adds tax [2]."]),
        ("It adds the prices. [1][2] Then tax.", ["It adds the prices. [1][2]", "Then tax."]),  # the citation stays
        ("## Summary\n- First, `a()` runs [1].\n2. Then `b()`.", ["Summary", "First, `a()` runs [1].", "Then `b()`."]),
        ("Calls a helper, e.g. `f()` [1].", ["Calls a helper, e.g. `f()` [1]."]),
        ("Before:\n```python\nx = 1. y = 2.\n```\nAfter `self.client.send()` [3].", ["Before:", "After `self.client.send()` [3]."]),
    ],
)
def test_answers_are_split_into_one_claim_per_sentence_outside_code(export, answer, claims):
    assert export.split_claims(answer) == claims


@pytest.fixture
def graded_runs(tmp_path):
    """Two finished runs (prompts P0 and P1 of one model) over three questions, one answer failed."""
    import json

    questions = tmp_path / "q.jsonl"
    rows = [
        {"id": "q1", "type": "explain", "question": "How is the total worked out?", "gold_files": ["cart.py"],
         "key_facts": ["Prices times quantities are summed (cart.py)."]},
        {"id": "q2", "type": "locate", "question": "Where is tax added?", "gold_files": ["tax.py"]},
        {"id": "q3", "type": "explain", "question": "How are refunds made?", "gold_files": ["refund.py"]},
    ]
    questions.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def run(prompt, answers):
        results = [
            {"id": qid, "question": q["question"], "answer": text, "error": None if text else "timed out",
             "cited_code": [[1, "cart.py:1-3", "def total():\n    return sum(...)"]] if "[1]" in (text or "") else [],
             "judgement": {"verdict": "supported", "unsupported_claims": [], "reason": "JUDGE SAID"}}
            for (qid, text), q in zip(answers.items(), rows, strict=True)
        ]
        env = {"model": "secret-model-7b", "questions_file": str(questions), "retrieval": "frozen: f.json",
               "prompt": {"name": prompt, "system": "SYSTEM TEXT", "reminder": None, "sources": "as retrieved"}}
        path = tmp_path / f"{prompt}.json"
        path.write_text(json.dumps({"env": env, "summary": {}, "results": results}), encoding="utf-8")
        return path

    return [
        run("P0", {"q1": "It sums the lines [1]. Then it returns.", "q2": "=SUM(A1) is not used.", "q3": None}),
        run("P1", {"q1": "The total adds prices [1].", "q2": "- Tax is in `tax.py`.", "q3": "Refunds are not shown."}),
    ]


def read_csv(path):
    import csv

    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_the_export_is_blinded_complete_and_keyed(export, graded_runs, tmp_path, capsys):
    import json

    out = tmp_path / "hg"
    assert export.main([*map(str, graded_runs), "--out", str(out), "--second-share", "0.4"]) == 0
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))["answers"]
    assert len(key) == 5  # six answers, one failed
    assert {(v["prompt"], v["question_id"]) for v in key.values()} == {("P0", "q1"), ("P0", "q2"), ("P1", "q1"), ("P1", "q2"), ("P1", "q3")}

    main_answers = [r["answer"] for r in read_csv(out / "main" / "answers.csv")]
    second_answers = [r["answer"] for r in read_csv(out / "second" / "answers.csv")]
    assert sorted(main_answers) == sorted(key) and len(second_answers) == 2 and set(second_answers) <= set(key)
    for folder in ("main", "second"):  # nothing in a grader's files says which model or prompt wrote an answer
        text = "".join(p.read_text(encoding="utf-8-sig") for p in (out / folder).iterdir())
        for leak in ("secret-model", "P0", "P1", "JUDGE SAID", "SYSTEM TEXT", "frozen"):
            assert leak not in text, leak

    claims = read_csv(out / "main" / "claims.csv")
    by_text = {r["text"]: r for r in claims}
    assert by_text["It sums the lines [1]."]["cites"] == "1" and by_text["Then it returns."]["cites"] == ""
    assert "'=SUM(A1) is not used." in by_text  # a spreadsheet would run it as a formula otherwise
    assert "Tax is in `tax.py`." in by_text  # the list marker is not part of the claim
    packet = (out / "main" / "packet.md").read_text(encoding="utf-8")
    assert "Prices times quantities are summed (cart.py)." in packet and "**[1] cart.py:1-3**" in packet
    assert "Key facts:** none recorded" in packet and "keep" in capsys.readouterr().out.lower()


def test_the_same_seed_gives_the_same_batch_and_a_batch_is_never_overwritten(export, graded_runs, tmp_path, capsys):
    first, again = tmp_path / "a", tmp_path / "b"
    assert export.main([*map(str, graded_runs), "--out", str(first)]) == 0
    assert export.main([*map(str, graded_runs), "--out", str(again)]) == 0
    assert (first / "main" / "claims.csv").read_bytes() == (again / "main" / "claims.csv").read_bytes()
    assert export.main([*map(str, graded_runs), "--out", str(first)]) == 2
    assert "not empty" in capsys.readouterr().err


def test_a_sample_or_an_id_list_selects_questions_with_all_their_answers(export, graded_runs, tmp_path):
    import json

    assert export.main([*map(str, graded_runs), "--out", str(tmp_path / "s"), "--sample", "1", "--second-share", "0"]) == 0
    sampled = json.loads((tmp_path / "s" / "key.json").read_text(encoding="utf-8"))["answers"].values()
    assert len({v["question_id"] for v in sampled}) == 1 and not (tmp_path / "s" / "second").exists()
    ids = tmp_path / "ids.txt"
    ids.write_text("q1\n", encoding="utf-8")
    assert export.main([*map(str, graded_runs), "--out", str(tmp_path / "i"), "--ids", str(ids)]) == 0
    chosen = json.loads((tmp_path / "i" / "key.json").read_text(encoding="utf-8"))["answers"].values()
    assert sorted(v["prompt"] for v in chosen) == ["P0", "P1"]  # both answers to q1: the pair stays together


# --- the claim-level judge (research plan, step 3.7) -----------------------------------------------------------------


@pytest.fixture
def claim_judge(monkeypatch):
    """judge_claims.py with a scripted judge model in place of a real one."""
    from helpers import ScriptedProvider

    module = load_script("judge_claims")
    judge = ScriptedProvider('{"label": "Supported", "reason": "The excerpt shows it."}', name="ollama", model="judge-4b")
    judge.keep_alive = None  # as an Ollama provider has
    monkeypatch.setattr(module, "create_provider", lambda settings, *, provider=None, model=None: judge)
    module.judge = judge
    return module


def judged_rows(path):
    import json

    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0])["_run"], [json.loads(line) for line in lines[1:]]


def test_each_claim_is_judged_against_its_own_citations_and_uncited_claims_by_the_rule(claim_judge, graded_runs, tmp_path, capsys):
    out = tmp_path / "judged.jsonl"
    assert claim_judge.main([*map(str, graded_runs), "--judge-model", "judge-4b", "--out", str(out)]) == 0
    header, rows = judged_rows(out)
    assert header["judge"] == "ollama/judge-4b"
    assert header["fresh_model"] is True and claim_judge.judge.keep_alive == 0  # D11: reloaded for every claim
    # Six claims in the five answered questions; only the two that cite code went to the model.
    assert len(rows) == 6 and len(claim_judge.judge.calls) == 2
    by_text = {r["text"]: r for r in rows}
    assert (by_text["It sums the lines [1]."]["label"], by_text["It sums the lines [1]."]["by"]) == ("supported", "judge")
    assert (by_text["Then it returns."]["label"], by_text["Then it returns."]["by"]) == ("unsupported", "rule")
    prompt = claim_judge.judge.calls[0]["prompt"]
    assert "<claim>\nIt sums the lines [1].\n</claim>" in prompt and "[1] cart.py:1-3" in prompt
    assert "Then it returns." in prompt  # the whole answer is there for context
    assert all(Path(r["results_file"]).is_absolute() for r in rows)
    assert "Share judged supported: 100%" in capsys.readouterr().out


def test_only_the_excerpts_a_claim_cites_are_shown(claim_judge, tmp_path):
    import json

    results = tmp_path / "r.json"
    cited = [[1, "a.py:1-2", "def first(): ..."], [2, "b.py:5-9", "def second(): ..."]]
    row = {"id": "q1", "question": "How?", "answer": "Uses `first` [1]. Then `second` [2].", "error": None, "cited_code": cited}
    results.write_text(json.dumps({"env": {}, "results": [row]}), encoding="utf-8")
    assert claim_judge.main([str(results), "--judge-model", "judge-4b", "--out", str(tmp_path / "j.jsonl")]) == 0
    second = claim_judge.judge.calls[1]["prompt"].split("<cited_code>")[1]
    assert "b.py:5-9" in second and "a.py:1-2" not in second


def test_failed_claims_are_retried_on_the_next_run_and_another_judges_file_is_refused(claim_judge, graded_runs, tmp_path, capsys):
    out = tmp_path / "judged.jsonl"
    replies = iter(["no json here", '{"label": "contradicted", "reason": "x"}'])
    claim_judge.judge.reply = lambda prompt: next(replies)
    assert claim_judge.main([*map(str, graded_runs), "--judge-model", "judge-4b", "--out", str(out)]) == 0
    assert "retry" in capsys.readouterr().out
    claim_judge.judge.calls.clear()
    claim_judge.judge.reply = '{"label": "unsupported", "reason": "y"}'
    assert claim_judge.main([*map(str, graded_runs), "--judge-model", "judge-4b", "--out", str(out)]) == 0
    assert len(claim_judge.judge.calls) == 1  # only the claim whose reply could not be read
    _, rows = judged_rows(out)
    assert len(rows) == 6 and all(r["error"] is None for r in rows)
    claim_judge.judge.model = "another-judge"
    assert claim_judge.main([*map(str, graded_runs), "--judge-model", "another-judge", "--out", str(out)]) == 2


# --- the analysis package (research plan, step 3.8) ------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyze():
    return load_script("analyze")


def test_the_statistics_match_known_values(analyze):
    import numpy as np
    from sklearn.metrics import cohen_kappa_score

    assert analyze.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert analyze.mcnemar_exact(1, 8) == pytest.approx(20 / 512) and analyze.mcnemar_exact(0, 0) == 1.0
    rng = np.random.default_rng(1)
    first, second = list(rng.choice(["s", "u", "c"], 200)), list(rng.choice(["s", "u", "c"], 200))
    assert analyze.cohen_kappa(first, second) == pytest.approx(cohen_kappa_score(first, second))
    # Worked by hand: 4 items, one disagreement; 3 "a" and 5 "b" values, so alpha = 1 - 7 * 2 / 30.
    assert analyze.krippendorff_alpha_nominal([["a", "a"], ["a", "b"], ["b", "b"], ["b", "b"], ["a"]]) == pytest.approx(16 / 30)
    boot = np.random.default_rng(2).normal(0.3, 0.05, 10_000)
    assert analyze.bootstrap_p(0.3, boot, 0.0, "two-sided") < 0.001
    assert analyze.bootstrap_p(0.3, boot, 0.0, "greater") < 0.001 and analyze.bootstrap_p(0.3, boot, 0.0, "less") > 0.99
    assert analyze.size_of("qwen2.5-coder:7b") == 7 and analyze.family_of("llama3.1:8b") == "llama"


@pytest.fixture
def study(tmp_path, export):
    """A small finished study: two repositories, two models, P0/P1/P3, one failed answer, graded HG-A and HG-C
    batches and one judge's labels. Every primary estimate below is worked out by hand from these sets."""
    import csv
    import json

    results = tmp_path / "results"
    seven, four = "qwen2.5-coder:7b", "gemma3:4b"
    everything = {"a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"}
    grounded = {(seven, "P0"): {"a1", "a4", "b3"}, (seven, "P1"): everything, (seven, "P3"): everything,
                (four, "P0"): set(), (four, "P1"): {"a1", "a2", "a3", "a4"}}
    for repo, ids in (("r1", ["a1", "a2", "a3", "a4"]), ("r2", ["b1", "b2", "b3", "b4"])):
        questions = tmp_path / f"{repo}.jsonl"
        questions.write_text("\n".join(json.dumps({"id": q, "type": "explain", "question": f"question {q}",
                                                   "gold_files": ["src.py"]}) for q in ids), encoding="utf-8")
        frozen = results / repo / "frozen_retrieval.json"
        frozen.parent.mkdir(parents=True)
        stored = {f"question {q}": {"id": q, "sources": [{"path": "src.py"}, {"path": "other.py"}]} for q in ids}
        frozen.write_text(json.dumps({"meta": {}, "questions": stored}), encoding="utf-8")
        for (model, prompt), yes in grounded.items():
            rows = []
            for q in ids:
                failed = (model, prompt, q) == (four, "P1", "b4")
                g = q in yes and not failed
                rows.append({"id": q, "question": f"question {q}", "error": "timed out" if failed else None,
                             "answer": "" if failed else ("It works [1]. Then more." if g else "It works. Then more."),
                             "grounded": g, "precision": (1.0 if prompt == "P3" else 0.5) if g else None, "gold_cited": g,
                             "unverified_locations": 0, "cited_code": [[1, "src.py:1-2", "def f(): ..."]] if g else []})
            path = results / repo / f"{prompt}-frozen" / (model.replace(":", "_") + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            env = {"model": model, "questions_file": str(questions), "retrieval": f"frozen: {frozen}", "prompt": {"name": prompt}}
            path.write_text(json.dumps({"env": env, "results": rows}), encoding="utf-8")

    def grade(folder):  # the "graders": P0 wrong on a4, P1 wrong on b4; claim 2 of P1/b4 is not a claim
        key = json.loads((folder / "key.json").read_text(encoding="utf-8"))["answers"]
        for grader in ("main", "second"):
            if not (folder / grader).exists():
                continue
            answers = read_csv(folder / grader / "answers.csv")
            for row in answers:
                k = key[row["answer"]]
                row["correctness"] = "wrong" if (k["prompt"], k["question_id"]) in {("P0", "a4"), ("P1", "b4")} else "correct"
            claims = read_csv(folder / grader / "claims.csv")
            for row in claims:
                k = key[row["answer"]]
                if row["claim"] == "1":
                    row["support"] = "supported" if "[1]" in row["text"] else "unsupported"
                else:
                    row["support"] = "not a claim" if (k["prompt"], k["question_id"]) == ("P1", "b4") else "unsupported"
            for name, rows in (("answers.csv", answers), ("claims.csv", claims)):
                with (folder / grader / name).open("w", encoding="utf-8-sig", newline="") as f:
                    out = csv.DictWriter(f, fieldnames=list(rows[0]))
                    out.writeheader()
                    out.writerows(rows)

    hg_a, hg_c = tmp_path / "HG-A", tmp_path / "HG-C"
    seven_files = sorted(str(p) for p in results.glob("*/P[01]-frozen/qwen*.json"))
    assert export.main([*seven_files, "--out", str(hg_a), "--second-share", "0.5"]) == 0
    ids = tmp_path / "hg_c_ids.txt"
    ids.write_text("a1\na2\n", encoding="utf-8")
    assert export.main([*map(str, results.glob("*/P3-frozen/qwen*.json")), "--out", str(hg_c), "--ids", str(ids), "--second-share", "0"]) == 0
    grade(hg_a)
    grade(hg_c)

    judge = tmp_path / "judge.jsonl"  # a judge that agrees with the main grader on every claim
    rows = [json.dumps({"_run": {"judge": "ollama/llama3.1:8b", "system": "x"}})]
    for path in seven_files:
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["results"]:
            for n, text in enumerate(export.split_claims(row["answer"]), start=1):
                label = "supported" if "[1]" in text else "unsupported"
                rows.append(json.dumps({"results_file": str(Path(path).resolve()), "question_id": row["id"], "claim": n,
                                        "text": text, "label": label, "by": "judge" if "[1]" in text else "rule", "error": None}))
    judge.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"results": results, "grading": [hg_a, hg_c], "judges": [judge]}


def test_the_primary_comparisons_on_a_study_worked_out_by_hand(analyze, study):
    answers = analyze.load_answers(study["results"])
    grades, claims = analyze.load_grades(study["grading"])
    judges = analyze.load_judges(study["judges"])
    itt = {r["id"]: r for r in analyze.primary(answers, grades, claims, judges, -0.10, 0, drop_failed=False)}
    complete = {r["id"]: r for r in analyze.primary(answers, grades, claims, judges, -0.10, 0, drop_failed=True)}

    # C1: per question, the P1 - P0 gain averaged over the two models; 4.5 / 8, or 5 / 8 without gemma's failed b4.
    assert itt["C1"]["estimate"] == pytest.approx(4.5 / 8) and complete["C1"]["estimate"] == pytest.approx(5 / 8)
    assert itt["C1"]["ci"][0] <= itt["C1"]["estimate"] <= itt["C1"]["ci"][1]
    assert itt["C2"]["estimate"] == pytest.approx(0.0)  # one question better (a4), one worse (b4)
    assert itt["C3a"]["estimate"] == pytest.approx(0.5) and itt["C3b"]["p"] == 1.0  # precision 1.0 vs 0.5; same grades
    assert itt["C4 llama3.1:8b"]["estimate"] == pytest.approx(1.0) and "0.7" in itt["C4 llama3.1:8b"]["test"]
    assert itt["C5"]["estimate"] == 1.0 and itt["C5"]["n"] == "1 answers"  # b4's only claim is supported
    assert all(r["p_holm"] >= r["p"] for r in itt.values())

    gains = analyze.per_model_gain(answers).set_index("model")["gain"]
    assert gains["gemma3:4b"] == pytest.approx(0.5) and gains["qwen2.5-coder:7b"] == pytest.approx(5 / 8)
    sources = analyze.error_sources(answers, grades)
    assert sources["not_fully_correct"].sum() == 2 and sources["retrieval_failures"].sum() == 0  # gold was supplied
    assert analyze.grader_agreement(grades, claims).set_index("what").loc["correctness", "items"] == 8


def test_the_report_is_written_and_a_mistyped_grade_is_named(analyze, study, tmp_path, capsys):
    out = tmp_path / "analysis"
    args = [str(study["results"]), "--grading", *map(str, study["grading"]), "--judges", *map(str, study["judges"])]
    assert analyze.main([*args, "--out", str(out)]) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "| C1 |" in report and "without failed answers" in report and (out / "primary.csv").exists()
    sheet = study["grading"][0] / "main" / "answers.csv"
    sheet.write_text(sheet.read_text(encoding="utf-8-sig").replace(",correct,", ",corect,", 1), encoding="utf-8-sig")
    assert analyze.main([*args, "--out", str(tmp_path / "again")]) == 2
    assert "answers.csv:" in capsys.readouterr().err


def test_every_figure_is_drawn_from_the_analysis_tables(analyze, study, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    figures = load_script("figures")
    out = tmp_path / "analysis"
    args = [str(study["results"]), "--grading", *map(str, study["grading"]), "--judges", *map(str, study["judges"])]
    assert analyze.main([*args, "--out", str(out)]) == 0
    assert figures.main([str(out)]) == 0
    for name in ("primary", "grounded_by_model", "judges", "error_sources", "power"):
        for suffix in (".png", ".pdf"):
            assert (out / "figures" / f"{name}{suffix}").stat().st_size > 1000, name
    (out / "judges.csv").write_text("", encoding="utf-8")  # a table with no data yet is skipped, not an error
    assert figures.main([str(out), "--no-power"]) == 0 and "no data yet for: judges" in capsys.readouterr().out


def test_the_power_simulation_agrees_with_the_hand_estimate(tmp_path):
    import numpy as np

    power = load_script("power_simulation")
    rng = np.random.default_rng(0)
    # Protocol section 8 by hand: 150 questions, 20% disagreement, no true difference -> interval about +/-7 points,
    # so the lower end clears -10 most of the time (P(Z > -0.78) is about 78%).
    assert power.noninferiority_power(rng, 150, 0.20, 0.0, -0.10) == pytest.approx(0.78, abs=0.03)
    assert power.mcnemar_power(rng, 150, 0.25, 0.20, 0.05 / 8) > 0.99
    with pytest.raises(ValueError):
        power.paired_counts(rng, 150, 0.05, 0.10)
    assert power.main(["--out-md", str(tmp_path / "p.md")]) == 0 and "| 150 | 20% |" in (tmp_path / "p.md").read_text(encoding="utf-8")


# --- the determinism check (research plan, step 3.9) -----------------------------------------------------------------


def test_two_runs_are_compared_answer_by_answer(tmp_path, capsys):
    import json

    check = load_script("check_determinism")
    env = {"provider": "ollama", "model": "m:7b", "questions_file": "q.jsonl", "retrieval": "frozen: f.json",
           "prompt": {"name": "P1"}}

    def run(name, answers, model="m:7b"):
        rows = [{"id": qid, "answer": text, "grounded": "[1]" in text, "cited": [1] if "[1]" in text else []}
                for qid, text in answers.items()]
        path = tmp_path / name
        path.write_text(json.dumps({"env": {**env, "model": model}, "results": rows}), encoding="utf-8")
        return str(path)

    first = run("a.json", {"q1": "Same text [1].", "q2": "It parses the header [1].", "q3": "Only here."})
    second = run("b.json", {"q1": "Same text [1].", "q2": "It parses the header and nothing else."})
    out = tmp_path / "det.md"
    assert check.main([first, second, "--out-md", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "**1 of 2 answers identical**; scores changed for 1" in text  # q3 is only in the first run
    assert "| q2 | no |" in text and "grounded, cited" in text and "the header [1]." in text
    assert "1 answer(s) differ" in capsys.readouterr().out
    other = run("c.json", {"q1": "Same text [1]."}, model="other:8b")
    assert check.main([first, other]) == 0 and "not set up the same way** (model differ)" in capsys.readouterr().out
    assert check.main([first]) == 2


def test_a_hosted_judge_needs_yes_and_bad_replies_never_raise(claim_judge, graded_runs, tmp_path):
    claim_judge.judge.sends_code_off_machine = True
    assert claim_judge.main([str(graded_runs[0]), "--judge-model", "x", "--out", str(tmp_path / "h.jsonl")]) == 2
    assert claim_judge.judge.calls == []
    assert claim_judge.parse_label('{"label": "partly"}')[2] == "the judge gave an unknown label: 'partly'"
    assert claim_judge.parse_label("Sure! {broken")[2] == "the judge's JSON could not be read"
    assert claim_judge.parse_label('Here: {"label": " Contradicted ", "reason": "r"}') == ("contradicted", "r", None)
