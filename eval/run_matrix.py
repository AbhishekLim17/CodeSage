r"""Run the study's experiment matrix: every model x repository x condition, resumable (research plan, step 3.3).

    python eval/run_matrix.py research/matrix.dev.toml --dry-run   # list the jobs and check the models are pulled
    python eval/run_matrix.py research/matrix.dev.toml             # run them; re-run the same command to resume

Each job is one ``run_answers.py`` run with ``--resume``, so a crash or Ctrl+C loses at most one answer and running the
command again carries on. Retrieval is frozen once per repository before any model runs, and every job replays it, so
all models and conditions see exactly the same code. Every job execution is appended to ``runs.jsonl`` in the results
folder, with what is needed to reproduce it: the code commit (and whether there were uncommitted changes), the Ollama
version, the model's digest, quantisation and size, and the settings.

Only local models (Ollama) are run. A condition is a prompt (P0 to P4, ``answer_eval.apply_prompt``) and a context:
``frozen`` (the retrieved code) or ``oracle`` (the same retrieval limited to each question's gold files, frozen
separately). A condition may list ``models`` to run on only some of them (P2 to P4 and the oracle run on the 7B model
only). Use ``LLM_TIMEOUT_SECONDS=600`` on a laptop (set by default when run as a script).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_answers

from codebase_ai.answer_eval import PROMPTS
from codebase_ai.config import get_settings
from codebase_ai.llm.ollama_provider import DEFAULT_TEMPERATURE, OUTPUT_CAP

CONTEXTS = {"frozen": "frozen_retrieval.json", "oracle": "frozen_oracle.json"}  # the stored sources each context replays
PROJECT = Path(__file__).resolve().parent.parent
SETTINGS_RECORDED = {
    "ollama_num_ctx", "answer_max_tokens", "answer_temperature", "llm_timeout_seconds", "embedding_provider",
    "local_embedding_model", "retrieval_mode", "test_penalty", "changelog_penalty", "context_token_budget",
    "retrieve_top_k", "use_repo_map", "repo_map_tokens",
}


def load_config(path: Path) -> dict:
    """The TOML config, with every path resolved relative to the config file."""
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    config["results_dir"] = base / config["results_dir"]
    if config.get("index_root"):
        config["index_root"] = base / config["index_root"]
    for repo in config["repos"]:
        repo["path"] = base / repo["path"]
        repo["questions"] = base / repo["questions"]
    return config


def problems_in(config: dict) -> list[str]:
    found = []
    names = [repo["name"] for repo in config["repos"]]
    if len(names) != len(set(names)):
        found.append("repository names must be unique")
    for condition in config["conditions"]:
        if condition["prompt"] not in PROMPTS:
            found.append(f"unknown prompt {condition['prompt']!r}; expected one of {', '.join(PROMPTS)}")
        if condition["context"] not in CONTEXTS:
            found.append(f"unknown context {condition['context']!r}; expected one of {', '.join(CONTEXTS)}")
        unknown = set(condition.get("models", [])) - set(config["models"])
        if unknown:
            found.append(f"condition {condition['prompt']}-{condition['context']} names models not in `models`: {sorted(unknown)}")
    return found


def jobs_of(config: dict) -> list[dict]:
    """Every model x repository x condition, grouped by model so each model is loaded only once."""
    return [
        {
            "model": model,
            "repo": repo["name"],
            "condition": f"{condition['prompt']}-{condition['context']}",
            "prompt": condition["prompt"],
            "context": condition["context"],
        }
        for model in config["models"]
        for repo in config["repos"]
        for condition in config["conditions"]
        if model in condition.get("models", config["models"])
    ]


def ollama_state(host: str) -> tuple[str | None, dict[str, dict]]:
    """Ollama's version and the pulled models with their digest, size and quantisation; (None, {}) if unreachable."""
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=10) as response:
            version = json.load(response)["version"]
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as response:
            pulled = json.load(response)["models"]
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return None, {}
    return version, {
        m["name"]: {
            "digest": m["digest"],
            "parameter_size": m["details"].get("parameter_size"),
            "quantization_level": m["details"].get("quantization_level"),
        }
        for m in pulled
    }


def code_version() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(PROJECT), *args], capture_output=True, text=True, check=False).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD") or None, "uncommitted_changes": bool(git("status", "--porcelain"))}
    except FileNotFoundError:  # git is not installed
        return {"commit": None, "uncommitted_changes": None}


def now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="list the jobs and check the models; run nothing")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions of each repository (smoke runs)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    problems = problems_in(config)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 2
    jobs = jobs_of(config)
    settings = get_settings()
    version, pulled = ollama_state(settings.ollama_host)
    code = code_version()

    print(f"{len(jobs)} job(s), grouped by model:")
    for job in jobs:
        print(f"  {job['model']:22} {job['repo']:14} {job['condition']}")
    if version is None:
        print(f"error: Ollama is not reachable at {settings.ollama_host}. Start it with: ollama serve", file=sys.stderr)
        return 2
    missing = [model for model in config["models"] if model not in pulled]
    if missing:
        print("error: these models are not pulled:" + "".join(f"\n  ollama pull {m}" for m in missing), file=sys.stderr)
        return 2
    if code["uncommitted_changes"]:
        print("Warning: the code has uncommitted changes, so these results are not tied to a commit. Commit and tag the "
              "code before the real test runs.")
    if args.dry_run:
        return 0

    results = config["results_dir"]
    common = ["--provider", "ollama"]
    if config.get("index_root"):
        common += ["--index-root", str(config["index_root"])]
    if args.limit is not None:
        common += ["--limit", str(args.limit)]
    repos = {repo["name"]: repo for repo in config["repos"]}

    def frozen_file(name: str, context: str) -> Path:
        return results / name / CONTEXTS[context]

    for context in sorted({job["context"] for job in jobs}, key=list(CONTEXTS).index):
        for name, repo in repos.items():
            if frozen_file(name, context).exists():  # once frozen, never re-frozen: the stored sources are the contract
                continue
            print(f"\n=== freezing {'the oracle' if context == 'oracle' else 'retrieval'} for {name}")
            argv_freeze = [str(repo["path"]), str(repo["questions"]), *common, "--freeze-retrieval", str(frozen_file(name, context))]
            if run_answers.main(argv_freeze + (["--oracle"] if context == "oracle" else [])) != 0:
                print(f"error: could not freeze the {context} context for {name}", file=sys.stderr)
                return 1

    failed = []
    for job in jobs:
        repo = repos[job["repo"]]
        out = results / job["repo"] / job["condition"] / (job["model"].replace(":", "_").replace("/", "_") + ".json")
        job_argv = [
            str(repo["path"]), str(repo["questions"]), *common, "--model", job["model"],
            "--frozen", str(frozen_file(job["repo"], job["context"])), "--prompt", job["prompt"], "--out-json", str(out), "--resume",
            "--fresh-model",  # decision D11: every answer from a freshly loaded model, so any job can be re-run exactly
        ]
        print(f"\n=== {job['model']} | {job['repo']} | {job['condition']}")
        started = now()
        exit_code = run_answers.main(job_argv)
        record = {
            **job, "started": started, "finished": now(), "exit_code": exit_code, "argv": job_argv,
            "model_info": pulled[job["model"]], "ollama_version": version, "code": code,
            "settings": settings.model_dump(include=SETTINGS_RECORDED, mode="json"),
            # What the model actually got: the settings above are what was configured, and Ollama narrows them.
            "applied": {
                "code_budget_tokens": json.loads(frozen_file(job["repo"], job["context"]).read_text(encoding="utf-8"))["meta"]["budget_tokens"],
                "max_output_tokens": min(settings.answer_max_tokens, OUTPUT_CAP),
                "temperature": DEFAULT_TEMPERATURE if settings.answer_temperature is None else settings.answer_temperature,
                "fresh_model": True,
            },
        }
        results.mkdir(parents=True, exist_ok=True)
        with (results / "runs.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
        if exit_code != 0:
            failed.append(job)

    print(f"\n{len(jobs) - len(failed)} of {len(jobs)} job(s) finished.")
    for job in failed:
        print(f"  failed: {job['model']} | {job['repo']} | {job['condition']}")
    if failed:
        print("Run the same command again to resume; finished answers are kept.")
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("LLM_TIMEOUT_SECONDS", "600")  # a prompt can take over a minute to read on a laptop
    sys.exit(main())
