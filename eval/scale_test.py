r"""How does indexing and searching scale with repository size? (everything except the embedding model)

Generates synthetic Python repositories of increasing size and indexes each with a cheap hash-based embedder, so the
time and disk use of everything *around* the embedding model can be measured: walking, parsing and chunking, the SQLite
and Chroma writes, a search, and an incremental re-run when nothing changed. Embedding with the real model is the slow
step and is not measured here (it depends entirely on your hardware); the report prints the rate observed in a real run
if you give one with ``--real-chunks-per-second``, and extrapolates from it, clearly labelled as an extrapolation.

    python eval/scale_test.py --sizes 200,1000,4000
    python eval/scale_test.py --sizes 1000 --real-chunks-per-second 7.8

Synthetic code is not real code: it has no unusual file sizes, deep nesting or giant functions, so treat the results as
a check that nothing grows faster than the repository does, not as a promise about any particular project.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import tempfile
import time
from pathlib import Path

from codebase_ai.index.indexer import Indexer, RepoIndex
from codebase_ai.retrieval.retriever import Retriever

DIM = 384  # the size of the default embedding model's vectors, so Chroma's storage is realistic
FUNCTIONS_PER_FILE = 6
WORDS = ("order", "user", "cart", "price", "invoice", "session", "token", "queue", "cache", "retry", "stream", "batch")


class HashEmbedder:
    """A deterministic stand-in for the embedding model: fast, with vectors of the real size."""

    model_id = "hash:scale-test"
    dim = DIM

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIM
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def make_repo(root: Path, files: int) -> None:
    for number in range(files):
        package = root / f"pkg{number // 50}" / f"mod{number // 10}"
        package.mkdir(parents=True, exist_ok=True)
        word = WORDS[number % len(WORDS)]
        body = [f'"""Module {number}: handles {word} processing."""', "", "import os", ""]
        for index in range(FUNCTIONS_PER_FILE):
            name = f"{word}_{number}_{index}"
            body += [
                f"def {name}(items, limit={index + 1}):",
                f'    """Process {word} items number {index} of module {number}."""',
                "    total = 0",
                "    for item in items:",
                "        if item > limit:",
                "            total += item * limit",
                f"    return total + {number}",
                "",
            ]
        (package / f"m{number}.py").write_text("\n".join(body), encoding="utf-8")


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def measure(files: int, real_rate: float | None) -> dict:
    work = Path(tempfile.mkdtemp(prefix="scale-"))
    try:
        repo = work / "repo"
        make_repo(repo, files)
        index = RepoIndex(repo, work / "index")
        embedder = HashEmbedder()
        try:
            started = time.perf_counter()
            report = Indexer(index, embedder).run()
            index_seconds = time.perf_counter() - started
            chunks = index.keyword.count()

            started = time.perf_counter()
            Indexer(index, embedder).run()
            rerun_seconds = time.perf_counter() - started

            retriever = Retriever(index, embedder, mode="vector")
            timings = []
            for query in ("process order items", "retry the queue", "where is the cart total", "invoice token cache"):
                started = time.perf_counter()
                retriever.retrieve(query)
                timings.append((time.perf_counter() - started) * 1000)
            keyword = Retriever(index, None, mode="keyword")
            started = time.perf_counter()
            keyword.retrieve("process order items")
            keyword_ms = (time.perf_counter() - started) * 1000
            size = folder_size(index.dir)
        finally:
            index.close()
        result = {
            "files": files,
            "chunks": chunks,
            "index_seconds": index_seconds,
            "rerun_seconds": rerun_seconds,
            "search_ms": sum(timings) / len(timings),
            "keyword_ms": keyword_ms,
            "index_mb": size / 1_048_576,
            "files_indexed": report.files_indexed,
        }
        if real_rate:
            result["embedding_estimate_minutes"] = chunks / real_rate / 60
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def render(rows: list[dict], real_rate: float | None) -> str:
    header = "| Files | Chunks | Index (no embedding model) | Re-run, nothing changed | Vector search | Keyword search | Index size |"
    divider = "|---|---|---|---|---|---|---|"
    if real_rate:
        header += " Embedding with the real model (extrapolated) |"
        divider += "---|"
    lines = [header, divider]
    for r in rows:
        line = (
            f"| {r['files']:,} | {r['chunks']:,} | {r['index_seconds']:.1f} s | {r['rerun_seconds']:.1f} s | "
            f"{r['search_ms']:.0f} ms | {r['keyword_ms']:.0f} ms | {r['index_mb']:.0f} MB |"
        )
        if real_rate:
            line += f" about {r['embedding_estimate_minutes']:.0f} min |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="200,1000,4000", help="comma-separated file counts")
    parser.add_argument("--real-chunks-per-second", type=float, default=None, help="rate seen with the real model")
    args = parser.parse_args()
    rows = []
    for size in (int(s) for s in args.sizes.split(",") if s.strip()):
        print(f"measuring {size} files...", flush=True)
        rows.append(measure(size, args.real_chunks_per_second))
    print(render(rows, args.real_chunks_per_second))


if __name__ == "__main__":
    main()
