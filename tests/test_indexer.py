from __future__ import annotations

import pytest

from codebase_ai.index.indexer import Indexer, IndexMismatchError, RepoIndex, repo_id_for
from codebase_ai.retrieval.retriever import search_keyword, search_vector
from helpers import FakeEmbedder

INDEXABLE_FILES = 8  # sample_repo: 6 code files + README.md + settings.yaml


@pytest.fixture
def index_root(tmp_path):
    return tmp_path / "idx"


@pytest.fixture
def open_index(sample_repo, index_root):
    opened: list[RepoIndex] = []

    def _open(repo=None):
        idx = RepoIndex(repo or sample_repo, index_root)
        opened.append(idx)
        return idx

    yield _open
    for idx in opened:
        idx.close()


def run(idx, embedder, **kwargs):
    run_kwargs = {"full": kwargs.pop("full", False)}
    return Indexer(idx, embedder, **kwargs).run(**run_kwargs)


def assert_consistent(idx):
    """Vector store, keyword index and manifest must agree about which chunks exist."""
    manifest_ids = {i for record in idx.manifest.files().values() for i in record.chunk_ids}
    assert idx.keyword.count() == idx.vectors.count() == len(manifest_ids)
    assert {c.id for c in idx.keyword.get_chunks(sorted(manifest_ids))} == manifest_ids


def test_builds_an_index_end_to_end(open_index, fake_embedder):
    idx = open_index()
    assert not idx.is_built()
    report = run(idx, fake_embedder)
    assert report.files_new == report.files_indexed == INDEXABLE_FILES
    assert report.files_unchanged == report.files_changed == report.files_removed == 0
    assert report.chunks_added > INDEXABLE_FILES
    assert report.embedding_model_id == "fake:hash-64"
    assert report.seconds >= 0
    assert idx.is_built()
    assert len(idx.manifest.files()) == INDEXABLE_FILES
    assert_consistent(idx)
    info = idx.info()
    assert info["embedding_model_id"] == "fake:hash-64" and info["embedding_dim"] == "64"
    assert info["repo_root"] == str(idx.repo_root)


def test_secrets_ignored_and_generated_files_never_reach_any_store(open_index, fake_embedder, sample_repo):
    (sample_repo / ".env.local").write_text("TOKEN=abc\n")
    (sample_repo / "deploy.pem").write_text("x")
    (sample_repo / "node_modules").mkdir()
    (sample_repo / "node_modules" / "lib.js").write_text("export const a = 1;\n")
    (sample_repo / ".gitignore").write_text("scratch/\n")
    (sample_repo / "scratch").mkdir()
    (sample_repo / "scratch" / "tmp.py").write_text("def tmp():\n    pass\n")
    (sample_repo / "leaky.py").write_text('KEY = "' + "AKIA" + 'IOSFODNN7EXAMPLE"\n')

    idx = open_index()
    report = run(idx, fake_embedder)

    assert report.skipped["secret_file"] == 2
    assert report.skipped["secret_pattern"] == 1
    assert report.skipped["denied_dir"] == 1 and report.skipped["gitignored"] == 1
    indexed_paths = {c.path for c in idx.keyword.get_chunks([i for r in idx.manifest.files().values() for i in r.chunk_ids])}
    assert not {p for p in indexed_paths if "env" in p or "pem" in p or "node_modules" in p or "scratch" in p or "leaky" in p}
    assert all("AKIA" not in c.text for c in idx.keyword.get_chunks([i for r in idx.manifest.files().values() for i in r.chunk_ids]))


def test_second_run_does_nothing(open_index, fake_embedder):
    idx = open_index()
    run(idx, fake_embedder)
    again = run(idx, fake_embedder)
    assert again.files_unchanged == INDEXABLE_FILES
    assert again.files_indexed == again.chunks_added == again.chunks_removed == 0
    assert_consistent(idx)


def test_only_changed_files_are_reembedded(open_index, sample_repo):
    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.embedded = 0

        def embed_documents(self, texts):
            self.embedded += len(texts)
            return super().embed_documents(texts)

    embedder = CountingEmbedder()
    idx = open_index()
    run(idx, embedder)
    first_run_embeddings = embedder.embedded

    cart = sample_repo / "web" / "cart.js"
    cart.write_text(cart.read_text() + "\nexport function clearCart(items) {\n  items.length = 0;\n}\n")
    report = run(idx, embedder)

    assert (report.files_changed, report.files_new, report.files_unchanged) == (1, 0, INDEXABLE_FILES - 1)
    assert 0 < embedder.embedded - first_run_embeddings < first_run_embeddings
    assert_consistent(idx)
    symbols = {c.symbol for c in idx.keyword.get_chunks([h.chunk_id for h in idx.keyword.search("clear cart", 5)])}
    assert "clearCart" in symbols


def test_changed_file_leaves_no_stale_chunks_behind(open_index, fake_embedder, sample_repo):
    idx = open_index()
    run(idx, fake_embedder)
    (sample_repo / "go" / "server.go").write_text("package server\n\nfunc OnlyThing() {}\n")
    report = run(idx, fake_embedder)
    assert report.chunks_removed > 0
    assert_consistent(idx)
    chunks = [c for c in idx.keyword.get_chunks([i for r in idx.manifest.files().values() for i in r.chunk_ids]) if c.path == "go/server.go"]
    assert "NewServer" not in {c.symbol for c in chunks}
    assert "OnlyThing" in {c.symbol for c in chunks}


def test_new_and_deleted_files_are_picked_up(open_index, fake_embedder, sample_repo):
    idx = open_index()
    run(idx, fake_embedder)
    (sample_repo / "app" / "billing.py").write_text("def charge_card(card):\n    return card\n")
    (sample_repo / "go" / "server.go").unlink()
    report = run(idx, fake_embedder)
    assert (report.files_new, report.files_removed) == (1, 1)
    assert report.chunks_removed > 0
    assert_consistent(idx)
    assert "app/billing.py" in idx.manifest.files() and "go/server.go" not in idx.manifest.files()
    assert search_keyword(idx, "start server", 10) == [] or all(c.path != "go/server.go" for c, _ in search_keyword(idx, "start server", 10))


def test_a_file_that_becomes_a_secret_is_removed_from_the_index(open_index, fake_embedder, sample_repo):
    idx = open_index()
    run(idx, fake_embedder)
    target = sample_repo / "app" / "user_service.py"
    target.write_text(target.read_text() + '\nKEY = "' + "AKIA" + 'IOSFODNN7EXAMPLE"\n')
    report = run(idx, fake_embedder)
    assert report.files_removed == 1 and report.skipped["secret_pattern"] == 1
    assert "app/user_service.py" not in idx.manifest.files()
    assert_consistent(idx)


def test_a_crash_mid_batch_leaves_no_orphaned_chunks(open_index, fake_embedder, sample_repo, monkeypatch):
    idx = open_index()
    run(idx, fake_embedder)
    target = sample_repo / "web" / "cart.js"
    original = target.read_text()

    # The run dies after the vector store was written but before the keyword index was.
    target.write_text(original + "\nexport function intermediateOnly() {\n  return 1;\n}\n")

    def killed(chunks):
        raise RuntimeError("process killed")

    with monkeypatch.context() as patched:
        patched.setattr(idx.keyword, "upsert", killed)
        with pytest.raises(RuntimeError, match="process killed"):
            run(idx, fake_embedder)

    # The file changes again before indexing resumes.
    target.write_text(original + "\nexport function finalVersion() {\n  return 2;\n}\n")
    run(idx, fake_embedder)

    assert_consistent(idx)  # vectors == keyword == manifest, so the intermediate chunks are gone everywhere
    stored = idx.keyword.get_chunks([i for r in idx.manifest.files().values() for i in r.chunk_ids])
    symbols = {c.symbol for c in stored}
    assert "finalVersion" in symbols and "intermediateOnly" not in symbols


def test_small_batches_give_the_same_index(open_index, fake_embedder, sample_repo, tmp_path):
    big = open_index()
    run(big, fake_embedder, batch_size=64)
    small = RepoIndex(sample_repo, tmp_path / "idx-small")
    try:
        run(small, fake_embedder, batch_size=2)
        assert {i for r in small.manifest.files().values() for i in r.chunk_ids} == {
            i for r in big.manifest.files().values() for i in r.chunk_ids
        }
        assert_consistent(small)
    finally:
        small.close()


def test_changing_the_embedding_model_requires_a_full_rebuild(open_index, sample_repo):
    idx = open_index()
    run(idx, FakeEmbedder("fake:model-a"))
    with pytest.raises(IndexMismatchError, match="embedding model changed.*--full"):
        run(idx, FakeEmbedder("fake:model-b"))
    # the refused run must not have damaged the index
    assert_consistent(idx)
    assert idx.info()["embedding_model_id"] == "fake:model-a"

    report = run(idx, FakeEmbedder("fake:model-b"), full=True)
    assert report.full and report.files_new == INDEXABLE_FILES
    assert idx.info()["embedding_model_id"] == "fake:model-b"
    assert_consistent(idx)


def test_changed_chunking_logic_requires_a_full_rebuild(open_index, fake_embedder):
    idx = open_index()
    run(idx, fake_embedder)
    idx.manifest.set_meta("chunker_version", "0")
    idx.manifest.commit()
    with pytest.raises(IndexMismatchError, match="chunking logic changed"):
        run(idx, fake_embedder)


def test_full_rebuild_matches_a_fresh_build(open_index, fake_embedder):
    idx = open_index()
    run(idx, fake_embedder)
    before = {i for r in idx.manifest.files().values() for i in r.chunk_ids}
    report = run(idx, fake_embedder, full=True)
    assert report.files_new == INDEXABLE_FILES and report.files_unchanged == 0
    assert {i for r in idx.manifest.files().values() for i in r.chunk_ids} == before
    assert_consistent(idx)


def test_progress_callback_sees_the_final_report(open_index, fake_embedder):
    seen = []
    Indexer(open_index(), fake_embedder, batch_size=4).run(progress=lambda r: seen.append((r.files_indexed, r.chunks_added)))
    assert len(seen) >= 2  # at least one mid-run flush plus the final call
    assert seen[-1][0] == INDEXABLE_FILES
    assert [s[1] for s in seen] == sorted(s[1] for s in seen)


def test_index_survives_reopening(open_index, fake_embedder, sample_repo, index_root):
    first = open_index()
    run(first, fake_embedder)
    first.close()
    reopened = RepoIndex(sample_repo, index_root)
    try:
        assert reopened.is_built()
        assert run(reopened, fake_embedder).files_unchanged == INDEXABLE_FILES
    finally:
        reopened.close()


def test_vector_search_finds_relevant_code(open_index, fake_embedder):
    idx = open_index()
    run(idx, fake_embedder)
    results = search_vector(idx, fake_embedder, "validate email and create a user", k=3)
    assert results and results[0][0].path == "app/user_service.py"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    chunk, _ = search_vector(idx, fake_embedder, "shopping cart total price", k=1)[0]
    assert chunk.path == "web/cart.js"


def test_repo_id_is_stable_readable_and_distinct(tmp_path):
    a, b = tmp_path / "proj", tmp_path / "other" / "proj"
    a.mkdir()
    b.mkdir(parents=True)
    assert repo_id_for(a) == repo_id_for(a)
    assert repo_id_for(a) != repo_id_for(b)
    assert repo_id_for(a).startswith("proj-")


def test_index_status_reports_files_chunks_and_model(sample_repo, tmp_path, fake_embedder):
    from codebase_ai.index.indexer import index_status

    idx = RepoIndex(sample_repo, tmp_path / "status-idx")
    try:
        Indexer(idx, fake_embedder).run()
        chunks = idx.keyword.count()
        files = len(idx.manifest.files())
    finally:
        idx.close()
    status = index_status(sample_repo, tmp_path / "status-idx")
    assert status is not None
    assert (status.embedding_model_id, status.files, status.chunks) == ("fake:hash-64", files, chunks)
    assert status.files > 0 and status.chunks >= status.files
    assert status.location == idx.dir


def test_index_status_of_a_repo_that_was_never_indexed_is_none_and_creates_nothing(sample_repo, tmp_path):
    from codebase_ai.index.indexer import index_status

    root = tmp_path / "empty-root"
    assert index_status(sample_repo, root) is None
    assert not root.exists()


def test_index_status_of_a_half_built_index_is_none(sample_repo, tmp_path):
    from codebase_ai.index.indexer import index_status

    idx = RepoIndex(sample_repo, tmp_path / "half-idx")  # opened but never indexed: no embedding model recorded
    idx.close()
    assert index_status(sample_repo, tmp_path / "half-idx") is None
