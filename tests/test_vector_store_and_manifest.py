from __future__ import annotations

import pytest

from codebase_ai.chunking import chunk_file
from codebase_ai.index.manifest import FileRecord, Manifest
from codebase_ai.index.vector_store import ChromaVectorStore
from helpers import make_source


def chunks_named(*names):
    return [
        chunk_file(make_source(f"def {name}():\n    return '{name}'\n", f"{name}.py"), "repo")[0] for name in names
    ]


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(tmp_path / "chroma")


def test_query_returns_nearest_first_with_cosine_scores(store):
    a, b, c = chunks_named("alpha", "beta", "gamma")
    store.upsert([a, b, c], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]])
    hits = store.query([1.0, 0.0, 0.0], 3)
    assert [h.chunk_id for h in hits] == [a.id, c.id, b.id]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[1].score == pytest.approx(0.7071, abs=1e-3)
    assert hits[2].score == pytest.approx(0.0, abs=1e-4)


def test_k_larger_than_collection_and_empty_collection(store):
    assert store.query([1.0, 0.0], 5) == []
    (a,) = chunks_named("alpha")
    store.upsert([a], [[1.0, 0.0]])
    assert len(store.query([1.0, 0.0], 50)) == 1
    assert store.query([1.0, 0.0], 0) == []


def test_upsert_replaces_and_delete_removes(store):
    a, b = chunks_named("alpha", "beta")
    store.upsert([a, b], [[1.0, 0.0], [0.0, 1.0]])
    store.upsert([a], [[0.0, 1.0]])
    assert store.count() == 2
    assert store.query([0.0, 1.0], 1)[0].score == pytest.approx(1.0, abs=1e-4)
    store.delete([a.id])
    assert store.count() == 1
    assert [h.chunk_id for h in store.query([1.0, 0.0], 5)] == [b.id]
    store.delete([])  # no-op


def test_reset_clears_the_collection(store):
    (a,) = chunks_named("alpha")
    store.upsert([a], [[1.0, 0.0]])
    store.reset()
    assert store.count() == 0
    store.upsert([a], [[1.0, 0.0]])
    assert store.count() == 1


def test_mismatched_lengths_are_rejected(store):
    (a,) = chunks_named("alpha")
    with pytest.raises(ValueError):
        store.upsert([a], [])


def test_vectors_persist_across_reopen(tmp_path):
    path = tmp_path / "chroma"
    (a,) = chunks_named("alpha")
    ChromaVectorStore(path).upsert([a], [[1.0, 0.0]])
    assert ChromaVectorStore(path).count() == 1


# --- manifest -----------------------------------------------------------------------------------------


@pytest.fixture
def manifest(tmp_path):
    m = Manifest(tmp_path / "m" / "manifest.db")
    yield m
    m.close()


def test_meta_round_trip(manifest):
    assert manifest.get_meta("k") is None
    manifest.set_meta("k", "v1")
    manifest.set_meta("k", "v2")
    assert manifest.get_meta("k") == "v2"


def test_files_upsert_list_delete(manifest):
    manifest.upsert_file("a.py", "h1", ["c1", "c2"])
    manifest.upsert_file("b.py", "h2", [])
    manifest.upsert_file("a.py", "h3", ["c9"])
    assert manifest.files() == {"a.py": FileRecord("h3", ("c9",)), "b.py": FileRecord("h2", ())}
    manifest.delete_file("a.py")
    assert set(manifest.files()) == {"b.py"}


def test_manifest_persists_only_after_commit_and_reset_clears(tmp_path):
    path = tmp_path / "manifest.db"
    m = Manifest(path)
    m.set_meta("k", "v")
    m.upsert_file("a.py", "h", ["c"])
    m.commit()
    m.close()
    again = Manifest(path)
    assert again.get_meta("k") == "v" and set(again.files()) == {"a.py"}
    again.reset()
    assert again.get_meta("k") is None and again.files() == {}
    again.close()


# --- closing releases the files ------------------------------------------------------------------------------------


def test_closing_an_index_releases_its_files_so_the_folder_can_be_removed(sample_repo, tmp_path, fake_embedder):
    """Regression: RepoIndex.close() left Chroma's handles open. On Windows that blocks deleting the folder, and a long
    session (or this test suite) ran out of the 512 files a process may keep open."""
    import shutil

    from codebase_ai.index.indexer import Indexer, RepoIndex

    index = RepoIndex(sample_repo, tmp_path / "closable")
    Indexer(index, fake_embedder).run()
    folder = index.dir
    index.close()
    shutil.rmtree(folder)  # PermissionError on Windows if anything is still open
    assert not folder.exists()


def test_a_closed_index_can_be_opened_again_and_still_holds_its_data(sample_repo, tmp_path, fake_embedder):
    from codebase_ai.index.indexer import Indexer, RepoIndex

    first = RepoIndex(sample_repo, tmp_path / "again")
    Indexer(first, fake_embedder).run()
    chunks = first.vectors.count()
    first.close()
    second = RepoIndex(sample_repo, tmp_path / "again")
    try:
        assert second.vectors.count() == chunks > 0
    finally:
        second.close()


def test_closing_twice_is_harmless_and_a_store_without_close_is_tolerated(sample_repo, tmp_path):
    from codebase_ai.index.indexer import RepoIndex

    class NoClose:
        def count(self):
            return 0

    index = RepoIndex(sample_repo, tmp_path / "twice", vectors=NoClose())  # type: ignore[arg-type]
    index.close()
    real = RepoIndex(sample_repo, tmp_path / "twice2")
    real.close()
    real.vectors.close()  # a second close on the store itself
