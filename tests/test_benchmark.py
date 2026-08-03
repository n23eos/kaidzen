from pathlib import Path

import pytest

from kaidzen.benchmark import BenchmarkEmpty, load_benchmark


def make_ideas(root, names):
    d = root / "business" / "ideas"
    d.mkdir(parents=True)
    for n in names:
        (d / f"{n}.md").write_text(f"идея {n}", encoding="utf-8")
    return root


def test_splits_train_and_holdout(tmp_path):
    b = load_benchmark(make_ideas(tmp_path, list("abcdefghij")), domain="business")
    assert len(b.train) + len(b.holdout) == 10
    assert len(b.holdout) >= 1
    assert not (set(b.train) & set(b.holdout))


def test_split_is_deterministic(tmp_path):
    root = make_ideas(tmp_path, list("abcdefgh"))
    first = load_benchmark(root, domain="business")
    second = load_benchmark(root, domain="business")
    assert first.train == second.train and first.holdout == second.holdout


def test_single_idea_goes_to_train_and_holdout_is_empty(tmp_path):
    b = load_benchmark(make_ideas(tmp_path, ["only"]), domain="business")
    assert len(b.train) == 1 and b.holdout == []


def test_empty_benchmark_raises(tmp_path):
    (tmp_path / "business" / "ideas").mkdir(parents=True)
    with pytest.raises(BenchmarkEmpty, match="business"):
        load_benchmark(tmp_path, domain="business")


def test_missing_domain_raises(tmp_path):
    with pytest.raises(BenchmarkEmpty):
        load_benchmark(tmp_path, domain="games")


# --- арифметика разбивки на маленьких наборах --------------------------------


@pytest.mark.parametrize("count,expected_holdout", [
    (1, 0),   # одна идея: holdout пуст, иначе train остался бы пустым
    (2, 1),   # int(2 * 0.3) == 0, но защита от Гудхарта важнее размера train
    (3, 1),
    (8, 2),
    (10, 3),
])
def test_holdout_size_on_small_sets(tmp_path, count, expected_holdout):
    names = [f"idea{i:02d}" for i in range(count)]
    b = load_benchmark(make_ideas(tmp_path, names), domain="business")
    assert len(b.holdout) == expected_holdout
    assert len(b.train) == count - expected_holdout
    assert b.train, "train не должен пустеть ни при каком размере бенчмарка"


def test_domain_is_carried(tmp_path):
    b = load_benchmark(make_ideas(tmp_path, ["a", "b"]), domain="business")
    assert b.domain == "business"


def test_ideas_are_paths_to_existing_files(tmp_path):
    b = load_benchmark(make_ideas(tmp_path, ["a", "b", "c"]), domain="business")
    for path in [*b.train, *b.holdout]:
        assert isinstance(path, Path)
        assert path.is_file() and path.suffix == ".md"


def test_non_markdown_files_are_ignored(tmp_path):
    root = make_ideas(tmp_path, ["a", "b"])
    (root / "business" / "ideas" / "notes.txt").write_text("не идея",
                                                          encoding="utf-8")
    b = load_benchmark(root, domain="business")
    assert len(b.train) + len(b.holdout) == 2


def test_only_requested_domain_is_loaded(tmp_path):
    make_ideas(tmp_path, ["a", "b", "c"])
    other = tmp_path / "games" / "ideas"
    other.mkdir(parents=True)
    (other / "z.md").write_text("чужая идея", encoding="utf-8")
    b = load_benchmark(tmp_path, domain="business")
    assert all(p.parent.parent.name == "business" for p in b.train)


def test_ideas_are_sorted_by_name(tmp_path):
    b = load_benchmark(make_ideas(tmp_path, ["c", "a", "b"]), domain="business")
    ordered = [p.stem for p in [*b.train, *b.holdout]]
    assert ordered == sorted(ordered)
