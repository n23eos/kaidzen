import json

import pytest

from kaidzen.candidate import load_candidate
from kaidzen.mutation import (META_FILE, CandidatePatch, ancestry, set_status,
                              write_candidate)
from tests.test_candidate import make_candidate


def test_writes_new_candidate_with_parent_link(tmp_path):
    parent = make_candidate(tmp_path)
    patch = CandidatePatch(prompts={"judge": "новый промпт судьи, достаточно длинный"})
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-a", patch=patch)
    loaded = load_candidate(new_dir)
    assert loaded.candidate_id == "gen001-a"
    assert loaded.prompts["judge"].startswith("новый промпт")
    # непатченные промпты унаследованы дословно
    assert loaded.prompts["analyzer"] == load_candidate(parent).prompts["analyzer"]


def test_meta_json_records_parent_and_generation(tmp_path):
    parent = make_candidate(tmp_path)
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-a", patch=CandidatePatch())
    meta = json.loads((new_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["parent"] == parent.name
    assert meta["generation"] == 1
    assert meta["status"] == "pending"


def test_config_patch_is_applied_and_validated(tmp_path):
    parent = make_candidate(tmp_path)
    patch = CandidatePatch(config={"loop": {"assumptions_per_iteration": 5}})
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-b", patch=patch)
    assert load_candidate(new_dir).config.loop.assumptions_per_iteration == 5


def test_invalid_patch_is_rejected_and_nothing_written(tmp_path):
    """Сломанный мутантом конфиг не должен оставлять мусор на диске."""
    parent = make_candidate(tmp_path)
    # max_iterations ограничен сверху 20 (kaidzen.candidate.MAX_ITERATIONS)
    patch = CandidatePatch(config={"loop": {"max_iterations": 999}})
    with pytest.raises(ValueError):
        write_candidate(parent_dir=parent, root=tmp_path,
                        candidate_id="gen001-c", patch=patch)
    assert not (tmp_path / "gen001-c").exists()


def test_blank_prompt_patch_is_rejected(tmp_path):
    parent = make_candidate(tmp_path)
    with pytest.raises(ValueError):
        write_candidate(parent_dir=parent, root=tmp_path, candidate_id="gen001-d",
                        patch=CandidatePatch(prompts={"judge": "   "}))


def test_ancestry_walks_to_root(tmp_path):
    parent = make_candidate(tmp_path)
    first = write_candidate(parent_dir=parent, root=tmp_path,
                            candidate_id="gen001-a", patch=CandidatePatch())
    second = write_candidate(parent_dir=first, root=tmp_path,
                             candidate_id="gen002-a", patch=CandidatePatch())
    assert ancestry(second) == ["gen002-a", "gen001-a", parent.name]


# --- иммутабельность и отказы ------------------------------------------------


def test_parent_is_untouched_by_mutation(tmp_path):
    parent = make_candidate(tmp_path)
    before = load_candidate(parent)
    write_candidate(parent_dir=parent, root=tmp_path, candidate_id="gen001-a",
                    patch=CandidatePatch(
                        prompts={"judge": "совсем другой промпт судьи"},
                        config={"loop": {"max_iterations": 3}}))
    after = load_candidate(parent)
    assert after.prompts == before.prompts
    assert after.config == before.config
    assert not (parent / META_FILE).exists()


def test_existing_candidate_id_is_rejected(tmp_path):
    parent = make_candidate(tmp_path)
    write_candidate(parent_dir=parent, root=tmp_path, candidate_id="gen001-a",
                    patch=CandidatePatch())
    with pytest.raises(ValueError, match="gen001-a"):
        write_candidate(parent_dir=parent, root=tmp_path,
                        candidate_id="gen001-a", patch=CandidatePatch())


def test_existing_candidate_is_not_overwritten(tmp_path):
    """Отказ по занятому имени не должен затирать уже готового кандидата."""
    parent = make_candidate(tmp_path)
    first = write_candidate(parent_dir=parent, root=tmp_path,
                            candidate_id="gen001-a",
                            patch=CandidatePatch(prompts={"judge": "первый судья"}))
    with pytest.raises(ValueError):
        write_candidate(parent_dir=parent, root=tmp_path,
                        candidate_id="gen001-a",
                        patch=CandidatePatch(prompts={"judge": "второй судья"}))
    assert load_candidate(first).prompts["judge"] == "первый судья"


def test_unknown_role_in_prompt_patch_is_rejected(tmp_path):
    parent = make_candidate(tmp_path)
    with pytest.raises(ValueError, match="designer"):
        write_candidate(parent_dir=parent, root=tmp_path, candidate_id="gen001-e",
                        patch=CandidatePatch(prompts={"designer": "текст"}))
    assert not (tmp_path / "gen001-e").exists()


def test_unknown_config_key_is_rejected_and_nothing_written(tmp_path):
    """extra=forbid в конфиге ловит опечатку мутатора — до записи на диск."""
    parent = make_candidate(tmp_path)
    with pytest.raises(ValueError):
        write_candidate(parent_dir=parent, root=tmp_path, candidate_id="gen001-f",
                        patch=CandidatePatch(config={"lop": {"max_iterations": 3}}))
    assert not (tmp_path / "gen001-f").exists()


def test_empty_patch_copies_parent_verbatim(tmp_path):
    parent = make_candidate(tmp_path)
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-g", patch=CandidatePatch())
    assert load_candidate(new_dir).prompts == load_candidate(parent).prompts
    assert load_candidate(new_dir).config == load_candidate(parent).config


def test_rationale_is_stored_in_meta(tmp_path):
    parent = make_candidate(tmp_path)
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-h",
                              patch=CandidatePatch(rationale="гипотеза Н1"))
    meta = json.loads((new_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["rationale"] == "гипотеза Н1"
    assert meta["eval"] is None


def test_config_patch_merges_deeply_without_dropping_siblings(tmp_path):
    parent = make_candidate(tmp_path)
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-i",
                              patch=CandidatePatch(
                                  config={"loop": {"max_iterations": 4}}))
    loop = load_candidate(new_dir).config.loop
    assert loop.max_iterations == 4
    assert loop.assumptions_per_iteration == 3   # соседний ключ не потерян


# --- статусы и линия предков -------------------------------------------------


def test_set_status_updates_meta(tmp_path):
    parent = make_candidate(tmp_path)
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-a", patch=CandidatePatch())
    set_status(new_dir, "rejected", {"win_rate": 0.4})
    meta = json.loads((new_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["status"] == "rejected"
    assert meta["eval"] == {"win_rate": 0.4}


def test_set_status_keeps_eval_when_not_given(tmp_path):
    parent = make_candidate(tmp_path)
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-a", patch=CandidatePatch())
    set_status(new_dir, "evaluated", {"win_rate": 0.8})
    set_status(new_dir, "promoted")
    meta = json.loads((new_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["status"] == "promoted"
    assert meta["eval"] == {"win_rate": 0.8}


def test_ancestry_of_root_candidate_is_itself(tmp_path):
    parent = make_candidate(tmp_path)
    assert ancestry(parent) == [parent.name]


def test_ancestry_stops_on_missing_parent_dir(tmp_path):
    """Родителя удалили руками — цепочка обрывается, а не падает."""
    parent = make_candidate(tmp_path)
    child = write_candidate(parent_dir=parent, root=tmp_path,
                            candidate_id="gen001-a", patch=CandidatePatch())
    import shutil
    shutil.rmtree(parent)
    assert ancestry(child) == ["gen001-a"]


def test_ancestry_survives_self_referencing_meta(tmp_path):
    """Испорченный meta.json не должен зациклить обход предков."""
    parent = make_candidate(tmp_path)
    child = write_candidate(parent_dir=parent, root=tmp_path,
                            candidate_id="gen001-a", patch=CandidatePatch())
    meta_path = child / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["parent"] = child.name
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    assert ancestry(child) == ["gen001-a"]


def test_ancestry_survives_cycle_between_two_candidates(tmp_path):
    parent = make_candidate(tmp_path)
    first = write_candidate(parent_dir=parent, root=tmp_path,
                            candidate_id="gen001-a", patch=CandidatePatch())
    second = write_candidate(parent_dir=first, root=tmp_path,
                             candidate_id="gen002-a", patch=CandidatePatch())
    # порча: предок ссылается на своего же потомка
    meta_path = first / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["parent"] = second.name
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    assert ancestry(second) == ["gen002-a", "gen001-a"]


def test_broken_meta_json_names_the_file(tmp_path):
    """Битый meta.json — обычная порча артефакта на диске, а не сбой кода:
    ошибка должна называть файл, как это делает журнал эволюции."""
    candidate_dir = tmp_path / "gen001-a"
    candidate_dir.mkdir()
    (candidate_dir / "meta.json").write_text("{не json", encoding="utf-8")

    with pytest.raises(ValueError, match="meta.json"):
        ancestry(candidate_dir)


def test_ancestry_stops_on_parent_that_is_a_path(tmp_path):
    """_apply_patch пишет в parent имя папки. Путь там — порча файла, и
    обход по нему уходит за пределы каталога кандидатов: '..' указывает на
    существующий каталог, поэтому проверкой на exists() он не отсекается и
    попадает в линию предков как кандидат с именем '..'.
    """
    child = tmp_path / "gen001-a"
    child.mkdir()
    (child / "meta.json").write_text(
        json.dumps({"parent": "..", "generation": 1}), encoding="utf-8")

    assert ancestry(child) == ["gen001-a"]
