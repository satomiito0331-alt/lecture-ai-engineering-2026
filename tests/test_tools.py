"""ツールのテスト（LLMもAPIキーも使わない）。"""

import json

import pytest

from app import tools


def test_glossary_ids_are_unique_and_related_links_exist():
    terms = tools.load_glossary()
    ids = [t["id"] for t in terms]
    assert len(ids) == len(set(ids))
    for t in terms:
        for r in t["related"]:
            assert r in ids, f"{t['id']} の related に存在しない id: {r}"


def test_search_finds_term_by_name_and_alias():
    for query in ["LoRA", "lora", "Low-Rank Adaptation"]:
        hits = json.loads(tools.search_glossary(query))
        assert hits[0]["term"] == "LoRA", query


def test_search_finds_course_specific_term():
    hits = json.loads(tools.search_glossary("lecNN-start"))
    assert hits[0]["term"] == "lecNN-start タグ"


def test_search_limits_number_of_results():
    hits = json.loads(tools.search_glossary("エージェント"))
    assert 1 <= len(hits) <= tools.MAX_RESULTS


def test_search_reports_not_found():
    assert "見つかりませんでした" in tools.search_glossary("存在しない用語xyz")


def test_lecture_summary():
    assert "エージェント" in tools.get_lecture_summary(3)
    assert "まだありません" in tools.get_lecture_summary(99)


def test_save_learning_log_appends_entry_to_date_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")

    result = tools.save_learning_log("  今日学んだこと  ")
    tools.save_learning_log("次に試したいこと")
    files = list(tools.LEARNING_LOG_DIR.glob("*.md"))

    assert len(files) == 1
    contents = files[0].read_text(encoding="utf-8")
    assert "今日学んだこと" in contents
    assert "次に試したいこと" in contents
    assert result.startswith("学習記録を learning_log/")


def test_save_learning_log_rejects_empty_entry():
    with pytest.raises(ValueError, match="学習記録が空"):
        tools.save_learning_log("   ")


def test_make_quiz_uses_a_term_from_the_requested_lecture(monkeypatch):
    terms = [term for term in tools.load_glossary() if term["lecture"] == 2]
    monkeypatch.setattr(tools.random, "choice", lambda choices: choices[0])

    quiz = json.loads(tools.make_quiz(2))

    assert quiz == {
        "question": f"「{terms[0]['term']}」の意味は？",
        "answer": terms[0]["definition"],
    }


def test_make_quiz_reports_when_lecture_has_no_terms():
    assert "第99回の用語が見つかりませんでした" in tools.make_quiz(99)


def test_default_question_of_lecture1_is_answerable_from_glossary():
    """第1回の既定の質問（app.ask）は、第3回のエージェントが用語集で正しく答えられること。"""
    from app.ask import DEFAULT_QUESTION

    assert "3レベル" in DEFAULT_QUESTION
    hits = json.loads(tools.search_glossary("3レベル"))
    assert hits[0]["term"] == "宿題の3レベル"
