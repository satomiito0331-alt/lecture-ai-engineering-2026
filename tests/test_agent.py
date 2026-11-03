"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import sys

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

import app.agent as agent_module
from app.agent import _request_tool_approvals, agent

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz", "save_learning_log"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called(tmp_path, monkeypatch):
    monkeypatch.setattr("app.tools.LEARNING_LOG_DIR", tmp_path / "learning_log")
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_name for call in result.output.approvals] == ["save_learning_log"]
    assert not (tmp_path / "learning_log").exists()
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_tool_approvals_are_collected_by_call_id(monkeypatch):
    requests = DeferredToolRequests(
        approvals=[
            ToolCallPart(tool_name="save_learning_log", args={"entry": "保存する記録"}, tool_call_id="approved"),
            ToolCallPart(tool_name="save_learning_log", args={"entry": "保存しない記録"}, tool_call_id="denied"),
        ]
    )
    answers = iter(["はい", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    results = _request_tool_approvals(requests)

    assert results.approvals == {"approved": True, "denied": False}


def test_cli_resumes_run_with_approval_results(monkeypatch, capsys):
    request = DeferredToolRequests(
        approvals=[
            ToolCallPart(tool_name="save_learning_log", args={"entry": "保存する記録"}, tool_call_id="call-1")
        ]
    )
    pending = type("Pending", (), {"output": request, "all_messages": lambda self: ["history"]})()
    completed = type("Completed", (), {"output": "保存しました", "all_messages": lambda self: []})()

    class StubAgent:
        calls = []

        def run_sync(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return pending if len(self.calls) == 1 else completed

    stub = StubAgent()
    monkeypatch.setattr(agent_module, "agent", stub)
    monkeypatch.setattr(agent_module, "build_model", lambda: "test-model")
    monkeypatch.setattr(sys, "argv", ["agent", "学習記録を保存して"])
    monkeypatch.setattr("builtins.input", lambda _: "y")

    agent_module.main()

    assert len(stub.calls) == 2
    _, resumed = stub.calls[1]
    assert resumed["message_history"] == ["history"]
    assert resumed["deferred_tool_results"].approvals == {"call-1": True}
    assert "保存しました" in capsys.readouterr().out


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"
