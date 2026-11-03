from app.report.qa import context_for


class Snapshot:
    def get_detail(self):
        return {"gaps": [{"rule_title": "控龙", "severity": "high", "time_label": "第 8 分钟", "expected_text": "优先控龙"}]}


class Match:
    analyses = [Snapshot()]
    def get_payload(self):
        return {"player": {"hero": "赵云", "position": "jungle"}, "outcome": {"result": "win"}}


def test_qa_context_only_contains_match_summary():
    context = context_for(Match(), {"tasks": [{"title": "控龙", "when": "第 8 分钟", "action": "优先控龙", "completed": False}]})
    assert context["hero"] == "赵云"
    assert context["gaps"][0]["rule_title"] == "控龙"
    assert context["training_tasks"][0]["completed"] is False
