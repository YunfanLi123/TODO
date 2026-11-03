from app.report.training import build_training_plan


class Snapshot:
    def get_detail(self):
        return {"gaps": [{"rule_id": "low", "rule_title": "低优先级", "severity": "low", "time_label": "第 3 分钟", "expected_text": "做低优先级动作。"}, {"rule_id": "high", "rule_title": "高优先级", "severity": "high", "time_label": "第 5 分钟", "expected_text": "先控资源。", "why_text": "这局错过了资源窗口"}, {"rule_id": "high", "rule_title": "高优先级", "severity": "high", "time_label": "第 8 分钟", "expected_text": "先控资源。"}]}


class Match:
    analyses = [Snapshot()]


def test_training_plan_prioritizes_and_deduplicates_rules():
    plan = build_training_plan(Match())
    assert plan["status"] == "ready"
    assert [task["title"] for task in plan["tasks"]] == ["高优先级", "低优先级"]
    assert plan["tasks"][0]["action"] == "先控资源。"


def test_training_plan_marks_previously_completed_task():
    plan = build_training_plan(Match(), {"high"})
    assert plan["tasks"][0]["completed"] is True
    assert plan["tasks"][1]["completed"] is False


def test_training_center_template_renders_empty_state():
    from app.report.views import render_page
    html = render_page("training_center.html", tasks=[], completed=0)
    assert "训练中心" in html
    assert "暂时没有待完成训练" in html
