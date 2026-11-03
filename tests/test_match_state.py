from app.collection.match_state import consensus, recommend


def test_state_consensus_and_build_recommendation():
    state = consensus([{"time_min": 8, "confidence": .9, "player": {"hero": "赵云", "items": ["暗影战斧"]}, "matchup": {"hero": "貂蝉", "damage_type": "magic"}, "build": {}}, {"time_min": 8, "confidence": .8, "player": {"hero": "赵云", "items": ["暗影战斧"]}, "matchup": {"hero": "貂蝉", "damage_type": "magic"}, "build": {}}])
    assert state["status"] == "ready"
    assert state["player"]["hero"] == "赵云"
    assert recommend(state)["direction"] == "魔抗/法术防御"


def test_state_refuses_low_confidence_frames():
    assert consensus([{"confidence": .2}])["status"] == "uncertain"


def test_vlm_state_parser_requires_two_frames_and_confidence():
    from app.collection.vlm_recognizer import parse_answer
    parsed = parse_answer({"state": {"frames": [1, 2], "confidence": .8, "player": {"hero": "赵云"}, "matchup": {"hero": "貂蝉"}}}, {})
    assert parsed.state["player"]["hero"] == "赵云"
    rejected = parse_answer({"state": {"frames": [1], "confidence": .9, "player": {"hero": "赵云"}}}, {})
    assert rejected.state == {}


def test_window_state_reaches_recognition_round_and_is_saved():
    from app.collection.video_upload import _window_rounds
    from app.main import _save_recognized_state_snapshots

    windows = [{
        "lo": 8, "slices": [{"timestamp": 480, "image": "frame.jpg"}],
        "facts": {"state_snapshot": {
            "confidence": .86, "player": {"hero": "赵云", "items": ["暗影战斧"]},
            "matchup": {"hero": "貂蝉", "damage_type": "magic"},
        }},
    }]
    rounds = _window_rounds(windows, "hok")
    assert rounds[0].state_snapshot["player"]["hero"] == "赵云"

    class FakeSession:
        def __init__(self): self.rows = []; self.flushed = False
        def add(self, row): self.rows.append(row)
        def flush(self): self.flushed = True

    session = FakeSession()
    assert _save_recognized_state_snapshots(session, "demo", "local", rounds) == 1
    assert session.flushed
    assert session.rows[0].get_payload()["matchup"]["hero"] == "貂蝉"


def test_purchase_feed_events_are_kept_as_a_timeline():
    from app.collection.vlm_recognizer import parse_answer

    parsed = parse_answer({"state": {
        "frames": [1, 2], "confidence": .85,
        "player": {"hero": "赵云"}, "matchup": {},
        "build": {"purchase_events": [{
            "item": "暗影战斧", "frames": [2],
            "evidence": "左侧购买提示显示暗影战斧", "confidence": .9,
        }]},
    }}, {})
    assert parsed.state["build"]["purchase_events"][0]["item"] == "暗影战斧"

    state = consensus([
        {"time_min": 4, "confidence": .85, "player": {"hero": "赵云"},
         "matchup": {}, "build": parsed.state["build"]},
    ])
    assert state["build"]["purchase_events"] == [{
        "time_min": 4, "item": "暗影战斧",
        "evidence": "左侧购买提示显示暗影战斧", "confidence": .9,
    }]


def test_purchase_scan_does_not_hide_hud_level_or_gold():
    state = consensus([
        {"time_min": 1, "confidence": .92,
         "player": {"hero": "曜", "items": ["陨星"]}, "matchup": {}, "build": {}},
        {"time_min": 10, "confidence": .72,
         "player": {"hero": "曜", "level": 14, "gold": 9035}, "matchup": {}, "build": {}},
    ])
    assert state["player"]["items"] == ["陨星"]
    assert state["player"]["level"] == 14
    assert state["player"]["gold"] == 9035


def test_scoreboard_roster_drives_counter_build_advice():
    from app.collection.vlm_recognizer import parse_answer

    parsed = parse_answer({"state": {
        "frames": [1, 2], "confidence": .86,
        "player": {"hero": "曜"}, "matchup": {},
        "build": {"enemy_roster": [{"hero": "张飞", "frames": [1, 2],
                                       "evidence": "战绩面板敌方英雄栏显示张飞", "confidence": .9}],
                  "enemy_traits": {"control_heavy": True, "tank_heavy": True}},
    }}, {})
    state = consensus([{ "time_min": 5, **parsed.state }])
    assert state["build"]["enemy_roster"][0]["hero"] == "张飞"
    assert recommend(state)["direction"] == "韧性/解控"
