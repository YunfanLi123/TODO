"""开局小地图轨迹的分路判定：只在证据充分时覆盖 VLM。"""

from app.collection.vlm_prefill import infer_lane_from_track


def _row(t, x, y):
    return {"t": t, "nx": x, "ny": y, "conf": 0.9}


def test_track_with_stable_mid_presence_is_mid():
    rows = [_row(i * 10, 0.45 + (i % 3) * 0.02, 0.55) for i in range(12)]
    result = infer_lane_from_track(rows, 0.0)
    assert result["position"] == "mid"
    assert result["confidence"] >= 0.6


def test_track_with_too_few_points_does_not_guess_a_lane():
    rows = [_row(i * 10, 0.45, 0.55) for i in range(4)]
    result = infer_lane_from_track(rows, 0.0)
    assert result["position"] == "unknown"
