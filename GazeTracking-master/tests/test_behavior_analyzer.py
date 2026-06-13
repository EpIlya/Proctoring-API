import sys
import time
import pytest
from unittest.mock import MagicMock, patch

for mod_name in ("cv2", "gaze_tracking", "mediapipe", "mss", "winsound"):
    sys.modules.setdefault(mod_name, MagicMock())

import types
_tk = types.ModuleType("tkinter")
for _attr in ("Tk", "Toplevel", "Label", "Button", "Entry", "Frame",
              "Canvas", "ttk", "messagebox", "BOTH", "LEFT", "TOP"):
    setattr(_tk, _attr, MagicMock())
sys.modules["tkinter"] = _tk
sys.modules["tkinter.ttk"] = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()

mp_mock = MagicMock()
mp_mock.solutions.face_mesh.FaceMesh.return_value = MagicMock()
sys.modules["mediapipe"] = mp_mock

import main_with_head_tracking_model as mod

def make_analyzer(max_actions=1, window_size=15, min_consecutive=6, head_min=6):
    """Создаёт BehaviorAnalyzer с управляемыми параметрами."""
    with patch.dict(mod.CONFIG, {
        "max_suspicious_actions": max_actions,
        "analysis_window": 5,
        "sleep_interval": 0.33,
        "head_min_consecutive": head_min,
    }):
        ba = mod.BehaviorAnalyzer(max_suspicious_actions=max_actions)
    ba.window_size            = window_size
    ba.min_consecutive_offcenter = min_consecutive
    return ba


def offcenter_gaze_info():
    return {
        "horizontal_deviation": 0.2,
        "vertical_deviation":   0.1,
        "total_deviation":      0.224,
        "angle_from_real_center":        10.0,
        "angle_from_calibrated_center":  8.0,
    }


def make_head_info(direction="left", is_suspicious=True, yaw=25.0, pitch=0.0):
    return {
        "direction":      direction,
        "is_suspicious":  is_suspicious,
        "yaw_deviation":  yaw,
        "pitch_deviation": pitch,
    }

class TestBehaviorAnalyzerInit:
    def test_initial_state(self):
        ba = make_analyzer()
        assert ba.suspicious_actions == 0
        assert ba.gaze_history       == []
        assert ba._head_history      == []
        assert ba._gaze_triggers     == 0
        assert ba._head_triggers     == 0
        assert ba.consecutive_offcenter == 0

    def test_custom_max_suspicious_actions(self):
        ba = make_analyzer(max_actions=3)
        assert ba.max_suspicious_actions == 3

class TestAnalyzeGazePattern:
    def test_offcenter_direction_increments_consecutive(self):
        ba = make_analyzer(min_consecutive=10)
        ba.analyze_gaze_pattern("left")
        assert ba.consecutive_offcenter == 1

    def test_center_direction_decrements_consecutive(self):
        ba = make_analyzer(min_consecutive=10)
        ba.consecutive_offcenter = 3
        ba.analyze_gaze_pattern("center")
        assert ba.consecutive_offcenter == 2

    def test_blink_counts_as_center(self):
        ba = make_analyzer(min_consecutive=10)
        ba.consecutive_offcenter = 2
        ba.analyze_gaze_pattern("blink")
        assert ba.consecutive_offcenter == 1

    def test_offcenter_ratio_trigger(self):
        ba = make_analyzer(min_consecutive=999, window_size=4)
        for _ in range(4):
            ba.analyze_gaze_pattern("left")
        assert ba.suspicious_actions >= 1

    def test_last_direction_updated(self):
        ba = make_analyzer()
        ba.analyze_gaze_pattern("down")
        assert ba.last_direction == "down"

    def test_consecutive_never_goes_negative(self):
        ba = make_analyzer()
        ba.consecutive_offcenter = 0
        for _ in range(5):
            ba.analyze_gaze_pattern("center")
        assert ba.consecutive_offcenter == 0

class TestAnalyzeHeadPose:
    def test_consecutive_never_goes_negative_head(self):
        ba = make_analyzer(head_min=10)
        ba._head_consecutive = 0
        for _ in range(3):
            ba.analyze_head_pose(make_head_info(is_suspicious=False))
        assert ba._head_consecutive == 0

    def test_consecutive_resets_after_trigger(self):
        ba = make_analyzer(head_min=3)
        for _ in range(3):
            ba.analyze_head_pose(make_head_info(is_suspicious=True))
        assert ba._head_consecutive == 0

    def test_head_history_window_size(self):
        ba = make_analyzer(window_size=5, head_min=999)
        for i in range(10):
            ba.analyze_head_pose(make_head_info(is_suspicious=False))
        assert len(ba._head_history) <= 5

class TestDetectCheating:
    def test_no_suspicious_actions_is_false(self):
        ba = make_analyzer(max_actions=1)
        assert ba.detect_cheating() is False

    def test_suspicious_equals_threshold_is_true(self):
        ba = make_analyzer(max_actions=1)
        ba.suspicious_actions = 1
        assert ba.detect_cheating() is True

class TestGenerateReport:
    def _make_report_with_triggers(self, gaze_t=1, head_t=0):
        ba = make_analyzer()
        ba.suspicious_actions = gaze_t + head_t
        ba._gaze_triggers = gaze_t
        ba._head_triggers = head_t
        ba.gaze_history = [{"direction": "left", "timestamp": time.time()}]
        ba._head_history = [{"direction": "center", "is_suspicious": False,
                             "yaw_deviation": 0.0, "pitch_deviation": 0.0,
                             "timestamp": time.time()}]
        return ba.generate_report()
    def test_report_has_required_keys(self):
        report = self._make_report_with_triggers()
        for key in ("suspicious_actions", "gaze_history", "head_history",
                    "current_status", "cheating_trigger",
                    "gaze_trigger_count", "head_trigger_count"):
            assert key in report, f"Отсутствует ключ: {key}"

    def test_counters_reset_after_report(self):
        ba = make_analyzer()
        ba.suspicious_actions = 3
        ba._gaze_triggers     = 2
        ba._head_triggers     = 1
        ba.generate_report()
        assert ba.suspicious_actions == 0
        assert ba._gaze_triggers     == 0
        assert ba._head_triggers     == 0

    def test_report_trigger_counts_match(self):
        ba = make_analyzer()
        ba._gaze_triggers = 3
        ba._head_triggers = 2
        report = ba.generate_report()
        assert report["gaze_trigger_count"] == 3
        assert report["head_trigger_count"] == 2