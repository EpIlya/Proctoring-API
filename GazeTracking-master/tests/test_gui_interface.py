import sys
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import main_with_head_tracking_model as mod

_cv2      = sys.modules["cv2"]
_winsound = sys.modules["winsound"]


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_app():
    app = MagicMock()
    app.sleep_interval = 0.01

    gt = MagicMock()
    gt.calibrated        = True
    gt.horizontal_center = 0.5
    gt.vertical_center   = 0.5
    gt.calibration_time  = 0.05
    cam = MagicMock(); cam.read.return_value = (True, _frame()); gt.camera = cam
    gt.get_frame_with_eyes_status.return_value = (_frame(), True)
    gt.get_gaze_direction.return_value = {"direction": "center"}
    app.gaze_tracker = gt

    ht = MagicMock()
    ht.neutral_yaw = 0.; ht.neutral_pitch = 0.
    ht.get_head_direction.return_value = {
        "direction": "center", "is_suspicious": False,
        "yaw_deviation": 0., "pitch_deviation": 0.}
    app.head_pose_tracker = ht

    ba = MagicMock()
    ba.detect_cheating.return_value = False
    ba.generate_report.return_value = {
        "suspicious_actions": 0, "gaze_history": [], "head_history": [],
        "current_status": "normal", "cheating_trigger": "unknown",
        "gaze_trigger_count": 0, "head_trigger_count": 0}
    app.behavior_analyzer = ba
    app.logger = MagicMock()
    app.recorder = MagicMock()
    app.model_referee = MagicMock()
    app.ui = MagicMock()
    app.participant_number = None
    return app


def make_gui(app=None):
    if app is None:
        app = _make_app()
    g = mod.GUIInterface.__new__(mod.GUIInterface)
    g.main_app               = app
    g.participant_number     = None
    g.window1                = None
    g.window2_active         = False
    g.window3                = None
    g.testing_active         = False
    g.calibration_completed  = False
    g.calibrating            = False
    g._testing_thread        = None
    g.root                   = MagicMock()
    g.button_window          = MagicMock()
    g.calibration_button     = MagicMock()
    g.progress_label         = MagicMock()
    g.calibration_overlay    = MagicMock()
    g.calibration_canvas     = MagicMock()
    return g


class TestInit:
    def test_defaults(self):
        g = make_gui()
        assert g.testing_active is False
        assert g.calibration_completed is False
        assert g._testing_thread is None

    def test_stores_main_app(self):
        app = _make_app()
        assert make_gui(app).main_app is app


class TestLogCheating:
    def setup_method(self):
        _winsound.Beep.reset_mock()

    def test_beep_called(self):
        g = make_gui()
        g.log_cheating()
        _winsound.Beep.assert_called_once_with(1000, 300)

    def test_gaze_log_appended(self):
        g = make_gui(); g.log_cheating()
        g.main_app.logger.gaze_logs.append.assert_called()

class TestEndTesting:
    def test_calls_stop(self):
        g = make_gui(); g.end_testing()
        g.main_app.stop.assert_called_once()

    def test_window3_destroyed(self):
        g = make_gui(); win3 = MagicMock(); g.window3 = win3
        g.end_testing(); win3.destroy.assert_called_once()

    def test_thread_joined_when_alive(self):
        g = make_gui(); t = MagicMock(); t.is_alive.return_value = True
        g._testing_thread = t; g.end_testing()
        t.join.assert_called_once()

class TestRunTestingLoop:
    def _run(self, gui, n=3, gaze_dir="center", detect_cheating=False,
             head_dir="center", cam_fail=False):
        counter = {"i": 0}

        def fake_sleep(t):
            if t < 0.4:
                counter["i"] += 1
                if counter["i"] >= n:
                    gui.testing_active = False

        gui.testing_active = True
        gui.main_app.gaze_tracker.get_gaze_direction.return_value = {"direction": gaze_dir}
        gui.main_app.behavior_analyzer.detect_cheating.return_value = detect_cheating
        gui.main_app.head_pose_tracker.get_head_direction.return_value = {
            "direction": head_dir, "is_suspicious": False,
            "yaw_deviation": 0., "pitch_deviation": 0.}

        if cam_fail:
            fail_cam = MagicMock(); fail_cam.read.return_value = (False, None)
            gui.main_app.gaze_tracker.camera = fail_cam
            new_cam = MagicMock(); new_cam.read.return_value = (False, None)
            original_vc = _cv2.VideoCapture
            _cv2.VideoCapture = MagicMock(return_value=new_cam)
            try:
                with patch("time.sleep", side_effect=fake_sleep):
                    gui.run_testing_loop()
            finally:
                _cv2.VideoCapture = original_vc
            return _cv2.VideoCapture  # вернём уже восстановленный, но звали ли — проверяем ниже
        else:
            with patch("time.sleep", side_effect=fake_sleep):
                gui.run_testing_loop()

    def test_loop_runs_and_stops(self):
        g = make_gui(); self._run(g, n=3)
        assert g.testing_active is False

    def test_gaze_analyzed_each_iteration(self):
        g = make_gui(); self._run(g, n=3)
        assert g.main_app.behavior_analyzer.analyze_gaze_pattern.call_count >= 3

    def test_head_analyzed_each_iteration(self):
        g = make_gui(); self._run(g, n=3)
        assert g.main_app.behavior_analyzer.analyze_head_pose.call_count >= 3

    def test_webcam_written_each_iteration(self):
        g = make_gui(); self._run(g, n=3)
        assert g.main_app.recorder.write_webcam_frame.call_count >= 3

    def test_cheating_log_behavior(self):
        g = make_gui(); self._run(g, n=2, detect_cheating=True)
        g.main_app.logger.log_behavior.assert_called()

    def test_exception_in_body_stops_loop(self):
        g = make_gui()
        g.main_app.gaze_tracker.get_gaze_direction.side_effect = RuntimeError("crash")
        g.testing_active = True
        with patch("time.sleep"):
            g.run_testing_loop()
        assert g.testing_active is True