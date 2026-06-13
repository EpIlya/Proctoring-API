import sys
import pytest
import numpy as np
from collections import deque
from unittest.mock import MagicMock, patch

import main_with_head_tracking_model as mod

_cv2     = sys.modules["cv2"]
_fm_inst = sys.modules["mediapipe"].solutions.face_mesh.FaceMesh.return_value


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _face_results(detected=True):
    r = MagicMock()
    if not detected:
        r.multi_face_landmarks = None
        return r
    lm = MagicMock(); lm.x = 0.5; lm.y = 0.5
    face = MagicMock(); face.landmark = [lm] * 468
    r.multi_face_landmarks = [face]
    return r


def make_tracker():
    t = mod.HeadPoseTracker.__new__(mod.HeadPoseTracker)
    t._face_mesh       = _fm_inst
    t.neutral_yaw      = 0.0
    t.neutral_pitch    = 0.0
    t.calibrated       = False
    t.yaw_threshold    = 15.0
    t.pitch_threshold  = 15.0
    t._yaw_buf         = __import__("collections").deque(maxlen=5)
    t._pitch_buf       = __import__("collections").deque(maxlen=5)
    t._h_offcenter     = False
    t._v_offcenter     = False
    t._threshold_enter = 1.0
    t._threshold_exit  = 0.6
    t._calib_yaw_buf   = []
    t._calib_pitch_buf = []
    return t


def _good_pnp():
    _cv2.solvePnP.return_value  = (True, np.zeros((3, 1)), np.zeros((3, 1)))
    _cv2.Rodrigues.return_value = (np.eye(3, dtype=np.float64), None)


class TestInit:
    def test_defaults(self):
        t = make_tracker()
        assert t.calibrated is False
        assert t.neutral_yaw == 0.0 and t.neutral_pitch == 0.0
        assert t._calib_yaw_buf == [] and t._calib_pitch_buf == []


class TestProcessFrame:
    def setup_method(self):
        _good_pnp()

    def test_returns_none_when_no_face(self):
        t = make_tracker()
        _fm_inst.process.return_value = _face_results(False)
        assert t.process_frame(_frame()) is None

    def test_returns_tuple_when_face_detected(self):
        t = make_tracker()
        _fm_inst.process.return_value = _face_results(True)
        out = t.process_frame(_frame())
        assert isinstance(out, tuple) and len(out) == 2

    def test_returns_none_when_solvePnP_fails(self):
        t = make_tracker()
        _fm_inst.process.return_value = _face_results(True)
        _cv2.solvePnP.return_value = (False, None, None)
        assert t.process_frame(_frame()) is None

class TestGetSmoothAngles:
    def setup_method(self):
        _good_pnp()

    def test_returns_tuple_when_face_detected(self):
        t = make_tracker()
        _fm_inst.process.return_value = _face_results(True)
        out = t.get_smooth_angles(_frame())
        assert isinstance(out, tuple) and len(out) == 2

    def test_smoothing_averages_buffer(self):
        t = make_tracker()
        _fm_inst.process.return_value = _face_results(True)
        for _ in range(5):
            out = t.get_smooth_angles(_frame())
        yaw, pitch = out
        assert isinstance(yaw, float) and isinstance(pitch, float)

class TestCalibration:
    def setup_method(self):
        _good_pnp()

    def test_accumulate_no_face_ignored(self):
        t = make_tracker()
        _fm_inst.process.return_value = _face_results(False)
        t.accumulate_calibration(_frame())
        assert t._calib_yaw_buf == [] and t._calib_pitch_buf == []

    def test_finalize_sets_calibrated(self):
        t = make_tracker()
        t._calib_yaw_buf = [5., 10.]; t._calib_pitch_buf = [2., 4.]
        assert t.finalize_calibration() is True and t.calibrated is True

    def test_finalize_computes_mean(self):
        t = make_tracker()
        t._calib_yaw_buf = [10., 20.]; t._calib_pitch_buf = [4., 8.]
        t.finalize_calibration()
        assert t.neutral_yaw == pytest.approx(15.) and t.neutral_pitch == pytest.approx(6.)

class TestGetHeadDirection:
    def setup_method(self):
        _good_pnp()

    def test_no_face_not_detected(self):
        t = make_tracker(); t.calibrated = True
        _fm_inst.process.return_value = _face_results(False)
        r = t.get_head_direction(_frame())
        assert r["direction"] == "not detected" and r["is_suspicious"] is False

    def test_direction_right(self):
        t = make_tracker(); t.calibrated = True; t.yaw_threshold = 5.
        with patch.object(t, "get_smooth_angles", return_value=(20., 0.)):
            r = t.get_head_direction(_frame())
        assert "right" in r["direction"] and r["is_suspicious"] is True

    def test_direction_left(self):
        t = make_tracker(); t.calibrated = True; t.yaw_threshold = 5.
        with patch.object(t, "get_smooth_angles", return_value=(-20., 0.)):
            r = t.get_head_direction(_frame())
        assert "left" in r["direction"]

    def test_direction_down(self):
        t = make_tracker(); t.calibrated = True; t.pitch_threshold = 5.
        with patch.object(t, "get_smooth_angles", return_value=(0., 20.)):
            r = t.get_head_direction(_frame())
        assert "down" in r["direction"]

    def test_direction_up(self):
        t = make_tracker(); t.calibrated = True; t.pitch_threshold = 5.
        with patch.object(t, "get_smooth_angles", return_value=(0., -20.)):
            r = t.get_head_direction(_frame())
        assert "up" in r["direction"]

    def test_deviation_values(self):
        t = make_tracker(); t.calibrated = True
        t.neutral_yaw = 5.; t.neutral_pitch = 2.
        with patch.object(t, "get_smooth_angles", return_value=(15., 7.)):
            r = t.get_head_direction(_frame())
        assert r["yaw_deviation"] == pytest.approx(10.) and r["pitch_deviation"] == pytest.approx(5.)


class TestRelease:
    def test_close_called(self):
        t = make_tracker(); t.release()
        t._face_mesh.close.assert_called_once()
