import sys
import math
import pytest
import numpy as np
from collections import deque
from unittest.mock import MagicMock, patch

import main_with_head_tracking_model as mod

_cv2 = sys.modules["cv2"]


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _cam(ok=True):
    c = MagicMock()
    c.read.return_value = (ok, _frame() if ok else None)
    return c


def make_gt(calibrated=True, debug=False):
    gt = mod.GazeTracker.__new__(mod.GazeTracker)
    gt.gaze              = MagicMock()
    gt.camera            = None
    gt.debug             = debug
    gt.debug_window_size = (800, 600)
    gt.calibration_threshold = 0.10
    gt.horizontal_center = 0.5
    gt.vertical_center   = 0.5
    gt.calibrated        = calibrated
    gt.calibration_time  = 10
    gt._h_buffer         = deque(maxlen=7)
    gt._v_buffer         = deque(maxlen=7)
    gt.threshold_enter   = 0.13
    gt.threshold_exit    = 0.07
    gt._h_offcenter      = False
    gt._v_offcenter      = False
    return gt



class TestGetGazeDirection:
    def _setup(self, gt, h, v):
        gt.gaze.horizontal_ratio.return_value = h
        gt.gaze.vertical_ratio.return_value   = v
        gt._h_buffer.clear(); gt._v_buffer.clear()
        gt._h_offcenter = gt._v_offcenter = False
    def test_center(self):
        gt = make_gt(); self._setup(gt, 0.5, 0.5)
        assert gt.get_gaze_direction()["direction"] == "center"

    def test_left(self):
        # h_diff = smooth_h - 0.5 > threshold_enter → "left"
        gt = make_gt(); self._setup(gt, 0.7, 0.5)
        assert "left" in gt.get_gaze_direction()["direction"]

    def test_right(self):
        gt = make_gt(); self._setup(gt, 0.3, 0.5)
        assert "right" in gt.get_gaze_direction()["direction"]

    def test_up(self):
        gt = make_gt(); self._setup(gt, 0.5, 0.3)
        assert "up" in gt.get_gaze_direction()["direction"]

    def test_down(self):
        gt = make_gt(); self._setup(gt, 0.5, 0.7)
        assert "down" in gt.get_gaze_direction()["direction"]

    def test_below_threshold_stays_center(self):
        gt = make_gt(); self._setup(gt, 0.55, 0.5)
        assert gt.get_gaze_direction()["direction"] == "center"

    def test_hysteresis_exit(self):
        gt = make_gt(); gt._h_offcenter = True
        self._setup(gt, 0.503, 0.5)
        gt.get_gaze_direction()
        assert gt._h_offcenter is False

class TestDetectGaze:
    def test_camera_read_fail_triggers_reinit(self):
        gt = make_gt(calibrated=True)
        fail_cam = _cam(False); gt.camera = fail_cam
        new_cam = _cam(True)
        gt.gaze.annotated_frame.return_value = _frame()
        gt.gaze.horizontal_ratio.return_value = 0.5
        gt.gaze.vertical_ratio.return_value   = 0.5
        original_vc = _cv2.VideoCapture
        try:
            _cv2.VideoCapture = MagicMock(return_value=new_cam)
            with patch("time.sleep"):
                gt.detect_gaze(show_debug=False)
            assert _cv2.VideoCapture.called
        finally:
            _cv2.VideoCapture = original_vc

class TestReleaseCamera:
    def test_releases_camera(self):
        gt = make_gt(); cam = MagicMock(); gt.camera = cam; gt.debug = False
        gt.release_camera()
        cam.release.assert_called_once()