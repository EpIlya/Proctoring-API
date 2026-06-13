import sys
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import main_with_head_tracking_model as mod

_cv2     = sys.modules["cv2"]
_mss_mod = sys.modules["mss"]
_mss_sct = _mss_mod.mss.return_value


@pytest.fixture
def recorder(tmp_path):
    r = mod.SessionRecorder.__new__(mod.SessionRecorder)
    r.recordings_dir     = tmp_path / "rec"
    r.recordings_dir.mkdir(parents=True, exist_ok=True)
    r.recording_fps      = 10
    r.screen_file_path   = None
    r.webcam_file_path   = None
    r.combined_file_path = None
    r.screen_writer      = None
    r.webcam_writer      = None
    r.screen_size        = None
    r.webcam_size        = None
    r.session_timestamp  = None
    r.recording_active   = False
    r.session_start_time = None
    r.session_end_time   = None
    r._screen_frame_idx  = 0
    r._webcam_frame_idx  = 0
    r._last_screen_time  = None
    r._last_webcam_time  = None
    r._screen_accum      = 0.0
    r._webcam_accum      = 0.0
    return r


def _good_writer():
    w = MagicMock(); w.isOpened.return_value = True; return w


def _bad_writer():
    w = MagicMock(); w.isOpened.return_value = False; return w


def _setup_good_writers(n=2):
    _cv2.VideoWriter.reset_mock()
    _cv2.VideoWriter.return_value = _good_writer()


def _setup_bad_writers():
    _cv2.VideoWriter.reset_mock()
    _cv2.VideoWriter.return_value = _bad_writer()

class TestStartRecording:
    def test_paths_set(self, recorder):
        _setup_good_writers()
        recorder.start_recording((640, 480), "5")
        assert recorder.screen_file_path   is not None
        assert recorder.webcam_file_path   is not None
        assert recorder.combined_file_path is not None

    def test_session_start_time_set(self, recorder):
        _setup_good_writers(); before = time.time()
        recorder.start_recording((640, 480), "1")
        assert recorder.session_start_time >= before


class TestCaptureScreenFrame:
    def test_none_when_not_active(self, recorder):
        recorder.recording_active = False
        assert recorder.capture_screen_frame() is None

    def test_none_when_no_screen_size(self, recorder):
        recorder.recording_active = True; recorder.screen_size = None
        assert recorder.capture_screen_frame() is None

    def test_returns_frame_when_active(self, recorder):
        recorder.recording_active = True; recorder.screen_size = (1920, 1080)
        frame = recorder.capture_screen_frame()
        assert frame is not None

    def test_returns_none_on_mss_exception(self, recorder):
        recorder.recording_active = True; recorder.screen_size = (1920, 1080)
        original_grab = _mss_sct.grab
        _mss_sct.grab = MagicMock(side_effect=RuntimeError("grab fail"))
        try:
            result = recorder.capture_screen_frame()
        finally:
            _mss_sct.grab = original_grab
        assert result is None

class TestStopRecording:
    def test_noop_when_not_active(self, recorder):
        recorder.recording_active = False
        recorder.stop_recording()

    def test_releases_writers(self, recorder):
        recorder.recording_active = True
        sw = _good_writer(); ww = _good_writer()
        recorder.screen_writer = sw; recorder.webcam_writer = ww
        recorder.stop_recording()
        sw.release.assert_called_once(); ww.release.assert_called_once()

    def test_sets_end_time(self, recorder):
        recorder.recording_active = True
        recorder.screen_writer = _good_writer(); recorder.webcam_writer = _good_writer()
        before = time.time(); recorder.stop_recording()
        assert recorder.session_end_time >= before

    def test_writers_none_after_stop(self, recorder):
        recorder.recording_active = True
        recorder.screen_writer = _good_writer(); recorder.webcam_writer = _good_writer()
        recorder.stop_recording()
        assert recorder.screen_writer is None and recorder.webcam_writer is None