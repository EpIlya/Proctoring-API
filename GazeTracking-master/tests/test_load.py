import sys
import time
import tracemalloc
import pytest
import numpy as np
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import main_with_head_tracking_model as mod

# Пороговые значения
FRAME_BUDGET_MS   = 50 # мс на кадр
MIN_FPS           = 20 # нижняя граница FPS
MEMORY_LEAK_MB    = 30 # допустимый рост RAM за длительный прогон
LONG_RUN_FRAMES   = 500 #кадров в длительном тесте
BURST_FRAMES      = 200 # кадров в обычном тесте
LOG_ITERS         = 10_000 # итераций для теста DataLogger

_cv2     = sys.modules["cv2"]
_fm_inst = sys.modules["mediapipe"].solutions.face_mesh.FaceMesh.return_value


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _face_results(detected=True):
    r = MagicMock()
    if not detected:
        r.multi_face_landmarks = None
        return r
    lm = MagicMock(); lm.x = 0.5; lm.y = 0.5
    face = MagicMock(); face.landmark = [lm] * 468
    r.multi_face_landmarks = [face]
    return r


def _setup_cv2_ok():
    _cv2.solvePnP.return_value  = (True, np.zeros((3, 1)), np.zeros((3, 1)))
    _cv2.Rodrigues.return_value = (np.eye(3, dtype=np.float64), None)


def _make_head_tracker():
    t = mod.HeadPoseTracker.__new__(mod.HeadPoseTracker)
    t._face_mesh       = _fm_inst
    t.neutral_yaw      = 0.0
    t.neutral_pitch    = 0.0
    t.calibrated       = True
    t.yaw_threshold    = 15.0
    t.pitch_threshold  = 15.0
    t._yaw_buf         = deque(maxlen=5)
    t._pitch_buf       = deque(maxlen=5)
    t._h_offcenter     = False
    t._v_offcenter     = False
    t._threshold_enter = 1.0
    t._threshold_exit  = 0.6
    t._calib_yaw_buf   = []
    t._calib_pitch_buf = []
    return t


def _make_gaze_tracker(calibrated=True):
    gt = mod.GazeTracker.__new__(mod.GazeTracker)
    gt.gaze              = MagicMock()
    gt.camera            = MagicMock()
    gt.debug             = False
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
    gt.gaze.annotated_frame.return_value = _frame()
    gt.gaze.horizontal_ratio.return_value = 0.5
    gt.gaze.vertical_ratio.return_value   = 0.5
    gt.camera.read.return_value = (True, _frame())
    return gt


def _make_recorder(tmp_path):
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

class TestHeadPoseTrackerLoad:

    def setup_method(self):
        _setup_cv2_ok()
        _fm_inst.process.return_value = _face_results(True)

    def test_process_frame_fps(self):
        tracker = _make_head_tracker()
        frame   = _frame()

        times = []
        for _ in range(BURST_FRAMES):
            t0 = time.perf_counter()
            tracker.process_frame(frame)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        fps    = 1000 / avg_ms if avg_ms > 0 else float("inf")
        p95_ms = sorted(times)[int(len(times) * 0.95)]

        print(f"\nHeadPose process_frame: avg={avg_ms:.2f}ms  fps={fps:.1f}  p95={p95_ms:.2f}ms")
        assert avg_ms < FRAME_BUDGET_MS, (
            f"Среднее время кадра {avg_ms:.2f}ms > лимит {FRAME_BUDGET_MS}ms"
        )
        assert fps >= MIN_FPS, f"FPS={fps:.1f} ниже {MIN_FPS}"

    def test_get_smooth_angles_fps(self):
        tracker = _make_head_tracker()
        frame   = _frame()

        t0 = time.perf_counter()
        for _ in range(BURST_FRAMES):
            tracker.get_smooth_angles(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed_ms / BURST_FRAMES

        print(f"\nHeadPose get_smooth_angles: avg={avg_ms:.2f}ms")
        assert avg_ms < FRAME_BUDGET_MS

    def test_get_head_direction_fps(self):
        tracker = _make_head_tracker()
        frame   = _frame()

        t0 = time.perf_counter()
        for _ in range(BURST_FRAMES):
            tracker.get_head_direction(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed_ms / BURST_FRAMES

        print(f"\nHeadPose get_head_direction: avg={avg_ms:.2f}ms")
        assert avg_ms < FRAME_BUDGET_MS

    def test_long_run_no_crash(self):
        tracker = _make_head_tracker()
        frame   = _frame()
        errors  = 0
        for i in range(LONG_RUN_FRAMES):
            try:
                tracker.get_head_direction(frame)
            except Exception:
                errors += 1
        assert errors == 0, f"Упало {errors} исключений за {LONG_RUN_FRAMES} кадров"

    def test_memory_no_leak(self):
        tracker = _make_head_tracker()
        frame   = _frame()

        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        for _ in range(LONG_RUN_FRAMES):
            tracker.get_head_direction(frame)

        snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats  = snapshot_end.compare_to(snapshot_start, "lineno")
        growth = sum(s.size_diff for s in stats if s.size_diff > 0) / 1024 / 1024

        print(f"\nHeadPose memory growth: {growth:.2f} MB")
        assert growth < MEMORY_LEAK_MB, f"Рост памяти {growth:.2f} MB > лимит {MEMORY_LEAK_MB} MB"

class TestGazeTrackerLoad:

    def test_detect_gaze_fps(self):
        gt = _make_gaze_tracker()
        times = []
        for _ in range(BURST_FRAMES):
            t0 = time.perf_counter()
            gt.detect_gaze(show_debug=False)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        fps    = 1000 / avg_ms if avg_ms > 0 else float("inf")
        p95_ms = sorted(times)[int(len(times) * 0.95)]

        print(f"\nGaze detect_gaze: avg={avg_ms:.2f}ms  fps={fps:.1f}  p95={p95_ms:.2f}ms")
        assert avg_ms < FRAME_BUDGET_MS
        assert fps >= MIN_FPS

    def test_get_frame_with_eyes_fps(self):
        gt = _make_gaze_tracker()
        gt.gaze.pupils_located = True

        t0 = time.perf_counter()
        for _ in range(BURST_FRAMES):
            gt.get_frame_with_eyes_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed_ms / BURST_FRAMES

        print(f"\nGaze get_frame_with_eyes_status: avg={avg_ms:.2f}ms")
        assert avg_ms < FRAME_BUDGET_MS

    def test_long_run_no_crash(self):
        gt = _make_gaze_tracker()
        errors = 0
        for _ in range(LONG_RUN_FRAMES):
            try:
                gt.detect_gaze(show_debug=False)
            except Exception:
                errors += 1
        assert errors == 0

    def test_memory_no_leak(self):
        gt = _make_gaze_tracker()

        tracemalloc.start()
        snap_start = tracemalloc.take_snapshot()
        for _ in range(LONG_RUN_FRAMES):
            gt.detect_gaze(show_debug=False)
        snap_end   = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats  = snap_end.compare_to(snap_start, "lineno")
        growth = sum(s.size_diff for s in stats if s.size_diff > 0) / 1024 / 1024
        print(f"\nGaze memory growth: {growth:.2f} MB")
        assert growth < MEMORY_LEAK_MB

class TestBehaviorAnalyzerLoad:

    DIRECTIONS = ["left", "right", "up", "down", "center", "blink"]

    def _make_analyzer(self):
        a = mod.BehaviorAnalyzer.__new__(mod.BehaviorAnalyzer)
        a.suspicious_actions    = 0
        a.max_suspicious_actions = 5
        a.gaze_history          = []
        a.analysis_window       = 5.0
        a.window_size           = 100
        a.min_consecutive_offcenter = 4
        a.offcenter_threshold   = 0.5
        a.last_direction        = "center"
        a.consecutive_offcenter = 0
        a.last_offcenter_time   = None
        a._head_consecutive     = 0
        a._head_min_consecutive = 6
        a._head_history         = []
        a._gaze_triggers        = 0
        a._head_triggers        = 0
        return a

    def _head_info(self, suspicious=False):
        return {
            "direction": "left" if suspicious else "center",
            "is_suspicious": suspicious,
            "yaw_deviation": 20.0 if suspicious else 2.0,
            "pitch_deviation": 5.0,
        }

    def test_analyze_gaze_throughput(self):
        a   = self._make_analyzer()
        t0  = time.perf_counter()
        for i in range(LOG_ITERS):
            direction = self.DIRECTIONS[i % len(self.DIRECTIONS)]
            a.analyze_gaze_pattern(direction)
        elapsed = time.perf_counter() - t0
        per_call = elapsed / LOG_ITERS * 1e6   # мкс

        print(f"\nBehaviorAnalyzer analyze_gaze: {per_call:.2f} мкс/вызов  total={elapsed:.2f}s")
        assert elapsed < 5.0, f"Слишком медленно: {elapsed:.2f}s для {LOG_ITERS} итераций"

    def test_analyze_head_throughput(self):
        a   = self._make_analyzer()
        t0  = time.perf_counter()
        for i in range(LOG_ITERS):
            a.analyze_head_pose(self._head_info(i % 10 == 0))
        elapsed = time.perf_counter() - t0
        per_call = elapsed / LOG_ITERS * 1e6

        print(f"\nBehaviorAnalyzer analyze_head: {per_call:.2f} мкс/вызов  total={elapsed:.2f}s")
        assert elapsed < 5.0

    def test_window_bounded(self):
        a = self._make_analyzer()
        for i in range(LOG_ITERS):
            a.analyze_gaze_pattern(self.DIRECTIONS[i % len(self.DIRECTIONS)])
            a.analyze_head_pose(self._head_info())
        assert len(a.gaze_history)   <= a.window_size, "gaze_history не ограничена!"
        assert len(a._head_history)  <= a.window_size, "_head_history не ограничена!"

    def test_memory_no_leak(self):
        a = self._make_analyzer()
        tracemalloc.start()
        snap_start = tracemalloc.take_snapshot()
        for i in range(LOG_ITERS):
            a.analyze_gaze_pattern(self.DIRECTIONS[i % len(self.DIRECTIONS)])
        snap_end = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats  = snap_end.compare_to(snap_start, "lineno")
        growth = sum(s.size_diff for s in stats if s.size_diff > 0) / 1024 / 1024
        print(f"\nBehaviorAnalyzer memory growth: {growth:.2f} MB")
        assert growth < MEMORY_LEAK_MB

class TestDataLoggerLoad:
    def _make_logger(self, tmp_path):
        lg = mod.DataLogger.__new__(mod.DataLogger)
        lg.gaze_logs       = []
        lg.behavior_logs   = []
        lg.logs_dir        = str(tmp_path)
        lg.gaze_log_file   = tmp_path / "gaze.log"
        lg.behavior_log_file = tmp_path / "behavior.json"
        return lg

    def test_log_gaze_throughput(self, tmp_path):
        lg = self._make_logger(tmp_path)
        gaze_info = {
            "horizontal_deviation": 0.05,
            "vertical_deviation": 0.03,
            "total_deviation": 0.06,
            "angle_from_real_center": 3.5,
            "angle_from_calibrated_center": 2.1,
        }
        t0 = time.perf_counter()
        for _ in range(LOG_ITERS):
            lg.log_gaze_data("center", gaze_info)
        elapsed = time.perf_counter() - t0
        per_call = elapsed / LOG_ITERS * 1e6

        print(f"\nDataLogger log_gaze: {per_call:.2f} мкс/вызов  total={elapsed:.2f}s")
        assert elapsed < 3.0

    def test_log_head_throughput(self, tmp_path):
        lg = self._make_logger(tmp_path)
        head_info = {"direction": "center", "yaw_deviation": 2.0, "pitch_deviation": 1.0, "is_suspicious": False}
        t0 = time.perf_counter()
        for _ in range(LOG_ITERS):
            lg.log_head_data(head_info)
        elapsed = time.perf_counter() - t0
        print(f"\nDataLogger log_head: {elapsed / LOG_ITERS * 1e6:.2f} мкс/вызов  total={elapsed:.2f}s")
        assert elapsed < 3.0

    def test_save_logs_to_file_speed(self, tmp_path):
        lg = self._make_logger(tmp_path)
        for i in range(1000):
            lg.log_gaze_data("center")
            lg.log_behavior({"event_type": "observation", "label": 0})
        t0 = time.perf_counter()
        lg.save_logs_to_file()
        elapsed = time.perf_counter() - t0
        print(f"\nDataLogger save_logs_to_file (1000 entries): {elapsed*1000:.2f}ms")
        assert elapsed < 1.0

    def test_memory_no_leak(self, tmp_path):
        lg = self._make_logger(tmp_path)
        tracemalloc.start()
        snap_start = tracemalloc.take_snapshot()
        for batch in range(50):
            for _ in range(200):
                lg.log_gaze_data("center")
            lg.save_logs_to_file()
        snap_end = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats  = snap_end.compare_to(snap_start, "lineno")
        growth = sum(s.size_diff for s in stats if s.size_diff > 0) / 1024 / 1024
        print(f"\nDataLogger memory growth: {growth:.2f} MB")
        assert growth < MEMORY_LEAK_MB


class TestSessionRecorderLoad:
    def test_write_with_timing_throughput(self, tmp_path):
        r = _make_recorder(tmp_path)
        r.recording_active = True
        w = _good_writer()
        frame = _frame()

        t0 = time.perf_counter()
        for _ in range(LONG_RUN_FRAMES):
            r._write_with_timing(w, frame, "_last_screen_time", "_screen_accum", "_screen_frame_idx")
        elapsed = time.perf_counter() - t0
        per_call = elapsed / LONG_RUN_FRAMES * 1e6

        print(f"\nSessionRecorder _write_with_timing: {per_call:.2f} мкс/вызов  total={elapsed:.2f}s")
        assert elapsed < 5.0

    def test_write_frame_counter_correct(self, tmp_path):
        r = _make_recorder(tmp_path)
        r.recording_active = True
        w = _good_writer()
        frame = _frame()

        r._write_with_timing(w, frame, "_last_screen_time", "_screen_accum", "_screen_frame_idx")
        written_by_writer = w.write.call_count
        assert r._screen_frame_idx == written_by_writer

    def test_write_concurrent_screen_webcam(self, tmp_path):
        r = _make_recorder(tmp_path)
        r.recording_active = True
        sw = _good_writer(); ww = _good_writer()
        r.screen_writer = sw; r.webcam_writer = ww
        frame = _frame()

        for _ in range(BURST_FRAMES):
            r.write_screen_frame(frame)
            r.write_webcam_frame(frame)

        assert r._screen_frame_idx == sw.write.call_count
        assert r._webcam_frame_idx == ww.write.call_count

    def test_memory_no_leak(self, tmp_path):
        r = _make_recorder(tmp_path)
        r.recording_active = True
        w = _good_writer()
        frame = _frame()

        tracemalloc.start()
        snap_start = tracemalloc.take_snapshot()
        for _ in range(LONG_RUN_FRAMES):
            r._write_with_timing(w, frame, "_last_screen_time", "_screen_accum", "_screen_frame_idx")
        snap_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats  = snap_end.compare_to(snap_start, "lineno")
        growth = sum(s.size_diff for s in stats if s.size_diff > 0) / 1024 / 1024
        print(f"\nSessionRecorder memory growth: {growth:.2f} MB")
        assert growth < MEMORY_LEAK_MB

class TestIntegratedLoad:

    DIRECTIONS = ["left", "right", "up", "down", "center", "blink"]

    def test_full_pipeline_fps(self, tmp_path):
        gt = _make_gaze_tracker()

        ba = mod.BehaviorAnalyzer.__new__(mod.BehaviorAnalyzer)
        ba.suspicious_actions      = 0
        ba.max_suspicious_actions  = 5
        ba.gaze_history            = []
        ba.analysis_window         = 5.0
        ba.window_size             = 100
        ba.min_consecutive_offcenter = 4
        ba.offcenter_threshold     = 0.5
        ba.last_direction          = "center"
        ba.consecutive_offcenter   = 0
        ba.last_offcenter_time     = None
        ba._head_consecutive       = 0
        ba._head_min_consecutive   = 6
        ba._head_history           = []
        ba._gaze_triggers          = 0
        ba._head_triggers          = 0

        lg = mod.DataLogger.__new__(mod.DataLogger)
        lg.gaze_logs           = []
        lg.behavior_logs       = []
        lg.logs_dir            = str(tmp_path)
        lg.gaze_log_file       = tmp_path / "gaze.log"
        lg.behavior_log_file   = tmp_path / "behavior.json"

        t0 = time.perf_counter()
        for i in range(LONG_RUN_FRAMES):
            direction = gt.detect_gaze(show_debug=False) or "center"
            gaze_info = gt.get_gaze_direction()
            ba.analyze_gaze_pattern(direction, gaze_info)
            lg.log_gaze_data(direction, gaze_info)
            if i % 100 == 0:
                lg.save_logs_to_file()

        elapsed = time.perf_counter() - t0
        fps = LONG_RUN_FRAMES / elapsed

        print(f"\nFull pipeline: {elapsed:.2f}s for {LONG_RUN_FRAMES} frames → FPS={fps:.1f}")
        assert fps >= MIN_FPS, f"Pipeline FPS={fps:.1f} ниже MIN_FPS={MIN_FPS}"

    def test_full_pipeline_no_crash(self, tmp_path):
        gt = _make_gaze_tracker()
        errors = 0
        for _ in range(LONG_RUN_FRAMES):
            try:
                direction = gt.detect_gaze(show_debug=False) or "center"
                gt.get_gaze_direction()
            except Exception:
                errors += 1
        assert errors == 0
