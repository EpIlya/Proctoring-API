import cv2
from gaze_tracking import GazeTracking
import time
import json
import math
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import threading
import winsound
from collections import deque
from mss import mss
import numpy as np
import mediapipe as mp
import joblib
import uuid
import os as _os


_TEST_MODE: bool = _os.environ.get("PROCTORING_TEST_MODE", "0").strip() == "1"
if _TEST_MODE:
    print("[PROCTORING] Тестовый режим активен (PROCTORING_TEST_MODE=1)")


from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

def load_config(config_path: str = "config.json") -> Dict:
    default_config = {
        "debug": True,
        "debug_window_size": [800, 600],
        "calibration_threshold": 0.10,
        "max_suspicious_actions": 1,
        "logs_dir": "logs",
        "gaze_log_file": "gaze_log.txt",
        "behavior_log_file": "behavior_log.json",
        "model_log_file": "behavior_log_with_model.json",
        "model_pkl_file": "best_model.pkl",
        "calibration_time": 10,
        "analysis_window": 5,
        "sleep_interval": 0.33,
        "recordings_dir": "recordings",
        "recording_fps": 10,
        "head_yaw_threshold": 15.0,
        "head_pitch_threshold": 15.0,
        "head_min_consecutive": 6,
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return {**default_config, **config}
    except FileNotFoundError:
        print(f"Файл конфигурации {config_path} не найден, используются значения по умолчанию")
        return default_config

CONFIG = load_config()
Path(CONFIG["logs_dir"]).mkdir(parents=True, exist_ok=True)




class HeadPoseTracker:
    """Отслеживание положения головы."""
    _LANDMARKS_IDS = [1, 33, 263, 61, 291, 199]
    _MODEL_POINTS = np.array([
        [0.0,    0.0,    0.0   ],
        [-225.0, 170.0, -135.0],
        [225.0,  170.0, -135.0],
        [-150.0,-150.0, -125.0],
        [150.0, -150.0, -125.0],
        [0.0,  -330.0,  -65.0 ],
    ], dtype=np.float64)

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.neutral_yaw: float = 0.0
        self.neutral_pitch: float = 0.0
        self.calibrated: bool = False
        self.yaw_threshold: float = CONFIG["head_yaw_threshold"]
        self.pitch_threshold: float = CONFIG["head_pitch_threshold"]
        self._yaw_buf = deque(maxlen=5)
        self._pitch_buf = deque(maxlen=5)
        self._h_offcenter = False
        self._v_offcenter = False
        self._threshold_enter = 1.0
        self._threshold_exit = 0.6
        self._calib_yaw_buf: List[float] = []
        self._calib_pitch_buf: List[float] = []

    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        image_points = np.array([
            [landmarks[idx].x * w, landmarks[idx].y * h]
            for idx in self._LANDMARKS_IDS
        ], dtype=np.float64)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
            yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        return yaw, pitch

    def get_smooth_angles(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        result = self.process_frame(frame)
        if result is None:
            return None
        yaw, pitch = result
        self._yaw_buf.append(yaw)
        self._pitch_buf.append(pitch)
        return sum(self._yaw_buf) / len(self._yaw_buf), sum(self._pitch_buf) / len(self._pitch_buf)

    def accumulate_calibration(self, frame: np.ndarray) -> None:
        result = self.process_frame(frame)
        if result is not None:
            self._calib_yaw_buf.append(result[0])
            self._calib_pitch_buf.append(result[1])

    def finalize_calibration(self) -> bool:
        if self._calib_yaw_buf and self._calib_pitch_buf:
            self.neutral_yaw   = sum(self._calib_yaw_buf)   / len(self._calib_yaw_buf)
            self.neutral_pitch = sum(self._calib_pitch_buf) / len(self._calib_pitch_buf)
            self.calibrated = True
            return True
        return False

    def get_head_direction(self, frame: np.ndarray) -> Dict[str, Any]:
        if not self.calibrated:
            return {"direction": "not calibrated", "is_suspicious": False}
        angles = self.get_smooth_angles(frame)
        if angles is None:
            return {"direction": "not detected", "yaw": None, "pitch": None,
                    "yaw_deviation": None, "pitch_deviation": None, "is_suspicious": False}
        yaw, pitch = angles
        yaw_dev   = yaw   - self.neutral_yaw
        pitch_dev = pitch - self.neutral_pitch
        if not self._h_offcenter and abs(yaw_dev)   > self.yaw_threshold   * self._threshold_enter:
            self._h_offcenter = True
        elif self._h_offcenter and abs(yaw_dev)     < self.yaw_threshold   * self._threshold_exit:
            self._h_offcenter = False
        if not self._v_offcenter and abs(pitch_dev) > self.pitch_threshold * self._threshold_enter:
            self._v_offcenter = True
        elif self._v_offcenter and abs(pitch_dev)   < self.pitch_threshold * self._threshold_exit:
            self._v_offcenter = False
        parts = []
        if self._h_offcenter:
            parts.append("right" if yaw_dev   > 0 else "left")
        if self._v_offcenter:
            parts.append("down"  if pitch_dev > 0 else "up")
        direction = " ".join(parts) if parts else "center"
        return {
            "direction": direction,
            "yaw": yaw, "pitch": pitch,
            "yaw_deviation": yaw_dev,
            "pitch_deviation": pitch_dev,
            "is_suspicious": bool(parts),
        }

    def release(self):
        self._face_mesh.close()


class ModelReferee:
    """Принимает данные наблюдения, извлекает фичи, запрашивает предсказание."""

    DIRECTION_MAP = {
        "center": 0, "up": 1, "down": 2, "left": 3, "right": 4,
        "left up": 5, "right up": 6, "left down": 7, "right down": 8,
        "blink": 9, "not detected": 10,
    }
    TRIGGER_MAP = {"gaze": 0, "head_pose": 1, "gaze_and_head": 2}

    def __init__(self):
        self.logs_dir  = Path(CONFIG["logs_dir"])
        self.log_file  = self.logs_dir / CONFIG.get("model_log_file", "behavior_log_with_model.json")
        self.pkl_path  = Path(CONFIG.get("model_pkl_file", "best_model.pkl"))
        self._model        = None
        self._preprocessor = None
        self._feat_names   = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        base      = self.pkl_path.parent
        model_path = self.pkl_path
        prep_path  = base / "preprocessor.pkl"
        feat_path  = base / "feature_names.pkl"
        missing = [p for p in (model_path, prep_path, feat_path) if not p.exists()]
        if missing:
            print(f"[ModelReferee] Файлы не найдены: {[str(p) for p in missing]}")
            return
        try:
            self._model        = joblib.load(model_path)
            self._preprocessor = joblib.load(prep_path)
            self._feat_names   = joblib.load(feat_path)
            self._model_loaded = True
            print(f"[ModelReferee] Модель загружена: {model_path.name}")
        except Exception as e:
            print(f"[ModelReferee] Ошибка загрузки модели: {e}")

    def _extract_features(self, report: Dict, calib: Dict):
        import pandas as pd
        dmap = self.DIRECTION_MAP
        tmap = self.TRIGGER_MAP
        gh = report.get("gaze_history", [])
        hh = report.get("head_history", [])
        g_dirs  = [dmap.get(f.get("direction", ""), 0)     for f in gh if isinstance(f, dict)]
        g_tdevs = [f["total_deviation"]                     for f in gh if isinstance(f, dict) and f.get("total_deviation") is not None]
        g_acals = [f["angle_from_calibrated_center"]        for f in gh if isinstance(f, dict) and f.get("angle_from_calibrated_center") is not None]
        h_dirs  = [dmap.get(f.get("direction", ""), 0)     for f in hh if isinstance(f, dict)]
        h_yaws  = [abs(f["yaw_deviation"])                  for f in hh if isinstance(f, dict) and f.get("yaw_deviation") is not None]
        h_pitch = [abs(f["pitch_deviation"])                for f in hh if isinstance(f, dict) and f.get("pitch_deviation") is not None]
        h_susp  = [int(f.get("is_suspicious", False))       for f in hh if isinstance(f, dict)]
        n_g = max(len(g_dirs), 1)
        n_h = max(len(h_dirs), 1)
        def safe_mean(lst): return float(np.nanmean(lst)) if lst else np.nan
        def safe_std(lst):  return float(np.nanstd(lst))  if lst else np.nan
        def safe_max(lst):  return float(np.nanmax(lst))  if lst else np.nan
        def safe_min(lst):  return float(np.nanmin(lst))  if lst else np.nan
        row = {
            "suspicious_actions":         report.get("suspicious_actions"),
            "cheating_trigger":           tmap.get(report.get("cheating_trigger"), np.nan),
            "gaze_trigger_count":         report.get("gaze_trigger_count"),
            "head_trigger_count":         report.get("head_trigger_count"),
            "calib_horizontal_ratio":     calib.get("horizontal_ratio"),
            "calib_vertical_ratio":       calib.get("vertical_ratio"),
            "calib_head_neutral_yaw":     calib.get("head_neutral_yaw"),
            "calib_head_neutral_pitch":   calib.get("head_neutral_pitch"),
            "gaze_num_mean":              safe_mean(g_tdevs),
            "gaze_num_std":               safe_std(g_tdevs),
            "gaze_num_max":               safe_max(g_tdevs),
            "gaze_num_min":               safe_min(g_tdevs),
            "gaze_total_dev_mean":        safe_mean(g_tdevs),
            "gaze_total_dev_std":         safe_std(g_tdevs),
            "gaze_total_dev_max":         safe_max(g_tdevs),
            "gaze_calib_angle_mean":      safe_mean(g_acals),
            "gaze_calib_angle_std":       safe_std(g_acals),
            "gaze_dir_non_center_ratio":  sum(1 for d in g_dirs if d != 0) / n_g,
            "gaze_dir_up_ratio":          sum(1 for d in g_dirs if d == 1) / n_g,
            "gaze_dir_down_ratio":        sum(1 for d in g_dirs if d == 2) / n_g,
            "gaze_dir_blink_ratio":       sum(1 for d in g_dirs if d == 9) / n_g,
            "gaze_n_frames":              float(len(g_dirs)),
            "head_num_mean":              safe_mean(h_yaws + h_pitch),
            "head_num_std":               safe_std(h_yaws + h_pitch),
            "head_yaw_abs_mean":          safe_mean(h_yaws),
            "head_yaw_abs_max":           safe_max(h_yaws),
            "head_pitch_abs_mean":        safe_mean(h_pitch),
            "head_pitch_abs_max":         safe_max(h_pitch),
            "head_dir_non_center_ratio":  sum(1 for d in h_dirs if d != 0) / n_h,
            "head_dir_up_ratio":          sum(1 for d in h_dirs if d == 1) / n_h,
            "head_dir_down_ratio":        sum(1 for d in h_dirs if d == 2) / n_h,
            "head_n_frames":              float(len(h_dirs)),
            "head_is_suspicious_ratio":   float(np.mean(h_susp)) if h_susp else np.nan,
            "head_is_suspicious_any":     float(int(any(h_susp))) if h_susp else 0.0,
        }
        import pandas as pd
        df_row = pd.DataFrame([row])
        for col in self._feat_names:
            if col not in df_row.columns:
                df_row[col] = np.nan
        return df_row[self._feat_names]

    def predict(self, report: Dict, calib: Dict) -> Dict[str, Any]:
        if not self._model_loaded:
            return {"model_verdict": "model_unavailable", "model_probability": None}
        try:
            X_row  = self._extract_features(report, calib)
            X_proc = self._preprocessor.transform(X_row)
            pred   = int(self._model.predict(X_proc)[0])
            proba  = None
            if hasattr(self._model, "predict_proba"):
                proba = round(float(self._model.predict_proba(X_proc)[0][1]), 4)
            verdict = "cheating" if pred == 1 else "normal"
            return {"model_verdict": verdict, "model_probability": proba}
        except Exception as e:
            print(f"[ModelReferee] Ошибка предсказания: {e}")
            return {"model_verdict": "model_unavailable", "model_probability": None}

    def log_verdict(self, report: Dict, calib: Dict) -> None:
        prediction = self.predict(report, calib)
        data_with_verdict = dict(report)
        data_with_verdict["model_verdict"]       = prediction["model_verdict"]
        data_with_verdict["model_probability"]   = prediction["model_probability"]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        entry = {"timestamp": timestamp, "data": data_with_verdict}
        try:
            existing = json.loads(self.log_file.read_text(encoding="utf-8")) if self.log_file.exists() else []
        except (json.JSONDecodeError, IOError):
            existing = []
        existing.append(entry)
        self.log_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


class GazeTracker:
    def __init__(self, debug: bool = True, calibration_threshold: float = 0.10):
        self.gaze = GazeTracking()
        self.camera = None
        self.debug = CONFIG["debug"] if debug is None else debug
        self.debug_window_size = tuple(CONFIG["debug_window_size"])
        self.calibration_threshold = CONFIG["calibration_threshold"] if calibration_threshold is None else calibration_threshold
        self.horizontal_center = 0.5
        self.vertical_center   = 0.5
        self.calibrated = False
        self.calibration_time = CONFIG["calibration_time"]
        self._h_buffer = deque(maxlen=7)
        self._v_buffer = deque(maxlen=7)
        self.threshold_enter = 0.13
        self.threshold_exit  = 0.07
        self._h_offcenter = False
        self._v_offcenter = False

    def initialize_camera(self) -> None:
        self.camera = cv2.VideoCapture(0)
        self.calibrate()

    def initialize_camera_without_calibration(self) -> None:
        self.camera = cv2.VideoCapture(0)

    def detect_gaze(self, show_debug: bool = None) -> Optional[str]:
        if self.camera is None:
            raise ValueError("Камера не инициализирована.")
        ret, frame = self.camera.read()
        if not ret or frame is None:
            try:
                self.camera.release()
            except Exception:
                pass
            time.sleep(0.1)
            self.camera = cv2.VideoCapture(0)
            time.sleep(0.2)
            ret, frame = self.camera.read()
            if not ret or frame is None:
                return None
        self.gaze.refresh(frame)
        gaze_info = self.get_gaze_direction()
        direction = gaze_info["direction"]
        return direction if direction != "not calibrated" else None

    def get_frame_with_eyes_status(self) -> Tuple[Optional[Any], bool]:
        if self.camera is None:
            return None, False
        ret, frame = self.camera.read()
        if not ret or frame is None:
            try:
                self.camera.release()
            except Exception:
                pass
            time.sleep(0.1)
            self.camera = cv2.VideoCapture(0)
            time.sleep(0.2)
            ret, frame = self.camera.read()
            if not ret or frame is None:
                return None, False
        self.gaze.refresh(frame)
        frame = self.gaze.annotated_frame()
        eyes_detected = self.gaze.pupils_located
        return frame, eyes_detected

    def release_camera(self) -> None:
        if self.camera is not None:
            self.camera.release()

    def calibrate(self, auto_start: bool = False) -> None:
        print("Калибровка начинается...")
        time.sleep(2)
        start_time = time.time()
        horizontal_values = []
        vertical_values   = []
        while time.time() - start_time < self.calibration_time:
            _, frame = self.camera.read()
            self.gaze.refresh(frame)
            if self.gaze.horizontal_ratio() is not None:
                horizontal_values.append(self.gaze.horizontal_ratio())
            if self.gaze.vertical_ratio() is not None:
                vertical_values.append(self.gaze.vertical_ratio())
            time.sleep(0.1)
        if horizontal_values and vertical_values:
            self.horizontal_center = sum(horizontal_values) / len(horizontal_values)
            self.vertical_center   = sum(vertical_values)   / len(vertical_values)
            self.calibrated = True
        else:
            print("Ошибка калибровки. Используются значения по умолчанию.")

    def get_gaze_direction(self) -> Dict[str, Any]:
        if not self.calibrated:
            return {"direction": "not calibrated"}
        horizontal = self.gaze.horizontal_ratio()
        vertical   = self.gaze.vertical_ratio()
        if horizontal is None or vertical is None:
            return {"direction": "blink"}
        self._h_buffer.append(horizontal)
        self._v_buffer.append(vertical)
        smooth_h = sum(self._h_buffer) / len(self._h_buffer)
        smooth_v = sum(self._v_buffer) / len(self._v_buffer)
        h_diff = smooth_h - self.horizontal_center
        v_diff = smooth_v - self.vertical_center
        direction = []
        if not self._h_offcenter and abs(h_diff) > self.threshold_enter:
            self._h_offcenter = True
        elif self._h_offcenter and abs(h_diff) < self.threshold_exit:
            self._h_offcenter = False
        if not self._v_offcenter and abs(v_diff) > self.threshold_enter:
            self._v_offcenter = True
        elif self._v_offcenter and abs(v_diff) < self.threshold_exit:
            self._v_offcenter = False
        if self._h_offcenter:
            direction.append("right" if h_diff < 0 else "left")
        if self._v_offcenter:
            direction.append("up"    if v_diff < 0 else "down")
        if not direction:
            direction.append("center")
        h_diff_real = smooth_h - 0.5
        v_diff_real = smooth_v - 0.5
        angle_from_real_center        = math.degrees(math.atan2(v_diff_real, h_diff_real))
        angle_from_calibrated_center  = math.degrees(math.atan2(v_diff, h_diff))
        return {
            "direction": " ".join(direction),
            "horizontal_ratio":             smooth_h,
            "vertical_ratio":               smooth_v,
            "horizontal_deviation":         h_diff,
            "vertical_deviation":           v_diff,
            "total_deviation":              (h_diff ** 2 + v_diff ** 2) ** 0.5,
            "angle_from_real_center":       angle_from_real_center,
            "angle_from_calibrated_center": angle_from_calibrated_center,
        }


class BehaviorAnalyzer:
    def __init__(self, max_suspicious_actions: int = 1):
        self.suspicious_actions      = 0
        self.max_suspicious_actions  = CONFIG["max_suspicious_actions"] if max_suspicious_actions is None else max_suspicious_actions
        self.gaze_history: List[Dict] = []
        self.analysis_window         = CONFIG["analysis_window"]
        self.window_size             = int(self.analysis_window / CONFIG["sleep_interval"])
        self.min_consecutive_offcenter = int(2.0 / CONFIG["sleep_interval"])
        self.offcenter_threshold     = 0.5
        self.last_direction          = "center"
        self.consecutive_offcenter   = 0
        self.last_offcenter_time     = None
        self._head_consecutive       = 0
        self._head_min_consecutive   = CONFIG.get("head_min_consecutive", 6)
        self._head_history: List[Dict] = []
        self._gaze_triggers: int       = 0
        self._head_triggers: int       = 0

    def analyze_gaze_pattern(self, gaze_data: str, gaze_info: Dict[str, Any] = None) -> None:
        timestamp = time.time()
        history_entry = {"direction": gaze_data, "timestamp": timestamp}
        if gaze_info:
            history_entry.update({
                "horizontal_deviation":         gaze_info.get("horizontal_deviation"),
                "vertical_deviation":           gaze_info.get("vertical_deviation"),
                "total_deviation":              gaze_info.get("total_deviation"),
                "horizontal_ratio":             gaze_info.get("horizontal_ratio"),
                "vertical_ratio":               gaze_info.get("vertical_ratio"),
                "angle_from_real_center":       gaze_info.get("angle_from_real_center"),
                "angle_from_calibrated_center": gaze_info.get("angle_from_calibrated_center"),
            })
        self.gaze_history.append(history_entry)
        if len(self.gaze_history) > self.window_size:
            self.gaze_history = self.gaze_history[-self.window_size:]
        if gaze_data not in ["center", "blink", "not calibrated"]:
            self.consecutive_offcenter += 1
        else:
            self.consecutive_offcenter = max(0, self.consecutive_offcenter - 1)
        self.last_direction = gaze_data
        if self.consecutive_offcenter >= self.min_consecutive_offcenter:
            self.suspicious_actions    += 1
            self._gaze_triggers        += 1
            self.consecutive_offcenter  = 0
        offcenter_count = sum(
            1 for entry in self.gaze_history
            if isinstance(entry, dict) and entry.get("direction") not in ["center", "blink", "not calibrated"]
        )
        if len(self.gaze_history) >= self.window_size:
            offcenter_ratio = offcenter_count / self.window_size
            if offcenter_ratio > self.offcenter_threshold:
                self.suspicious_actions += 1
                self._gaze_triggers     += 1
                self.gaze_history        = self.gaze_history[-self.window_size // 2:]

    def analyze_head_pose(self, head_info: Dict[str, Any]) -> None:
        if head_info is None:
            return
        is_suspicious = head_info.get("is_suspicious", False)
        direction     = head_info.get("direction", "center")
        timestamp     = time.time()
        self._head_history.append({
            "direction":       direction,
            "is_suspicious":   is_suspicious,
            "yaw_deviation":   head_info.get("yaw_deviation"),
            "pitch_deviation": head_info.get("pitch_deviation"),
            "timestamp":       timestamp,
        })
        if len(self._head_history) > self.window_size:
            self._head_history = self._head_history[-self.window_size:]
        if is_suspicious:
            self._head_consecutive += 1
        else:
            self._head_consecutive = max(0, self._head_consecutive - 1)
        if self._head_consecutive >= self._head_min_consecutive:
            self.suspicious_actions += 1
            self._head_triggers     += 1
            self._head_consecutive   = 0

    def detect_cheating(self) -> bool:
        return self.suspicious_actions >= self.max_suspicious_actions

    def generate_report(self) -> Dict[str, Any]:
        formatted_gaze = [
            entry if isinstance(entry, dict)
            else {"timestamp": entry[0], "direction": entry[1]}
            for entry in self.gaze_history
        ]
        if self._gaze_triggers > 0 and self._head_triggers > 0:
            cheating_trigger = "gaze_and_head"
        elif self._gaze_triggers > 0:
            cheating_trigger = "gaze"
        elif self._head_triggers > 0:
            cheating_trigger = "head_pose"
        else:
            cheating_trigger = "unknown"
        report = {
            "suspicious_actions": self.suspicious_actions,
            "gaze_history":       formatted_gaze,
            "head_history":       list(self._head_history),
            "current_status":     "cheating" if self.detect_cheating() else "normal",
            "cheating_trigger":   cheating_trigger,
            "gaze_trigger_count": self._gaze_triggers,
            "head_trigger_count": self._head_triggers,
        }
        self.suspicious_actions = 0
        self._gaze_triggers     = 0
        self._head_triggers     = 0
        return report


class SessionRecorder:
    def __init__(self):
        self.recordings_dir    = Path(CONFIG["recordings_dir"])
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.recording_fps     = CONFIG["recording_fps"]
        self.screen_file_path: Optional[Path]  = None
        self.webcam_file_path: Optional[Path]  = None
        self.combined_file_path: Optional[Path] = None
        self.screen_writer     = None
        self.webcam_writer     = None
        self.screen_size:  Optional[Tuple[int, int]] = None
        self.webcam_size:  Optional[Tuple[int, int]] = None
        self.session_timestamp: Optional[str]  = None
        self.recording_active  = False
        self.session_start_time: Optional[float] = None
        self.session_end_time:   Optional[float] = None
        self._screen_frame_idx   = 0
        self._webcam_frame_idx   = 0
        self._last_screen_time:  Optional[float] = None
        self._last_webcam_time:  Optional[float] = None
        self._screen_accum  = 0.0
        self._webcam_accum  = 0.0

    def _build_session_prefix(self, participant_number: str = None) -> str:
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        p = f"participant_{participant_number}" if participant_number else "participant_unknown"
        return f"{p}_{self.session_timestamp}"

    def start_recording(self, webcam_frame_size: Tuple[int, int], participant_number: str = None) -> None:
        if self.recording_active:
            return
        prefix = self._build_session_prefix(participant_number)
        self.screen_file_path   = self.recordings_dir / f"{prefix}_screen.avi"
        self.webcam_file_path   = self.recordings_dir / f"{prefix}_webcam.avi"
        self.combined_file_path = self.recordings_dir / f"{prefix}_combined.avi"
        with mss() as sct:
            monitor = sct.monitors[1]
            self.screen_size = (monitor["width"], monitor["height"])
        self.webcam_size = webcam_frame_size
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.screen_writer = cv2.VideoWriter(str(self.screen_file_path), fourcc, float(self.recording_fps), self.screen_size)
        self.webcam_writer = cv2.VideoWriter(str(self.webcam_file_path), fourcc, float(self.recording_fps), self.webcam_size)
        if not self.screen_writer.isOpened() or not self.webcam_writer.isOpened():
            print("ОШИБКА: не удалось открыть VideoWriter.")
            if self.screen_writer: self.screen_writer.release()
            if self.webcam_writer: self.webcam_writer.release()
            self.screen_writer = None
            self.webcam_writer = None
            return
        self._screen_frame_idx    = 0
        self._webcam_frame_idx    = 0
        self._last_screen_time    = None
        self._last_webcam_time    = None
        self._screen_accum        = 0.0
        self._webcam_accum        = 0.0
        self.session_start_time   = time.time()
        self.session_end_time     = None
        self.recording_active     = True

    def capture_screen_frame(self):
        if not self.recording_active or self.screen_size is None:
            return None
        try:
            with mss() as sct:
                monitor = sct.monitors[1]
                img = np.array(sct.grab(monitor))
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"Ошибка захвата экрана: {e}")
            return None

    def _write_with_timing(self, writer, frame, last_time_attr, accum_attr, idx_attr) -> None:
        if not self.recording_active or writer is None or frame is None:
            return
        now = time.time()
        last_time = getattr(self, last_time_attr)
        if last_time is None:
            writer.write(frame)
            setattr(self, idx_attr, getattr(self, idx_attr) + 1)
            setattr(self, last_time_attr, now)
            return
        delta = now - last_time
        setattr(self, last_time_attr, now)
        accum = getattr(self, accum_attr) + delta
        frame_interval = 1.0 / max(1, self.recording_fps)
        while accum >= frame_interval:
            writer.write(frame)
            setattr(self, idx_attr, getattr(self, idx_attr) + 1)
            accum -= frame_interval
        setattr(self, accum_attr, accum)

    def write_screen_frame(self, frame) -> None:
        self._write_with_timing(self.screen_writer, frame, "_last_screen_time", "_screen_accum", "_screen_frame_idx")

    def write_webcam_frame(self, frame) -> None:
        self._write_with_timing(self.webcam_writer, frame, "_last_webcam_time", "_webcam_accum", "_webcam_frame_idx")

    def stop_recording(self) -> None:
        if not self.recording_active:
            return
        self.recording_active  = False
        self.session_end_time  = time.time()
        if self.screen_writer:
            self.screen_writer.release()
            self.screen_writer = None
        if self.webcam_writer:
            self.webcam_writer.release()
            self.webcam_writer = None


class DataLogger:
    def __init__(self):
        self.gaze_logs: List[str]        = []
        self.behavior_logs: List[Dict]   = []
        self.logs_dir        = CONFIG["logs_dir"]
        self.gaze_log_file   = Path(self.logs_dir) / CONFIG["gaze_log_file"]
        self.behavior_log_file = Path(self.logs_dir) / CONFIG["behavior_log_file"]

    def log_gaze_data(self, gaze_data: str, gaze_info: Dict[str, Any] = None) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"{timestamp}: {gaze_data}"
        if gaze_info:
            h_dev      = gaze_info.get("horizontal_deviation")
            v_dev      = gaze_info.get("vertical_deviation")
            total_dev  = gaze_info.get("total_deviation")
            angle_cal  = gaze_info.get("angle_from_calibrated_center")
            if h_dev is not None and v_dev is not None:
                log_entry += f" | H_dev: {h_dev:.3f}, V_dev: {v_dev:.3f}, Total_dev: {total_dev:.3f}"
            if angle_cal is not None:
                log_entry += f" | Angle_calibrated: {angle_cal:.2f}°"
        self.gaze_logs.append(log_entry)

    def log_head_data(self, head_info: Dict[str, Any]) -> None:
        if head_info is None:
            return
        timestamp  = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        direction  = head_info.get("direction", "unknown")
        yaw_dev    = head_info.get("yaw_deviation")
        pitch_dev  = head_info.get("pitch_deviation")
        is_susp    = head_info.get("is_suspicious", False)
        log_entry  = f"{timestamp}: [HEAD] {direction}"
        if yaw_dev is not None:
            log_entry += f" | Yaw_dev: {yaw_dev:.1f}°, Pitch_dev: {pitch_dev:.1f}°"
        if is_susp:
            log_entry += " | SUSPICIOUS"
        self.gaze_logs.append(log_entry)

    def log_behavior(self, behavior_data: Dict[str, Any]) -> None:
        timestamp  = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry  = {"timestamp": timestamp, "data": behavior_data}
        self.behavior_logs.append(log_entry)

    def save_logs_to_file(self) -> None:
        if _TEST_MODE:
            self.gaze_logs = []
            self.behavior_logs = []
            return
        if self.gaze_logs:
            with open(self.gaze_log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(self.gaze_logs) + "\n")
            self.gaze_logs = []
        if self.behavior_logs:
            try:
                existing = json.loads(self.behavior_log_file.read_text(encoding="utf-8")) if self.behavior_log_file.exists() else []
            except json.JSONDecodeError:
                existing = []
            existing.extend(self.behavior_logs)
            self.behavior_log_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
            self.behavior_logs = []



class SessionState:
    """Состояние одной прокторинговой сессии."""

    STATUS_CREATED     = "created"
    STATUS_CALIBRATING = "calibrating"
    STATUS_CALIBRATED  = "calibrated"
    STATUS_MONITORING  = "monitoring"
    STATUS_STOPPED     = "stopped"

    def __init__(self, session_id: str, participant_number: str, quiz_id: Optional[str] = None):
        self.session_id          = session_id
        self.participant_number  = participant_number
        self.quiz_id             = quiz_id
        self.created_at          = time.time()
        self.status              = self.STATUS_CREATED

        # Управление фоновым потоком
        self._monitoring_thread: Optional[threading.Thread] = None
        self._lock               = threading.Lock()
        self._stop_event         = threading.Event()

        # Накопленные данные для API
        self.events: List[Dict]  = []
        self.last_verdict: Optional[Dict] = None
        self.calibration_values: Dict = {}

        self.model_referee = _shared_model_referee

        # В тестовом режиме не инициализируем камеру и ML-компоненты
        if _TEST_MODE:
            self.gaze_tracker      = None
            self.head_pose_tracker = None
            self.behavior_analyzer = BehaviorAnalyzer()
            self.logger            = DataLogger()
            self.recorder          = None
            return

        # Компоненты (полная инициализация)
        self.gaze_tracker        = GazeTracker(debug=False)
        self.head_pose_tracker   = HeadPoseTracker()
        self.behavior_analyzer   = BehaviorAnalyzer()
        self.logger              = DataLogger()
        self.recorder            = SessionRecorder()

    def _calibration_thread_fn(self):
        """Фоновая калибровка (без GUI)."""
        calib_time = CONFIG["calibration_time"]
        start      = time.time()
        h_vals, v_vals = [], []
        while time.time() - start < calib_time:
            frame, eyes_ok = self.gaze_tracker.get_frame_with_eyes_status()
            if frame is not None:
                gaze = self.gaze_tracker.gaze
                if gaze.horizontal_ratio() is not None:
                    h_vals.append(gaze.horizontal_ratio())
                if gaze.vertical_ratio() is not None:
                    v_vals.append(gaze.vertical_ratio())
                ret, raw = self.gaze_tracker.camera.read()
                if ret and raw is not None:
                    self.head_pose_tracker.accumulate_calibration(raw)
            time.sleep(0.1)

        if h_vals and v_vals:
            self.gaze_tracker.horizontal_center = sum(h_vals) / len(h_vals)
            self.gaze_tracker.vertical_center   = sum(v_vals) / len(v_vals)
            self.gaze_tracker.calibrated        = True
        self.head_pose_tracker.finalize_calibration()

        self.calibration_values = {
            "horizontal_ratio":  self.gaze_tracker.horizontal_center,
            "vertical_ratio":    self.gaze_tracker.vertical_center,
            "head_neutral_yaw":  self.head_pose_tracker.neutral_yaw,
            "head_neutral_pitch": self.head_pose_tracker.neutral_pitch,
        }

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger.behavior_logs.append({
            "timestamp": ts,
            "data": {
                "participant_number": self.participant_number,
                "message": "Калибровка завершена (API)",
                **self.calibration_values,
            }
        })
        self.logger.save_logs_to_file()

        # Переинициализация GazeTracking после калибровки
        self.gaze_tracker.gaze = GazeTracking()
        if self.gaze_tracker.camera is not None:
            for _ in range(3):
                ret, f = self.gaze_tracker.camera.read()
                if ret and f is not None:
                    self.gaze_tracker.gaze.refresh(f)
                time.sleep(0.1)

        with self._lock:
            self.status = self.STATUS_CALIBRATED

    def _monitoring_thread_fn(self):
        """Основной цикл мониторинга (аналог run_testing_loop)."""
        sleep_interval = CONFIG["sleep_interval"]
        while not self._stop_event.is_set():
            try:
                camera = self.gaze_tracker.camera
                if camera is None:
                    time.sleep(sleep_interval)
                    continue
                ret, frame = camera.read()
                if not ret or frame is None:
                    try:
                        camera.release()
                    except Exception:
                        pass
                    time.sleep(0.1)
                    self.gaze_tracker.camera = cv2.VideoCapture(0)
                    time.sleep(0.2)
                    continue

                self.recorder.write_webcam_frame(frame)
                screen_frame = self.recorder.capture_screen_frame()
                if screen_frame is not None:
                    self.recorder.write_screen_frame(screen_frame)

                self.gaze_tracker.gaze.refresh(frame)
                gaze_info = self.gaze_tracker.get_gaze_direction()
                gaze_data = gaze_info.get("direction")
                head_info = self.head_pose_tracker.get_head_direction(frame)

                if gaze_data and gaze_data != "not calibrated":
                    self.behavior_analyzer.analyze_gaze_pattern(gaze_data, gaze_info)
                    self.logger.log_gaze_data(gaze_data, gaze_info)

                if head_info.get("direction") not in ("not calibrated",):
                    self.behavior_analyzer.analyze_head_pose(head_info)
                    self.logger.log_head_data(head_info)

                if self.behavior_analyzer.detect_cheating():
                    report = self.behavior_analyzer.generate_report()
                    self.logger.log_behavior(report)
                    self.logger.save_logs_to_file()

                    calib = self.calibration_values
                    self.model_referee.log_verdict(report, calib)
                    verdict = self.model_referee.predict(report, calib)

                    with self._lock:
                        self.last_verdict = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "report":    report,
                            "verdict":   verdict,
                        }
                        self.events.append({
                            "type":      "auto_cheating_alert",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "verdict":   verdict,
                        })

                time.sleep(sleep_interval)
            except Exception as e:
                print(f"[Session {self.session_id}] Ошибка мониторинга: {e}")
                import traceback; traceback.print_exc()
                break

    def start_calibration(self):
        with self._lock:
            if self.status != self.STATUS_CREATED:
                raise ValueError(f"Нельзя начать калибровку в статусе '{self.status}'")
            self.status = self.STATUS_CALIBRATING
        if _TEST_MODE:
            self.calibration_values = {
                "horizontal_ratio": 0.5, "vertical_ratio": 0.5,
                "head_neutral_yaw": 0.0, "head_neutral_pitch": 0.0,
            }
            with self._lock:
                self.status = self.STATUS_CALIBRATED
            return
        self.gaze_tracker.initialize_camera_without_calibration()
        t = threading.Thread(target=self._calibration_thread_fn, daemon=True)
        t.start()

    def get_calibration_status(self) -> Dict:
        with self._lock:
            return {
                "status":     self.status,
                "calibrated": self.gaze_tracker.calibrated,
                "head_calibrated": self.head_pose_tracker.calibrated,
                "calibration_values": self.calibration_values,
            }

    def start_monitoring(self):
        with self._lock:
            if self.status != self.STATUS_CALIBRATED:
                raise ValueError(f"Нельзя начать мониторинг в статусе '{self.status}'")
            self.status = self.STATUS_MONITORING
        if _TEST_MODE:
            return
        ret, frame = self.gaze_tracker.camera.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            self.recorder.start_recording((w, h), self.participant_number)
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_thread_fn, daemon=True)
        self._monitoring_thread.start()

    def stop(self):
        if _TEST_MODE:
            with self._lock:
                self.status = self.STATUS_STOPPED
            return
        self._stop_event.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10)
        self.recorder.stop_recording()
        self.gaze_tracker.release_camera()
        self.head_pose_tracker.release()
        self.logger.save_logs_to_file()
        with self._lock:
            self.status = self.STATUS_STOPPED

    def get_current_status(self) -> Dict:
        with self._lock:
            return {
                "session_id":           self.session_id,
                "participant_number":   self.participant_number,
                "quiz_id":              self.quiz_id,
                "status":               self.status,
                "suspicious_actions":   self.behavior_analyzer.suspicious_actions if self.behavior_analyzer else 0,
                "cheating_detected":    self.behavior_analyzer.detect_cheating() if self.behavior_analyzer else False,
                "last_gaze_direction":  self.behavior_analyzer.last_direction if self.behavior_analyzer else None,
                "last_verdict":         self.last_verdict,
                "recording_active":     self.recorder.recording_active if self.recorder else False,
            }

    def add_manual_cheating_mark(self, comment: Optional[str] = None):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        entry = {
            "type":      "manual_cheating_mark",
            "timestamp": ts,
            "comment":   comment,
        }
        self.logger.behavior_logs.append({
            "timestamp": ts,
            "data": {
                "event_type": "manual_cheating_mark",
                "participant_number": self.participant_number,
                "comment": comment,
                "message": "Отмечена попытка списывания (API)",
            }
        })
        self.logger.save_logs_to_file()
        with self._lock:
            self.events.append(entry)

    def get_report(self) -> Dict:
        with self._lock:
            ba = self.behavior_analyzer
            return {
                "session_id":           self.session_id,
                "participant_number":   self.participant_number,
                "quiz_id":              self.quiz_id,
                "status":               self.status,
                "created_at":           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
                "suspicious_actions":   ba.suspicious_actions if ba else 0,
                "cheating_detected":    ba.detect_cheating() if ba else False,
                "events":               list(self.events),
                "last_verdict":         self.last_verdict,
                "calibration_values":   self.calibration_values,
                "gaze_history_len":     len(ba.gaze_history) if ba else 0,
                "head_history_len":     len(ba._head_history) if ba else 0,
            }



_sessions: Dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


_shared_model_referee = ModelReferee()



class CreateSessionRequest(BaseModel):
    participant_number: str = Field(
        ...,
        description="Идентификатор участника (номер студента / логин Moodle)"
    )
    quiz_id: Optional[str] = Field(
        None,
        description="ID теста в Moodle (опционально, для связи с конкретным quiz)"
    )

class CreateSessionResponse(BaseModel):
    session_id: str
    participant_number: str
    quiz_id: Optional[str]
    status: str
    message: str

class SessionStatusResponse(BaseModel):
    session_id: str
    participant_number: str
    quiz_id: Optional[str]
    status: str
    suspicious_actions: int
    cheating_detected: bool
    last_gaze_direction: Optional[str]
    last_verdict: Optional[Dict]
    recording_active: bool

class CalibrationStatusResponse(BaseModel):
    status: str
    calibrated: bool
    head_calibrated: bool
    calibration_values: Dict

class ManualCheatMarkRequest(BaseModel):
    comment: Optional[str] = Field(
        None,
        description="Необязательный комментарий проктора"
    )

class ConfigUpdateRequest(BaseModel):
    max_suspicious_actions:   Optional[int]   = None
    calibration_threshold:    Optional[float] = None
    head_yaw_threshold:       Optional[float] = None
    head_pitch_threshold:     Optional[float] = None
    head_min_consecutive:     Optional[int]   = None
    calibration_time:         Optional[int]   = None
    analysis_window:          Optional[int]   = None
    sleep_interval:           Optional[float] = None
    recording_fps:            Optional[int]   = None
    debug:                    Optional[bool]  = None



app = FastAPI(
    title="Proctoring Eye-Tracking API",
    description=(
        "REST API для прокторинга с отслеживанием взгляда и положения головы. "
        "Жизненный цикл сессии: create, start, status, "
        "monitoring, status, stop, report."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _get_session(session_id: str) -> SessionState:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Сессия '{session_id}' не найдена")
    return session


@app.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Сессии"],
    summary="Создать новую прокторинговую сессию",
    description=(
        "**POST** — создаёт ресурс новой сессии. "
        "Moodle передаёт participant_number (userId) и quiz_id перед началом теста. "
        "Возвращает session_id, который используется во всех последующих запросах."
    ),
)
def create_session(body: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    session    = SessionState(
        session_id         = session_id,
        participant_number = body.participant_number,
        quiz_id            = body.quiz_id,
    )
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    session.logger.behavior_logs.append({
        "timestamp": ts,
        "data": {
            "participant_number": body.participant_number,
            "quiz_id":           body.quiz_id,
            "message":           "Сессия создана (API)",
        }
    })
    session.logger.save_logs_to_file()
    with _sessions_lock:
        _sessions[session_id] = session
    return CreateSessionResponse(
        session_id         = session_id,
        participant_number = body.participant_number,
        quiz_id            = body.quiz_id,
        status             = session.status,
        message            = "Сессия создана. Запустите калибровку: POST /sessions/{session_id}/calibrate/start",
    )


@app.get(
    "/sessions",
    tags=["Сессии"],
    summary="Список всех сессий",
    description=(
        "**GET** — возвращает список существующих сессий (id, участник, статус). "
        "Используется администратором или Moodle-модулем для мониторинга."
    ),
)
def list_sessions():
    with _sessions_lock:
        return [
            {
                "session_id":          s.session_id,
                "participant_number":  s.participant_number,
                "quiz_id":             s.quiz_id,
                "status":              s.status,
                "created_at":          time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s.created_at)),
            }
            for s in _sessions.values()
        ]


@app.get(
    "/sessions/{session_id}",
    response_model=SessionStatusResponse,
    tags=["Сессии"],
    summary="Текущий статус сессии",
    description=(
        "**GET** — опрашивает состояние сессии (Moodle периодически polling-ит этот эндпоинт "
        "чтобы проверить, не зафиксировано ли нарушение). "
        "Возвращает флаг cheating_detected и последний ML-вердикт."
    ),
)
def get_session_status(session_id: str):
    session = _get_session(session_id)
    return SessionStatusResponse(**session.get_current_status())


@app.delete(
    "/sessions/{session_id}",
    tags=["Сессии"],
    summary="Принудительно удалить сессию",
    description=(
        "**DELETE** — останавливает запись и удаляет сессию из памяти. "
        "Использовать с осторожностью: незаписанные буферы будут потеряны. "
        "Нормальное завершение — через POST /sessions/{session_id}/stop."
    ),
)
def delete_session(session_id: str):
    session = _get_session(session_id)
    session.stop()
    with _sessions_lock:
        del _sessions[session_id]
    return {"message": f"Сессия '{session_id}' удалена"}



@app.post(
    "/sessions/{session_id}/calibrate/start",
    tags=["Калибровка"],
    summary="Запустить калибровку взгляда и головы",
    description=(
        "**POST** — инициирует фоновый поток калибровки. "
        "Участник должен смотреть прямо на экран. "
        "Длительность задаётся параметром calibration_time в config. "
        "После запуска опрашивайте GET /sessions/{session_id}/calibrate/status "
        "до получения status='calibrated'."
    ),
)
def start_calibration(session_id: str):
    session = _get_session(session_id)
    try:
        session.start_calibration()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "message": "Калибровка начата. Опрашивайте GET /sessions/{session_id}/calibrate/status",
        "calibration_time_sec": CONFIG["calibration_time"],
    }


@app.get(
    "/sessions/{session_id}/calibrate/status",
    response_model=CalibrationStatusResponse,
    tags=["Калибровка"],
    summary="Статус калибровки (polling)",
    description=(
        "**GET** — возвращает текущий статус калибровки. "
        "Moodle-фронтенд периодически опрашивает этот эндпоинт. "
        "Когда status == 'calibrated', можно запускать мониторинг."
    ),
)
def get_calibration_status(session_id: str):
    session = _get_session(session_id)
    return CalibrationStatusResponse(**session.get_calibration_status())



@app.post(
    "/sessions/{session_id}/monitoring/start",
    tags=["Мониторинг"],
    summary="Запустить мониторинг (начать тест)",
    description=(
        "**POST** — запускает фоновый поток мониторинга взгляда и головы, "
        "начинает запись видео с веб-камеры и экрана. "
        "Должен вызываться ПОСЛЕ успешной калибровки (status == 'calibrated'). "
        "Соответствует нажатию кнопки 'Начать тест' в интерфейсе Moodle."
    ),
)
def start_monitoring(session_id: str):
    session = _get_session(session_id)
    try:
        session.start_monitoring()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "Мониторинг запущен. Запись начата."}


@app.post(
    "/sessions/{session_id}/stop",
    tags=["Мониторинг"],
    summary="Завершить сессию (конец теста)",
    description=(
        "**POST** — штатное завершение: останавливает мониторинг, "
        "сохраняет видео и логи. "
        "Вызывается Moodle при сабмите формы теста или истечении таймера."
    ),
)
def stop_session(session_id: str):
    session = _get_session(session_id)
    if session.status == SessionState.STATUS_STOPPED:
        raise HTTPException(status_code=400, detail="Сессия уже завершена")
    session.stop()
    return {"message": "Сессия завершена. Логи сохранены."}



@app.post(
    "/sessions/{session_id}/cheating-mark",
    tags=["Нарушения"],
    summary="Вручную отметить попытку списывания",
    description=(
        "**POST** — проктор или автоматизированная система (Moodle) "
        "фиксирует нарушение вручную. "
        "Это изменяет состояние ресурса 'сессия' (добавляет событие), "
        "поэтому метод POST, а не GET."
    ),
)
def mark_cheating(session_id: str, body: ManualCheatMarkRequest):
    session = _get_session(session_id)
    if session.status == SessionState.STATUS_STOPPED:
        raise HTTPException(status_code=400, detail="Нельзя добавлять метки в завершённую сессию")
    session.add_manual_cheating_mark(comment=body.comment)
    return {"message": "Метка списывания добавлена", "comment": body.comment}


@app.get(
    "/sessions/{session_id}/events",
    tags=["Нарушения"],
    summary="Список событий сессии (нарушений)",
    description=(
        "**GET** — возвращает все зафиксированные события: "
        "автоматические алерты от ML-модели и ручные метки проктора. "
        "Не изменяет состояние — только чтение, поэтому GET."
    ),
)
def get_events(session_id: str):
    session = _get_session(session_id)
    with session._lock:
        return {"session_id": session_id, "events": list(session.events)}



@app.get(
    "/sessions/{session_id}/report",
    tags=["Отчёты"],
    summary="Итоговый отчёт по сессии",
    description=(
        "**GET** — возвращает полный отчёт: статистику взгляда/головы, "
        "ML-вердикт, список событий. "
        "Данные только читаются, состояние не меняется — GET."
    ),
)
def get_report(session_id: str):
    session = _get_session(session_id)
    return session.get_report()


@app.get(
    "/sessions/{session_id}/verdict",
    tags=["Отчёты"],
    summary="Последний ML-вердикт",
    description=(
        "**GET** — возвращает последний вердикт ML-модели (cheating / normal / model_unavailable) "
        "с вероятностью. Читающий запрос — GET."
    ),
)
def get_verdict(session_id: str):
    session = _get_session(session_id)
    with session._lock:
        verdict = session.last_verdict
    if verdict is None:
        return {"session_id": session_id, "verdict": None, "message": "Вердиктов ещё нет"}
    return {"session_id": session_id, **verdict}


@app.get(
    "/sessions/{session_id}/logs",
    tags=["Отчёты"],
    summary="Логи активности сессии",
    description=(
        "**GET** — возвращает содержимое behavior_log.json для данного участника. "
        "Только чтение — GET."
    ),
)
def get_logs(session_id: str):
    session = _get_session(session_id)
    log_file = Path(CONFIG["logs_dir"]) / CONFIG["behavior_log_file"]
    if not log_file.exists():
        return {"session_id": session_id, "logs": []}
    try:
        all_logs = json.loads(log_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        all_logs = []
    participant_logs = [
        entry for entry in all_logs
        if entry.get("data", {}).get("participant_number") == session.participant_number
    ]
    return {"session_id": session_id, "logs": participant_logs}



@app.get(
    "/sessions/{session_id}/recordings",
    tags=["Записи"],
    summary="Список видеофайлов сессии",
    description=(
        "**GET** — возвращает пути к файлам записи (экран и камера). "
        "Только чтение — GET."
    ),
)
def get_recordings(session_id: str):
    session = _get_session(session_id)
    files = {}
    if session.recorder.screen_file_path and session.recorder.screen_file_path.exists():
        files["screen"] = str(session.recorder.screen_file_path)
    if session.recorder.webcam_file_path and session.recorder.webcam_file_path.exists():
        files["webcam"] = str(session.recorder.webcam_file_path)
    return {"session_id": session_id, "recordings": files}


@app.get(
    "/sessions/{session_id}/recordings/{file_type}",
    tags=["Записи"],
    summary="Скачать видеофайл записи",
    description=(
        "**GET** — скачивает AVI-файл записи. "
        "file_type: 'screen' или 'webcam'. "
        "Только чтение — GET."
    ),
)
def download_recording(session_id: str, file_type: str):
    session = _get_session(session_id)
    if session.status != SessionState.STATUS_STOPPED:
        raise HTTPException(status_code=400, detail="Сессия должна быть завершена перед скачиванием")
    if file_type == "screen":
        path = session.recorder.screen_file_path
    elif file_type == "webcam":
        path = session.recorder.webcam_file_path
    else:
        raise HTTPException(status_code=400, detail="file_type должен быть 'screen' или 'webcam'")
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Файл записи не найден")
    return FileResponse(str(path), media_type="video/x-msvideo", filename=path.name)



@app.get(
    "/config",
    tags=["Конфигурация"],
    summary="Получить текущую конфигурацию",
    description=(
        "**GET** — возвращает активную конфигурацию сервиса. "
        "Только чтение — GET."
    ),
)
def get_config():
    return {"config": CONFIG}


@app.patch(
    "/config",
    tags=["Конфигурация"],
    summary="Обновить параметры конфигурации",
    description=(
        "**PATCH** — частично обновляет конфигурацию (только переданные поля). "
        "PATCH, а не PUT, потому что изменяются отдельные параметры, "
        "а не весь объект конфигурации целиком. "
        "Изменения применяются к новым сессиям; активные сессии не затрагиваются."
    ),
)
def update_config(body: ConfigUpdateRequest):
    updated_fields = {}
    for field, value in body.model_dump(exclude_none=True).items():
        CONFIG[field] = value
        updated_fields[field] = value
    return {"message": "Конфигурация обновлена", "updated": updated_fields}



@app.get(
    "/health",
    tags=["Служебные"],
    summary="Проверка работоспособности сервиса",
    description="**GET** — возвращает статус 'ok'. Используется load-balancer'ом или Moodle для проверки доступности.",
)
def health_check():
    with _sessions_lock:
        n = len(_sessions)
    return {
        "status": "ok",
        "active_sessions": n,
        "test_mode": _TEST_MODE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("proctoring_api:app", host="0.0.0.0", port=8085, reload=False)
