import cv2
from gaze_tracking import GazeTracking
import time
import json
import math
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import winsound
from collections import deque
from mss import mss
import numpy as np
import mediapipe as mp
import joblib

def load_config(config_path: str = "config.json") -> Dict:
    default_config = {
        "debug": True,
        "debug_window_size": [800, 600],
        "calibration_threshold": 0.10,
        "max_suspicious_actions": 1,
        "logs_dir": "logs",
        "gaze_log_file": "gaze_log.txt",
        "behavior_log_file": "behavior_log.json",
        # Модель
        "model_log_file": "behavior_log_with_model.json",
        "model_pkl_file": "best_model.pkl",
        "calibration_time": 10,
        "analysis_window": 5,
        "sleep_interval": 0.33,
        "recordings_dir": "recordings",
        "recording_fps": 10,
        # Голова
        "head_yaw_threshold": 15.0,
        "head_pitch_threshold": 15.0,
        "head_min_consecutive": 6,
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
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
        [0.0,    0.0,    0.0  ],
        [-225.0, 170.0, -135.0],
        [225.0,  170.0, -135.0],
        [-150.0,-150.0, -125.0],
        [150.0, -150.0, -125.0],
        [0.0,   -330.0,  -65.0],
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
            print(f"HeadPoseTracker: нейтраль зафиксирована. Yaw={self.neutral_yaw:.1f}°, Pitch={self.neutral_pitch:.1f}°")
            return True
        print("HeadPoseTracker: калибровка не удалась, данных недостаточно.")
        return False

    def get_head_direction(self, frame: np.ndarray) -> Dict[str, any]:
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
            parts.append("right" if yaw_dev > 0 else "left")
        if self._v_offcenter:
            parts.append("down" if pitch_dev > 0 else "up")
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
    """Принимает данные наблюдения,извлекает фичи, запрашивает предсказание и записывает результат."""

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
        self._model         = None
        self._preprocessor  = None
        self._feat_names    = None
        self._model_loaded  = False
        self._load_model()

    # Загрузка модели
    def _load_model(self) -> None:
        """Загружает best_model.pkl, preprocessor.pkl и feature_names.pkl."""
        base = self.pkl_path.parent

        model_path = self.pkl_path
        prep_path  = base / "preprocessor.pkl"
        feat_path  = base / "feature_names.pkl"

        missing = [p for p in (model_path, prep_path, feat_path) if not p.exists()]
        if missing:
            print(f"[ModelReferee] Файлы не найдены: {[str(p) for p in missing]}")
            print("[ModelReferee] Модель не загружена. Верификация будет отключена.")
            return

        try:
            self._model        = joblib.load(model_path)
            self._preprocessor = joblib.load(prep_path)
            self._feat_names   = joblib.load(feat_path)
            self._model_loaded = True
            print(f"[ModelReferee] Модель загружена: {model_path.name}")
        except Exception as e:
            print(f"[ModelReferee] Ошибка загрузки модели: {e}")

    # Извлечение признаков
    def _extract_features(self, report: Dict, calib: Dict) -> "pd.DataFrame":
        """Преобразует report из BehaviorAnalyzerв DataFrame с признаками, ожидаемыми моделью."""
        import pandas as pd

        dmap = self.DIRECTION_MAP
        tmap = self.TRIGGER_MAP

        gh = report.get("gaze_history", [])
        hh = report.get("head_history", [])

        # взгял
        g_dirs  = [dmap.get(f.get("direction", ""), 0)
                   for f in gh if isinstance(f, dict)]
        g_tdevs = [f["total_deviation"] for f in gh
                   if isinstance(f, dict) and f.get("total_deviation") is not None]
        g_acals = [f["angle_from_calibrated_center"] for f in gh
                   if isinstance(f, dict) and f.get("angle_from_calibrated_center") is not None]

        # голова
        h_dirs  = [dmap.get(f.get("direction", ""), 0)
                   for f in hh if isinstance(f, dict)]
        h_yaws  = [abs(f["yaw_deviation"]) for f in hh
                   if isinstance(f, dict) and f.get("yaw_deviation") is not None]
        h_pitch = [abs(f["pitch_deviation"]) for f in hh
                   if isinstance(f, dict) and f.get("pitch_deviation") is not None]
        h_susp  = [int(f.get("is_suspicious", False)) for f in hh
                   if isinstance(f, dict)]

        n_g = max(len(g_dirs), 1)
        n_h = max(len(h_dirs), 1)

        def safe_mean(lst): return float(np.nanmean(lst)) if lst else np.nan
        def safe_std(lst):  return float(np.nanstd(lst))  if lst else np.nan
        def safe_max(lst):  return float(np.nanmax(lst))  if lst else np.nan
        def safe_min(lst):  return float(np.nanmin(lst))  if lst else np.nan

        row = {
            "suspicious_actions":        report.get("suspicious_actions"),
            "cheating_trigger":          tmap.get(report.get("cheating_trigger"), np.nan),
            "gaze_trigger_count":        report.get("gaze_trigger_count"),
            "head_trigger_count":        report.get("head_trigger_count"),
            "calib_horizontal_ratio":    calib.get("horizontal_ratio"),
            "calib_vertical_ratio":      calib.get("vertical_ratio"),
            "calib_head_neutral_yaw":    calib.get("head_neutral_yaw"),
            "calib_head_neutral_pitch":  calib.get("head_neutral_pitch"),
            # взгляд
            "gaze_num_mean":             safe_mean(g_tdevs),
            "gaze_num_std":              safe_std(g_tdevs),
            "gaze_num_max":              safe_max(g_tdevs),
            "gaze_num_min":              safe_min(g_tdevs),
            "gaze_total_dev_mean":       safe_mean(g_tdevs),
            "gaze_total_dev_std":        safe_std(g_tdevs),
            "gaze_total_dev_max":        safe_max(g_tdevs),
            "gaze_calib_angle_mean":     safe_mean(g_acals),
            "gaze_calib_angle_std":      safe_std(g_acals),
            "gaze_dir_non_center_ratio": sum(1 for d in g_dirs if d != 0) / n_g,
            "gaze_dir_up_ratio":         sum(1 for d in g_dirs if d == 1) / n_g,
            "gaze_dir_down_ratio":       sum(1 for d in g_dirs if d == 2) / n_g,
            "gaze_dir_blink_ratio":      sum(1 for d in g_dirs if d == 9) / n_g,
            "gaze_n_frames":             float(len(g_dirs)),
            # голова
            "head_num_mean":             safe_mean(h_yaws + h_pitch),
            "head_num_std":              safe_std(h_yaws + h_pitch),
            "head_yaw_abs_mean":         safe_mean(h_yaws),
            "head_yaw_abs_max":          safe_max(h_yaws),
            "head_pitch_abs_mean":       safe_mean(h_pitch),
            "head_pitch_abs_max":        safe_max(h_pitch),
            "head_dir_non_center_ratio": sum(1 for d in h_dirs if d != 0) / n_h,
            "head_dir_up_ratio":         sum(1 for d in h_dirs if d == 1) / n_h,
            "head_dir_down_ratio":       sum(1 for d in h_dirs if d == 2) / n_h,
            "head_n_frames":             float(len(h_dirs)),
            "head_is_suspicious_ratio":  float(np.mean(h_susp)) if h_susp else np.nan,
            "head_is_suspicious_any":    float(int(any(h_susp))) if h_susp else 0.0,
        }

        df_row = pd.DataFrame([row])
        for col in self._feat_names:
            if col not in df_row.columns:
                df_row[col] = np.nan
        return df_row[self._feat_names]

    # Предсказание
    def predict(self, report: Dict, calib: Dict) -> Dict[str, any]:
        """Возвращает словарь с результатом предсказания"""
        if not self._model_loaded:
            return {"model_verdict": "model_unavailable", "model_probability": None}

        try:
            import pandas as pd
            X_row  = self._extract_features(report, calib)
            X_proc = self._preprocessor.transform(X_row)
            pred   = self._model.predict(X_proc)[0]
            proba  = None
            if hasattr(self._model, "predict_proba"):
                proba = round(float(self._model.predict_proba(X_proc)[0][1]), 4)
            if pred < 0.45:
                verdict = "normal"
            elif pred <= 0.55:
                verdict = "may be cheating"
            else:
                verdict = "cheating"
            return {"model_verdict": verdict, "model_probability": proba}
        except Exception as e:
            print(f"[ModelReferee] Ошибка предсказания: {e}")
            return {"model_verdict": "model_unavailable", "model_probability": None}

    # Запись в лог
    def log_verdict(self, report: Dict, calib: Dict) -> None:
        """Формирует запись в behavior_log_with_model.json."""
        prediction = self.predict(report, calib)

        data_with_verdict = dict(report)
        data_with_verdict["model_verdict"]     = prediction["model_verdict"]
        data_with_verdict["model_probability"] = prediction["model_probability"]

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        entry = {"timestamp": timestamp, "data": data_with_verdict}

        try:
            if self.log_file.exists():
                with open(self.log_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []
        except (json.JSONDecodeError, IOError):
            existing = []

        existing.append(entry)

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        verdict_str = prediction["model_verdict"].upper()
        prob_str    = f" (p={prediction['model_probability']:.4f})" if prediction["model_probability"] is not None else ""
        print(f"[ModelReferee] Вердикт: {verdict_str}{prob_str}")


class GazeTracker:
    def __init__(self, debug: bool = True, calibration_threshold: float = 0.10):
        self.gaze = GazeTracking()
        self.camera = None
        self.debug = CONFIG["debug"] if debug is None else debug
        self.debug_window_size = tuple(CONFIG["debug_window_size"])
        self.calibration_threshold = CONFIG["calibration_threshold"] if calibration_threshold is None else calibration_threshold
        self.horizontal_center = 0.5
        self.vertical_center = 0.5
        self.calibrated = False
        self.calibration_time = CONFIG["calibration_time"]
        self._h_buffer = deque(maxlen=7)
        self._v_buffer = deque(maxlen=7)
        self.threshold_enter = 0.13
        self.threshold_exit = 0.07
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
            except:
                pass
            time.sleep(0.1)
            self.camera = cv2.VideoCapture(0)
            time.sleep(0.2)
            ret, frame = self.camera.read()
            if not ret or frame is None:
                return None
        self.gaze.refresh(frame)
        frame = self.gaze.annotated_frame()
        gaze_info = self.get_gaze_direction()
        direction = gaze_info["direction"]
        should_show_debug = show_debug if show_debug is not None else self.debug
        if should_show_debug:
            debug_frame = frame.copy()
            debug_frame = cv2.resize(debug_frame, self.debug_window_size)
            cv2.putText(debug_frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"H: {gaze_info.get('horizontal_ratio', 0):.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"V: {gaze_info.get('vertical_ratio', 0):.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Debug: Eye Tracking", debug_frame)
            cv2.waitKey(1)
        return direction if direction != "not calibrated" else None

    def get_frame_with_eyes_status(self) -> Tuple[Optional[any], bool]:
        if self.camera is None:
            return None, False
        ret, frame = self.camera.read()
        if not ret or frame is None:
            try:
                self.camera.release()
            except:
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
        if self.debug:
            cv2.destroyAllWindows()

    def calibrate(self, auto_start: bool = False) -> None:
        if not auto_start:
            print("Калибровка: направьте глаза в центр экрана и нажмите 'c'")
            while True:
                _, frame = self.camera.read()
                self.gaze.refresh(frame)
                frame = self.gaze.annotated_frame()
                debug_frame = frame.copy()
                debug_frame = cv2.resize(debug_frame, self.debug_window_size)
                cv2.putText(debug_frame, "Calibration: Look straight and press 'c'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Calibration", debug_frame)
                key = cv2.waitKey(1)
                if key == ord('c') or key == ord('с'):
                    break
        else:
            print("Калибровка начинается... Направьте глаза в центр экрана")
            time.sleep(2)
        print("Калибровка... Не двигайте глазами 10 секунд")
        start_time = time.time()
        horizontal_values = []
        vertical_values = []
        calibration_window = None
        if self.debug and not auto_start:
            calibration_window = "Calibration"
        elif auto_start:
            cv2.namedWindow("Calibration Progress", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Calibration Progress", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            calibration_window = "Calibration Progress"
        while time.time() - start_time < self.calibration_time:
            _, frame = self.camera.read()
            self.gaze.refresh(frame)
            frame = self.gaze.annotated_frame()
            if self.gaze.horizontal_ratio() is not None:
                horizontal_values.append(self.gaze.horizontal_ratio())
            if self.gaze.vertical_ratio() is not None:
                vertical_values.append(self.gaze.vertical_ratio())
            if calibration_window:
                progress_frame = frame.copy()
                elapsed = time.time() - start_time
                remaining = self.calibration_time - elapsed
                progress_text = f"Калибровка... Осталось {int(remaining)} секунд. Не двигайте глазами."
                h, w = progress_frame.shape[:2]
                text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h // 2
                cv2.putText(progress_frame, progress_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.imshow(calibration_window, progress_frame)
                cv2.waitKey(1)
            time.sleep(0.1)
        if horizontal_values and vertical_values:
            self.horizontal_center = sum(horizontal_values) / len(horizontal_values)
            self.vertical_center = sum(vertical_values) / len(vertical_values)
            self.calibrated = True
            print(f"Калибровка завершена. Центр: H={self.horizontal_center:.2f}, V={self.vertical_center:.2f}")
        else:
            print("Ошибка калибровки. Используются значения по умолчанию.")
        if calibration_window:
            cv2.destroyAllWindows()

    def get_gaze_direction(self) -> Dict[str, any]:
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
            direction.append("up" if v_diff < 0 else "down")
        if not direction:
            direction.append("center")
        h_diff_real = smooth_h - 0.5
        v_diff_real = smooth_v - 0.5
        angle_from_real_center        = math.degrees(math.atan2(v_diff_real, h_diff_real))
        angle_from_calibrated_center  = math.degrees(math.atan2(v_diff, h_diff))
        return {
            "direction": " ".join(direction),
            "horizontal_ratio": smooth_h,
            "vertical_ratio": smooth_v,
            "horizontal_deviation": h_diff,
            "vertical_deviation": v_diff,
            "total_deviation": (h_diff ** 2 + v_diff ** 2) ** 0.5,
            "angle_from_real_center": angle_from_real_center,
            "angle_from_calibrated_center": angle_from_calibrated_center,
        }


class BehaviorAnalyzer:
    def __init__(self, max_suspicious_actions: int = 1):
        self.suspicious_actions = 0
        self.max_suspicious_actions = CONFIG["max_suspicious_actions"] if max_suspicious_actions is None else max_suspicious_actions
        self.gaze_history = []
        self.analysis_window = CONFIG["analysis_window"]
        self.window_size = int(self.analysis_window / CONFIG["sleep_interval"])
        self.min_consecutive_offcenter = int(2.0 / CONFIG["sleep_interval"])
        self.offcenter_threshold = 0.5
        self.last_direction = "center"
        self.consecutive_offcenter = 0
        self.last_offcenter_time = None
        self._head_consecutive = 0
        self._head_min_consecutive = CONFIG.get("head_min_consecutive", 6)
        self._head_history: List[Dict] = []
        self._gaze_triggers: int = 0
        self._head_triggers: int = 0

    def analyze_gaze_pattern(self, gaze_data: str, gaze_info: Dict[str, any] = None) -> None:
        timestamp = time.time()
        history_entry = {"direction": gaze_data, "timestamp": timestamp}
        if gaze_info:
            history_entry.update({
                "horizontal_deviation": gaze_info.get("horizontal_deviation"),
                "vertical_deviation": gaze_info.get("vertical_deviation"),
                "total_deviation": gaze_info.get("total_deviation"),
                "horizontal_ratio": gaze_info.get("horizontal_ratio"),
                "vertical_ratio": gaze_info.get("vertical_ratio"),
                "angle_from_real_center": gaze_info.get("angle_from_real_center"),
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
            self.suspicious_actions += 1
            self._gaze_triggers += 1
            self.consecutive_offcenter = 0
        offcenter_count = sum(
            1 for entry in self.gaze_history
            if isinstance(entry, dict) and entry.get("direction") not in ["center", "blink", "not calibrated"]
            or (isinstance(entry, tuple) and entry[1] not in ["center", "blink", "not calibrated"])
        )
        if len(self.gaze_history) >= self.window_size:
            offcenter_ratio = offcenter_count / self.window_size
            if offcenter_ratio > self.offcenter_threshold:
                self.suspicious_actions += 1
                self._gaze_triggers += 1
                self.gaze_history = self.gaze_history[-self.window_size // 2:]

    def analyze_head_pose(self, head_info: Dict[str, any]) -> None:
        if head_info is None:
            return
        is_suspicious = head_info.get("is_suspicious", False)
        direction     = head_info.get("direction", "center")
        timestamp     = time.time()
        self._head_history.append({
            "direction": direction,
            "is_suspicious": is_suspicious,
            "yaw_deviation": head_info.get("yaw_deviation"),
            "pitch_deviation": head_info.get("pitch_deviation"),
            "timestamp": timestamp,
        })
        if len(self._head_history) > self.window_size:
            self._head_history = self._head_history[-self.window_size:]
        if is_suspicious:
            self._head_consecutive += 1
        else:
            self._head_consecutive = max(0, self._head_consecutive - 1)
        if self._head_consecutive >= self._head_min_consecutive:
            self.suspicious_actions += 1
            self._head_triggers += 1
            self._head_consecutive = 0
            print(f"[HeadPose] Подозрительное поведение: голова отклонена ({direction})")

    def detect_cheating(self) -> bool:
        return self.suspicious_actions >= self.max_suspicious_actions

    def generate_report(self) -> Dict[str, any]:
        formatted_gaze = []
        for entry in self.gaze_history:
            if isinstance(entry, dict):
                formatted_gaze.append(entry)
            else:
                timestamp, direction = entry
                formatted_gaze.append({"timestamp": timestamp, "direction": direction})
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
            "gaze_history": formatted_gaze,
            "head_history": list(self._head_history),
            "current_status": "cheating" if self.detect_cheating() else "normal",
            "cheating_trigger": cheating_trigger,
            "gaze_trigger_count": self._gaze_triggers,
            "head_trigger_count": self._head_triggers,
        }
        self.suspicious_actions = 0
        self._gaze_triggers = 0
        self._head_triggers = 0
        return report


class UIInterface:
    @staticmethod
    def display_gaze_data(gaze_data: str) -> None:
        pass

    @staticmethod
    def show_alert() -> None:
        pass

    @staticmethod
    def display_report(report: Dict[str, any]) -> None:
        pass



class GUIInterface:
    """GUI интерфейс с тремя окнами для тестирования."""

    def __init__(self, main_app):
        self.main_app = main_app
        self.participant_number = None
        self.window1 = None
        self.window2_active = False
        self.window3 = None
        self.calibration_button_active = False
        self.testing_active = False
        self.calibration_completed = False
        self.calibrating = False
        self._testing_thread = None
        self.root = None
        self.button_window = None

    def show_window1(self):
        self.root = tk.Tk()
        self.window1 = self.root
        self.window1.title("Начало тестирования")
        self.window1.geometry("400x200")
        self.window1.resizable(False, False)
        self.window1.update_idletasks()
        x = (self.window1.winfo_screenwidth()  // 2) - 200
        y = (self.window1.winfo_screenheight() // 2) - 100
        self.window1.geometry(f"400x200+{x}+{y}")
        label = tk.Label(self.window1, text="Введите номер участника:", font=("Arial", 12))
        label.pack(pady=20)
        entry = tk.Entry(self.window1, font=("Arial", 12), width=30)
        entry.pack(pady=10)
        entry.focus()

        def start_testing():
            participant = entry.get().strip()
            if not participant:
                messagebox.showwarning("Предупреждение", "Пожалуйста, введите номер участника!")
                return
            self.participant_number = participant
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.main_app.logger.gaze_logs.append(f"{timestamp}: Номер участника: {participant}")
            self.main_app.logger.behavior_logs.append({
                "timestamp": timestamp,
                "data": {
                    "participant_number": participant,
                    "message": "Начало сеанса, номер участника записан",
                }
            })
            self.main_app.logger.save_logs_to_file()
            self.window1.withdraw()
            self.show_window2()

        button = tk.Button(self.window1, text="Начать тестирование",
                           command=start_testing, font=("Arial", 11),
                           bg="#4CAF50", fg="white", padx=20, pady=10)
        button.pack(pady=20)
        entry.bind("<Return>", lambda e: start_testing())
        self.window1.mainloop()

    def show_window2(self):
        self.window2_active = True
        self.main_app.gaze_tracker.initialize_camera_without_calibration()
        cv2.namedWindow("Calibration Window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.button_window = tk.Toplevel(self.root)
        self.button_window.overrideredirect(True)
        self.button_window.attributes('-topmost', True)
        self.button_window.configure(bg='black')
        screen_width  = self.button_window.winfo_screenwidth()
        screen_height = self.button_window.winfo_screenheight()
        self.calibration_overlay = tk.Toplevel(self.root)
        self.calibration_overlay.overrideredirect(True)
        self.calibration_overlay.attributes('-topmost', True)
        self.calibration_overlay.configure(bg='black')
        try:
            self.calibration_overlay.attributes('-alpha', 0.4)
        except:
            pass
        self.calibration_overlay.geometry(f"{screen_width}x{screen_height}+0+0")
        self.calibration_canvas = tk.Canvas(
            self.calibration_overlay, width=screen_width, height=screen_height,
            bg='black', highlightthickness=0
        )
        self.calibration_canvas.pack(fill=tk.BOTH, expand=True)
        dot_radius = 25
        center_x = screen_width  // 2
        center_y = screen_height // 2
        self.calibration_canvas.create_oval(
            center_x - dot_radius, center_y - dot_radius,
            center_x + dot_radius, center_y + dot_radius,
            fill='#FF0000', outline='white', width=5, tags='dot'
        )
        instruction_frame = tk.Frame(self.calibration_canvas, bg='black')
        instruction_label = tk.Label(
            instruction_frame,
            text="Во время калибровки смотрите на красную точку",
            font=("Arial", 28, "bold"), bg='black', fg='white'
        )
        instruction_label.pack()
        self.calibration_canvas.create_window(screen_width // 2, 120, window=instruction_frame, anchor='center')
        progress_frame = tk.Frame(self.calibration_canvas, bg='black')
        self.progress_label = tk.Label(progress_frame, text="", font=("Arial", 36, "bold"), bg='black', fg='#00FF00')
        self.progress_label.pack()
        progress_y = screen_height // 2 + 150
        self.calibration_canvas.create_window(screen_width // 2, progress_y, window=progress_frame, anchor='center')
        self.calibration_overlay.update()
        self.calibration_button = tk.Button(
            self.button_window, text="Начать калибровку",
            command=self.start_calibration, font=("Arial", 14),
            bg="#2196F3", fg="white", padx=30, pady=15, state="disabled"
        )
        self.calibration_button.pack(pady=20)
        self.button_window.geometry(f"300x100+{screen_width // 2 - 150}+{screen_height - 120}")

        def update_frame():
            if not self.window2_active:
                try:
                    self.button_window.destroy()
                except:
                    pass
                return
            try:
                frame, eyes_detected = self.main_app.gaze_tracker.get_frame_with_eyes_status()
                if frame is not None:
                    cv2.imshow("Calibration Window", frame)
                    cv2.waitKey(1)
                    if eyes_detected and not self.calibration_completed:
                        try:
                            self.calibration_button.config(state="normal")
                        except:
                            pass
                    elif not eyes_detected:
                        try:
                            self.calibration_button.config(state="disabled")
                        except:
                            pass
            except Exception as e:
                print(f"Ошибка при обновлении кадра: {e}")
            if self.window2_active:
                try:
                    self.root.after(30, update_frame)
                except:
                    pass
            else:
                try:
                    self.button_window.destroy()
                except:
                    pass

        update_frame()

    def start_calibration(self):
        try:
            self.calibration_button.config(state="disabled", text="Калибровка...")
        except:
            pass
        try:
            self.button_window.destroy()
        except:
            pass
        self.calibrating = True
        self.window2_active = False
        calibration_start_time = time.time()
        horizontal_values = []
        vertical_values   = []

        def calibration_loop():
            if not self.calibrating:
                return
            elapsed   = time.time() - calibration_start_time
            remaining = self.main_app.gaze_tracker.calibration_time - elapsed
            if remaining <= 0:
                if horizontal_values and vertical_values:
                    self.main_app.gaze_tracker.horizontal_center = sum(horizontal_values) / len(horizontal_values)
                    self.main_app.gaze_tracker.vertical_center   = sum(vertical_values)   / len(vertical_values)
                    self.main_app.gaze_tracker.calibrated = True
                    print(f"Калибровка завершена. Центр: H={self.main_app.gaze_tracker.horizontal_center:.2f}, V={self.main_app.gaze_tracker.vertical_center:.2f}")
                else:
                    print("Ошибка калибровки глаз. Используются значения по умолчанию.")
                self.main_app.head_pose_tracker.finalize_calibration()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                calibration_log_entry = {
                    "timestamp": timestamp,
                    "data": {
                        "participant_number": self.main_app.participant_number or self.participant_number or "unknown",
                        "message": "Калибровка завершена (взгляд + голова)",
                        "horizontal_ratio":   self.main_app.gaze_tracker.horizontal_center,
                        "vertical_ratio":     self.main_app.gaze_tracker.vertical_center,
                        "head_neutral_yaw":   self.main_app.head_pose_tracker.neutral_yaw,
                        "head_neutral_pitch": self.main_app.head_pose_tracker.neutral_pitch,
                    }
                }
                self.main_app.logger.behavior_logs.append(calibration_log_entry)
                self.main_app.logger.save_logs_to_file()
                self.calibrating = False
                self.calibration_completed = True
                try:
                    self.calibration_overlay.destroy()
                except:
                    pass
                try:
                    cv2.destroyWindow("Calibration Window")
                except:
                    pass
                time.sleep(0.2)
                from gaze_tracking import GazeTracking
                self.main_app.gaze_tracker.gaze = GazeTracking()
                if self.main_app.gaze_tracker.camera is not None:
                    for i in range(3):
                        test_ret, test_frame = self.main_app.gaze_tracker.camera.read()
                        if test_ret and test_frame is not None:
                            self.main_app.gaze_tracker.gaze.refresh(test_frame)
                        time.sleep(0.1)
                    test_ret, test_frame = self.main_app.gaze_tracker.camera.read()
                    if test_ret and test_frame is not None:
                        self.main_app.gaze_tracker.gaze.refresh(test_frame)
                        if not self.main_app.gaze_tracker.gaze.pupils_located:
                            print("Переинициализация камеры после калибровки...")
                            self.main_app.gaze_tracker.camera.release()
                            time.sleep(0.2)
                            self.main_app.gaze_tracker.camera = cv2.VideoCapture(0)
                            time.sleep(0.3)
                            test_ret, test_frame = self.main_app.gaze_tracker.camera.read()
                            if test_ret and test_frame is not None:
                                self.main_app.gaze_tracker.gaze.refresh(test_frame)
                if self.root:
                    self.root.after(0, self.show_window3)
                return
            try:
                frame, eyes_detected = self.main_app.gaze_tracker.get_frame_with_eyes_status()
                if frame is not None:
                    gaze = self.main_app.gaze_tracker.gaze
                    if gaze.horizontal_ratio() is not None:
                        horizontal_values.append(gaze.horizontal_ratio())
                    if gaze.vertical_ratio() is not None:
                        vertical_values.append(gaze.vertical_ratio())
                    raw_ret, raw_frame = self.main_app.gaze_tracker.camera.read()
                    if raw_ret and raw_frame is not None:
                        self.main_app.head_pose_tracker.accumulate_calibration(raw_frame)
                    try:
                        self.progress_label.config(text=f"Калибровка... Осталось {int(remaining)} секунд. Не двигайте.")
                    except:
                        pass
                    cv2.imshow("Calibration Window", frame)
                    cv2.waitKey(1)
            except Exception as e:
                print(f"Ошибка в цикле калибровки: {e}")
            if self.calibrating and self.root:
                self.root.after(30, calibration_loop)

        if self.root:
            self.root.after(100, calibration_loop)

    def show_window3(self):
        self.testing_active = True
        try:
            if self.button_window:
                self.button_window.destroy()
        except:
            pass
        cv2.destroyAllWindows()
        self.window3 = tk.Toplevel(self.root)
        self.window3.title("Тестирование")
        self.window3.geometry("350x150")
        self.window3.resizable(False, False)
        self.window3.attributes('-topmost', True)
        self.window3.attributes('-toolwindow', True)
        self.window3.update_idletasks()
        x = (self.window3.winfo_screenwidth()  // 2) - 175
        y = (self.window3.winfo_screenheight() // 2) - 75
        self.window3.geometry(f"350x150+{x}+{y}")
        label = tk.Label(self.window3, text="Тестирование проходит", font=("Arial", 14, "bold"))
        label.pack(pady=15)
        button_frame = tk.Frame(self.window3)
        button_frame.pack(pady=10)
        button1 = tk.Button(button_frame, text="Отметить списывание",
                            command=self.log_cheating, font=("Arial", 10),
                            bg="#f44336", fg="white", padx=15, pady=8)
        button1.pack(side=tk.LEFT, padx=10)
        button2 = tk.Button(button_frame, text="Завершить тест",
                            command=self.end_testing, font=("Arial", 10),
                            bg="#4CAF50", fg="white", padx=15, pady=8)
        button2.pack(side=tk.LEFT, padx=10)
        if self.main_app.gaze_tracker.camera is not None:
            _ret, _frame = self.main_app.gaze_tracker.camera.read()
            if _ret and _frame is not None:
                _h, _w    = _frame.shape[:2]
                _participant = self.main_app.participant_number or self.participant_number or "unknown"
                self.main_app.recorder.start_recording((_w, _h), _participant)
                self.main_app.gaze_tracker.gaze.refresh(_frame)
        self._testing_thread = threading.Thread(target=self.run_testing_loop, daemon=True)
        self._testing_thread.start()

    def log_cheating(self):
        winsound.Beep(1000, 300)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.main_app.logger.gaze_logs.append(f"{timestamp}: Отмечена попытка списывания (по нажатию кнопки)")
        self.main_app.logger.behavior_logs.append({
            "timestamp": timestamp,
            "data": {
                "event_type": "manual_cheating_mark",
                "message":    "Пользователь отметил попытку списывания по нажатию кнопки",
            }
        })
        self.main_app.logger.save_logs_to_file()

    def end_testing(self):
        self.testing_active = False
        if self._testing_thread is not None and self._testing_thread.is_alive():
            self._testing_thread.join(timeout=10)
        try:
            if self.window3:
                self.window3.destroy()
        except:
            pass
        try:
            if self.root:
                self.root.quit()
        except:
            pass
        self.main_app.stop()

    def run_testing_loop(self):
        """Основной цикл тестирования."""
        print(f"[DEBUG] run_testing_loop запущен. testing_active={self.testing_active}")
        time.sleep(0.5)
        iteration = 0
        while self.testing_active:
            try:
                camera = self.main_app.gaze_tracker.camera
                if camera is None:
                    time.sleep(self.main_app.sleep_interval)
                    iteration += 1
                    continue
                ret, frame = camera.read()
                if not ret or frame is None:
                    try:
                        camera.release()
                    except Exception:
                        pass
                    time.sleep(0.1)
                    self.main_app.gaze_tracker.camera = cv2.VideoCapture(0)
                    time.sleep(0.2)
                    iteration += 1
                    continue

                self.main_app.recorder.write_webcam_frame(frame)
                screen_frame = self.main_app.recorder.capture_screen_frame()
                if screen_frame is not None:
                    self.main_app.recorder.write_screen_frame(screen_frame)

                self.main_app.gaze_tracker.gaze.refresh(frame)
                gaze_info = self.main_app.gaze_tracker.get_gaze_direction()
                gaze_data = gaze_info.get("direction")

                head_info = self.main_app.head_pose_tracker.get_head_direction(frame)

                cv2.waitKey(1)

                if gaze_data and gaze_data != "not calibrated":
                    self.main_app.ui.display_gaze_data(gaze_data)
                    self.main_app.behavior_analyzer.analyze_gaze_pattern(gaze_data, gaze_info)
                    self.main_app.logger.log_gaze_data(gaze_data, gaze_info)
                elif gaze_data == "not calibrated":
                    print(f"Предупреждение: калибровка не завершена. calibrated={self.main_app.gaze_tracker.calibrated}")

                if head_info.get("direction") not in ("not calibrated",):
                    self.main_app.behavior_analyzer.analyze_head_pose(head_info)
                    self.main_app.logger.log_head_data(head_info)

                if self.main_app.behavior_analyzer.detect_cheating():
                    self.main_app.ui.show_alert()
                    report = self.main_app.behavior_analyzer.generate_report()

                    self.main_app.logger.log_behavior(report)
                    self.main_app.logger.save_logs_to_file()
                    self.main_app.ui.display_report(report)

                    calib = {
                        "horizontal_ratio":   self.main_app.gaze_tracker.horizontal_center,
                        "vertical_ratio":     self.main_app.gaze_tracker.vertical_center,
                        "head_neutral_yaw":   self.main_app.head_pose_tracker.neutral_yaw,
                        "head_neutral_pitch": self.main_app.head_pose_tracker.neutral_pitch,
                    }
                    self.main_app.model_referee.log_verdict(report, calib)

                iteration += 1
                time.sleep(self.main_app.sleep_interval)
            except Exception as e:
                print(f"Ошибка в цикле тестирования (итерация {iteration}): {e}")
                import traceback
                traceback.print_exc()
                break
        print(f"[DEBUG] run_testing_loop завершён. Итераций: {iteration}")


class SessionRecorder:
    def __init__(self):
        self.recordings_dir = Path(CONFIG["recordings_dir"])
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.recording_fps = CONFIG["recording_fps"]
        self.screen_file_path: Optional[Path] = None
        self.webcam_file_path: Optional[Path] = None
        self.combined_file_path: Optional[Path] = None
        self.screen_writer = None
        self.webcam_writer = None
        self.screen_size: Optional[Tuple[int, int]] = None
        self.webcam_size: Optional[Tuple[int, int]] = None
        self.session_timestamp: Optional[str] = None
        self.recording_active = False
        self.session_start_time: Optional[float] = None
        self.session_end_time: Optional[float] = None
        self._screen_frame_idx = 0
        self._webcam_frame_idx = 0
        self._last_screen_time: Optional[float] = None
        self._last_webcam_time: Optional[float] = None
        self._screen_accum = 0.0
        self._webcam_accum = 0.0

    def _build_session_prefix(self, participant_number: str = None) -> str:
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        p = f"participant_{participant_number}" if participant_number else "participant_unknown"
        return f"{p}_{self.session_timestamp}"

    def start_recording(self, webcam_frame_size: Tuple[int, int], participant_number: str = None) -> None:
        if self.recording_active:
            return
        prefix = self._build_session_prefix(participant_number)
        self.screen_file_path  = self.recordings_dir / f"{prefix}_screen.avi"
        self.webcam_file_path  = self.recordings_dir / f"{prefix}_webcam.avi"
        self.combined_file_path = self.recordings_dir / f"{prefix}_combined.avi"
        with mss() as sct:
            monitor = sct.monitors[1]
            self.screen_size = (monitor["width"], monitor["height"])
        self.webcam_size = webcam_frame_size
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.screen_writer = cv2.VideoWriter(str(self.screen_file_path), fourcc, float(self.recording_fps), self.screen_size)
        self.webcam_writer = cv2.VideoWriter(str(self.webcam_file_path), fourcc, float(self.recording_fps), self.webcam_size)
        if not self.screen_writer.isOpened() or not self.webcam_writer.isOpened():
            print("ОШИБКА: не удалось открыть VideoWriter для записи видео.")
            if self.screen_writer is not None: self.screen_writer.release()
            if self.webcam_writer is not None: self.webcam_writer.release()
            self.screen_writer = None
            self.webcam_writer = None
            self.recording_active = False
            return
        self._screen_frame_idx = 0
        self._webcam_frame_idx = 0
        self._last_screen_time = None
        self._last_webcam_time = None
        self._screen_accum = 0.0
        self._webcam_accum = 0.0
        self.session_start_time = time.time()
        self.session_end_time   = None
        self.recording_active   = True
        print("Запись сессии начата.")
        print(f" Экран: {self.screen_file_path.name}")
        print(f" Веб-камера: {self.webcam_file_path.name}")

    def capture_screen_frame(self):
        if not self.recording_active or self.screen_size is None:
            return None
        try:
            with mss() as sct:
                monitor = sct.monitors[1]
                img = np.array(sct.grab(monitor))
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f" Ошибка захвата экрана: {e}")
            return None

    def _write_with_timing(self, writer, frame, last_time_attr: str, accum_attr: str, idx_attr: str) -> None:
        if not self.recording_active or writer is None or frame is None:
            return
        now = time.time()
        last_time = getattr(self, last_time_attr)
        if last_time is None:
            writer.write(frame)
            setattr(self, idx_attr, getattr(self, idx_attr) + 1)
            setattr(self, last_time_attr, now)
            return
        delta  = now - last_time
        setattr(self, last_time_attr, now)
        accum  = getattr(self, accum_attr) + delta
        frame_interval = 1.0 / max(1, self.recording_fps)
        while accum >= frame_interval:
            writer.write(frame)
            setattr(self, idx_attr, getattr(self, idx_attr) + 1)
            accum -= frame_interval
        setattr(self, accum_attr, accum)

    def write_screen_frame(self, frame) -> None:
        self._write_with_timing(self.screen_writer, frame, '_last_screen_time', '_screen_accum', '_screen_frame_idx')

    def write_webcam_frame(self, frame) -> None:
        self._write_with_timing(self.webcam_writer, frame, '_last_webcam_time', '_webcam_accum', '_webcam_frame_idx')

    def create_combined_video_stub(self) -> None:
        if self.screen_file_path and self.webcam_file_path and self.combined_file_path:
            print("Заглушка create_combined_video вызвана:")
            print(f" Экран: {self.screen_file_path.name}")
            print(f" Веб-камера: {self.webcam_file_path.name}")
            print(f" Результат: {self.combined_file_path.name} (не создан)")

    def stop_recording(self) -> None:
        if not self.recording_active:
            return
        self.recording_active = False
        self.session_end_time = time.time()
        if self.screen_writer is not None:
            self.screen_writer.release()
            self.screen_writer = None
        if self.webcam_writer is not None:
            self.webcam_writer.release()
            self.webcam_writer = None
        print(f"Запись остановлена. Кадров записано: экран={self._screen_frame_idx}, камера={self._webcam_frame_idx}")
        self.create_combined_video_stub()


class DataLogger:
    def __init__(self):
        self.gaze_logs     = []
        self.behavior_logs = []
        self.logs_dir          = CONFIG["logs_dir"]
        self.gaze_log_file     = Path(self.logs_dir) / CONFIG["gaze_log_file"]
        self.behavior_log_file = Path(self.logs_dir) / CONFIG["behavior_log_file"]

    def log_gaze_data(self, gaze_data: str, gaze_info: Dict[str, any] = None) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"{timestamp}: {gaze_data}"
        if gaze_info:
            h_dev     = gaze_info.get("horizontal_deviation")
            v_dev     = gaze_info.get("vertical_deviation")
            total_dev = gaze_info.get("total_deviation")
            angle_real      = gaze_info.get("angle_from_real_center")
            angle_calibrated = gaze_info.get("angle_from_calibrated_center")
            if h_dev is not None and v_dev is not None:
                log_entry += f" | H_dev: {h_dev:.3f}, V_dev: {v_dev:.3f}, Total_dev: {total_dev:.3f}"
            if angle_real is not None:
                log_entry += f" | Angle_real: {angle_real:.2f}°"
            if angle_calibrated is not None:
                log_entry += f" | Angle_calibrated: {angle_calibrated:.2f}°"
        self.gaze_logs.append(log_entry)

    def log_head_data(self, head_info: Dict[str, any]) -> None:
        if head_info is None:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        direction = head_info.get("direction", "unknown")
        yaw_dev   = head_info.get("yaw_deviation")
        pitch_dev = head_info.get("pitch_deviation")
        is_susp   = head_info.get("is_suspicious", False)
        log_entry = f"{timestamp}: [HEAD] {direction}"
        if yaw_dev is not None:
            log_entry += f" | Yaw_dev: {yaw_dev:.1f}°, Pitch_dev: {pitch_dev:.1f}°"
        if is_susp:
            log_entry += " | SUSPICIOUS"
        self.gaze_logs.append(log_entry)

    def log_behavior(self, behavior_data: Dict[str, any]) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = {"timestamp": timestamp, "data": behavior_data}
        self.behavior_logs.append(log_entry)

    def save_logs_to_file(self) -> None:
        if self.gaze_logs:
            with open(self.gaze_log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(self.gaze_logs) + "\n")
            self.gaze_logs = []
        if self.behavior_logs:
            try:
                if self.behavior_log_file.exists():
                    with open(self.behavior_log_file, "r", encoding="utf-8") as f:
                        existing_logs = json.load(f)
                else:
                    existing_logs = []
            except json.JSONDecodeError:
                existing_logs = []
            existing_logs.extend(self.behavior_logs)
            with open(self.behavior_log_file, "w", encoding="utf-8") as f:
                json.dump(existing_logs, f, indent=2, ensure_ascii=False)
            self.behavior_logs = []


class MainApp:
    def __init__(self):
        self.gaze_tracker     = GazeTracker()
        self.head_pose_tracker = HeadPoseTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.ui               = UIInterface()
        self.logger           = DataLogger()
        self.recorder         = SessionRecorder()
        self.model_referee    = ModelReferee()
        self.sleep_interval   = CONFIG["sleep_interval"]
        self.gui              = None
        self.participant_number = None

    def run(self) -> None:
        try:
            self.gui = GUIInterface(self)
            self.gui.show_window1()
            if self.gui.participant_number:
                self.participant_number = self.gui.participant_number
        except Exception as e:
            print(f"Ошибка при запуске приложения: {e}")
            self.stop()

    def run_legacy(self) -> None:
        try:
            participant_number = input("Введите номер участника: ")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.logger.gaze_logs.append(f"{timestamp}: Номер участника: {participant_number}")
            self.logger.behavior_logs.append({
                "timestamp": timestamp,
                "data": {
                    "participant_number": participant_number,
                    "message": "Начало сеанса, номер участника записан",
                }
            })
            self.logger.save_logs_to_file()
            self.gaze_tracker.initialize_camera()
            print("Калибровка завершена. Приложение запущено. Нажмите C для остановки.")
            if self.gaze_tracker.calibrated:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.logger.behavior_logs.append({
                    "timestamp": timestamp,
                    "data": {
                        "participant_number": participant_number,
                        "message": "Калибровка завершена",
                        "horizontal_ratio": self.gaze_tracker.horizontal_center,
                        "vertical_ratio":   self.gaze_tracker.vertical_center,
                    }
                })
                self.logger.save_logs_to_file()
            while True:
                gaze_data = self.gaze_tracker.detect_gaze()
                if gaze_data and gaze_data != "not calibrated":
                    gaze_info = self.gaze_tracker.get_gaze_direction()
                    self.ui.display_gaze_data(gaze_data)
                    self.behavior_analyzer.analyze_gaze_pattern(gaze_data, gaze_info)
                    self.logger.log_gaze_data(gaze_data, gaze_info)
                if self.gaze_tracker.camera is not None:
                    ret, raw_frame = self.gaze_tracker.camera.read()
                    if ret and raw_frame is not None:
                        head_info = self.head_pose_tracker.get_head_direction(raw_frame)
                        if head_info.get("direction") not in ("not calibrated",):
                            self.behavior_analyzer.analyze_head_pose(head_info)
                            self.logger.log_head_data(head_info)
                if self.behavior_analyzer.detect_cheating():
                    self.ui.show_alert()
                    report = self.behavior_analyzer.generate_report()
                    self.logger.log_behavior(report)
                    self.logger.save_logs_to_file()
                    self.ui.display_report(report)
                    # Вердикт модели в legacy-режиме
                    calib = {
                        "horizontal_ratio":   self.gaze_tracker.horizontal_center,
                        "vertical_ratio":     self.gaze_tracker.vertical_center,
                        "head_neutral_yaw":   self.head_pose_tracker.neutral_yaw,
                        "head_neutral_pitch": self.head_pose_tracker.neutral_pitch,
                    }
                    self.model_referee.log_verdict(report, calib)
                time.sleep(self.sleep_interval)
                key = cv2.waitKey(1)
                if key == ord('x') or key == ord('ч'):
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    self.logger.gaze_logs.append(f"{timestamp}: Отмечена попытка списывания (по нажатию X)")
                    self.logger.behavior_logs.append({
                        "timestamp": timestamp,
                        "data": {
                            "event_type": "manual_cheating_mark",
                            "message":    "Пользователь отметил попытку списывания по нажатию X",
                        }
                    })
                    self.logger.save_logs_to_file()
                    print("Попытка списывания отмечена в логах")
                if key == ord('c') or key == ord('с'):
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self.recorder.stop_recording()
        self.gaze_tracker.release_camera()
        self.head_pose_tracker.release()
        self.logger.save_logs_to_file()
        print("Общий лог активности успешно сохранен!")
        print("Лог подозрительной активности успешно сохранен!")
        print("Приложение остановлено.")


if __name__ == "__main__":
    app = MainApp()
    app.run()
