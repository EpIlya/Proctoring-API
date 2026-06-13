import json
import sys
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

for mod_name in ("cv2", "gaze_tracking", "mediapipe", "mss", "winsound"):
    sys.modules.setdefault(mod_name, MagicMock())

import types
_tk = types.ModuleType("tkinter")
for _attr in ("Tk","Toplevel","Label","Button","Entry","Frame",
              "Canvas","ttk","messagebox","BOTH","LEFT","TOP"):
    setattr(_tk, _attr, MagicMock())
sys.modules["tkinter"] = _tk
sys.modules["tkinter.ttk"] = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()

mp_mock = MagicMock()
mp_mock.solutions.face_mesh.FaceMesh.return_value = MagicMock()
sys.modules["mediapipe"] = mp_mock

import main_with_head_tracking_model as mod


FEATURE_NAMES = [
    "suspicious_actions", "cheating_trigger", "gaze_trigger_count",
    "head_trigger_count", "calib_horizontal_ratio", "calib_vertical_ratio",
    "calib_head_neutral_yaw", "calib_head_neutral_pitch",
    "gaze_num_mean", "gaze_num_std", "gaze_num_max", "gaze_num_min",
    "gaze_total_dev_mean", "gaze_total_dev_std", "gaze_total_dev_max",
    "gaze_calib_angle_mean", "gaze_calib_angle_std",
    "gaze_dir_non_center_ratio", "gaze_dir_up_ratio",
    "gaze_dir_down_ratio", "gaze_dir_blink_ratio", "gaze_n_frames",
    "head_num_mean", "head_num_std", "head_yaw_abs_mean", "head_yaw_abs_max",
    "head_pitch_abs_mean", "head_pitch_abs_max",
    "head_dir_non_center_ratio", "head_dir_up_ratio", "head_dir_down_ratio",
    "head_n_frames", "head_is_suspicious_ratio", "head_is_suspicious_any",
]


def make_referee_no_model(tmp_path):
    ref = mod.ModelReferee.__new__(mod.ModelReferee)
    ref.logs_dir       = tmp_path
    ref.log_file       = tmp_path / "behavior_log_with_model.json"
    ref.pkl_path       = tmp_path / "best_model.pkl"
    ref._model         = None
    ref._preprocessor  = None
    ref._feat_names    = None
    ref._model_loaded  = False
    return ref


def make_referee_with_model(tmp_path):
    ref = make_referee_no_model(tmp_path)
    model_mock = MagicMock()
    model_mock.predict.return_value = [1]
    model_mock.predict_proba.return_value = [[0.1, 0.9]]
    prep_mock = MagicMock()
    prep_mock.transform.return_value = np.zeros((1, len(FEATURE_NAMES)))
    ref._model         = model_mock
    ref._preprocessor  = prep_mock
    ref._feat_names    = FEATURE_NAMES
    ref._model_loaded  = True
    return ref


def sample_report(gaze_trigger=1, head_trigger=0):
    return {
        "suspicious_actions": gaze_trigger + head_trigger,
        "gaze_history": [
            {"direction": "left", "timestamp": time.time(),
             "total_deviation": 0.3, "angle_from_calibrated_center": 20.0},
            {"direction": "center", "timestamp": time.time(),
             "total_deviation": 0.05, "angle_from_calibrated_center": 2.0},
        ],
        "head_history": [
            {"direction": "right", "is_suspicious": True,
             "yaw_deviation": 20.0, "pitch_deviation": 3.0,
             "timestamp": time.time()},
        ],
        "current_status": "cheating" if (gaze_trigger + head_trigger) > 0 else "normal",
        "cheating_trigger": "gaze" if gaze_trigger > 0 else "head_pose",
        "gaze_trigger_count": gaze_trigger,
        "head_trigger_count": head_trigger,
    }


def sample_calib():
    return {
        "horizontal_ratio":   0.51,
        "vertical_ratio":     0.49,
        "head_neutral_yaw":   2.0,
        "head_neutral_pitch": -1.0,
    }

class TestLoadModel:
    def test_model_loaded_when_all_files_present(self, tmp_path):
        import joblib
        model_mock = MagicMock()
        prep_mock  = MagicMock()
        feat_names = FEATURE_NAMES

        with patch("joblib.load") as load_mock:
            load_mock.side_effect = [model_mock, prep_mock, feat_names]
            (tmp_path / "best_model.pkl").touch()
            (tmp_path / "preprocessor.pkl").touch()
            (tmp_path / "feature_names.pkl").touch()
            ref = make_referee_no_model(tmp_path)
            ref._load_model()

        assert ref._model_loaded is True

    def test_load_exception_handled_gracefully(self, tmp_path):
        (tmp_path / "best_model.pkl").touch()
        (tmp_path / "preprocessor.pkl").touch()
        (tmp_path / "feature_names.pkl").touch()
        ref = make_referee_no_model(tmp_path)
        with patch("joblib.load", side_effect=Exception("corrupt file")):
            ref._load_model()
        assert ref._model_loaded is False

class TestExtractFeatures:
    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        import pandas as pd
        ref    = make_referee_with_model(tmp_path)
        report = sample_report()
        calib  = sample_calib()
        df     = ref._extract_features(report, calib)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == FEATURE_NAMES

class TestPredict:
    def test_model_unavailable_when_not_loaded(self, tmp_path):
        ref    = make_referee_no_model(tmp_path)
        result = ref.predict(sample_report(), sample_calib())
        assert result["model_verdict"]     == "model_unavailable"
        assert result["model_probability"] is None

    def test_predict_returns_cheating(self, tmp_path):
        ref = make_referee_with_model(tmp_path)
        ref._model.predict.return_value = [1]
        ref._model.predict_proba.return_value = [[0.05, 0.95]]
        result = ref.predict(sample_report(), sample_calib())
        assert result["model_verdict"]     == "cheating"
        assert result["model_probability"] == pytest.approx(0.95)

    def test_predict_returns_normal(self, tmp_path):
        ref = make_referee_with_model(tmp_path)
        ref._model.predict.return_value = [0]
        ref._model.predict_proba.return_value = [[0.8, 0.2]]
        result = ref.predict(sample_report(), sample_calib())
        assert result["model_verdict"]     == "normal"
        assert result["model_probability"] == pytest.approx(0.2)