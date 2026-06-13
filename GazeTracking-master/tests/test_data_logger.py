import json
import os
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("gaze_tracking", MagicMock())
sys.modules.setdefault("mediapipe", MagicMock())
sys.modules.setdefault("mss", MagicMock())
sys.modules.setdefault("winsound", MagicMock())
sys.modules.setdefault("tkinter", MagicMock())

import importlib, types
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

import numpy as np
np_mock = sys.modules.get("numpy", np)

with patch.dict("os.environ", {}):
    import importlib, main_with_head_tracking_model as mod

@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setitem(mod.CONFIG, "logs_dir",          str(tmp_path))
    monkeypatch.setitem(mod.CONFIG, "gaze_log_file",     "gaze_log.txt")
    monkeypatch.setitem(mod.CONFIG, "behavior_log_file", "behavior_log.json")
    tmp_path.mkdir(exist_ok=True)
    yield tmp_path


@pytest.fixture
def logger(tmp_logs):
    """Создаём свежий DataLogger с tmp-путями."""
    dl = mod.DataLogger.__new__(mod.DataLogger)
    dl.gaze_logs     = []
    dl.behavior_logs = []
    dl.logs_dir      = str(tmp_logs)
    dl.gaze_log_file     = tmp_logs / "gaze_log.txt"
    dl.behavior_log_file = tmp_logs / "behavior_log.json"
    return dl

class TestDataLoggerInit:
    def test_initial_lists_empty(self, logger):
        assert logger.gaze_logs     == []
        assert logger.behavior_logs == []

    def test_log_file_paths_set(self, logger, tmp_logs):
        assert "gaze_log.txt"     in str(logger.gaze_log_file)
        assert "behavior_log.json" in str(logger.behavior_log_file)

class TestLogGazeData:
    def test_basic_direction_appended(self, logger):
        logger.log_gaze_data("left")
        assert len(logger.gaze_logs) == 1
        assert "left" in logger.gaze_logs[0]

    def test_timestamp_format_in_entry(self, logger):
        logger.log_gaze_data("center")
        entry = logger.gaze_logs[0]
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", entry)

    def test_multiple_entries_accumulate(self, logger):
        for d in ["center", "left", "right", "up", "down"]:
            logger.log_gaze_data(d)
        assert len(logger.gaze_logs) == 5

class TestLogHeadData:
    def test_suspicious_flag_in_entry(self, logger):
        logger.log_head_data({"direction": "right", "yaw_deviation": 25.0,
                              "pitch_deviation": 3.0, "is_suspicious": True})
        assert "SUSPICIOUS" in logger.gaze_logs[0]

    def test_not_suspicious_no_flag(self, logger):
        logger.log_head_data({"direction": "center", "yaw_deviation": 1.0,
                              "pitch_deviation": 1.0, "is_suspicious": False})
        assert "SUSPICIOUS" not in logger.gaze_logs[0]

class TestSaveLogsToFile:
    def test_gaze_logs_written_to_txt(self, logger, tmp_logs):
        logger.gaze_logs = ["entry1", "entry2"]
        logger.save_logs_to_file()
        content = (tmp_logs / "gaze_log.txt").read_text(encoding="utf-8")
        assert "entry1" in content
        assert "entry2" in content

    def test_gaze_logs_cleared_after_save(self, logger):
        logger.gaze_logs = ["entry"]
        logger.save_logs_to_file()
        assert logger.gaze_logs == []

    def test_empty_logs_no_file_written(self, logger, tmp_logs):
        logger.save_logs_to_file()
        assert not (tmp_logs / "gaze_log.txt").exists()
        assert not (tmp_logs / "behavior_log.json").exists()

    def test_corrupt_json_file_handled_gracefully(self, logger, tmp_logs):
        (tmp_logs / "behavior_log.json").write_text("NOT_JSON", encoding="utf-8")
        logger.behavior_logs = [{"timestamp": "t", "data": {}}]
        logger.save_logs_to_file()  # не должно выбросить исключение
        result = json.loads((tmp_logs / "behavior_log.json").read_text())
        assert len(result) == 1

    def test_behavior_logs_preserve_all_fields(self, logger, tmp_logs):
        data = {
            "suspicious_actions": 3,
            "gaze_history": [{"direction": "left"}],
            "current_status": "cheating",
        }
        logger.log_behavior(data)
        logger.save_logs_to_file()
        result = json.loads((tmp_logs / "behavior_log.json").read_text())
        assert result[0]["data"]["suspicious_actions"] == 3
        assert result[0]["data"]["gaze_history"][0]["direction"] == "left"