import sys
import os
import types
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.makedirs("logs", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

_cv2 = MagicMock()
_cv2.COLOR_BGR2RGB       = 4
_cv2.COLOR_BGRA2BGR      = 0
_cv2.WND_PROP_FULLSCREEN = 0
_cv2.WINDOW_FULLSCREEN   = 1
_cv2.FONT_HERSHEY_SIMPLEX = 0
_cv2.SOLVEPNP_ITERATIVE  = 0
_cv2.VideoCapture        = MagicMock()
_cv2.VideoWriter         = MagicMock()
_cv2.VideoWriter_fourcc  = MagicMock(return_value=0)
_cv2.cvtColor            = MagicMock(side_effect=lambda x, *a: x)
_cv2.solvePnP            = MagicMock(return_value=(True, np.zeros((3,1)), np.zeros((3,1))))
_cv2.Rodrigues           = MagicMock(return_value=(np.eye(3, dtype=np.float64), None))
_cv2.resize              = MagicMock()
_cv2.putText             = MagicMock()
_cv2.imshow              = MagicMock()
_cv2.waitKey             = MagicMock(return_value=-1)
_cv2.destroyAllWindows   = MagicMock()
_cv2.namedWindow         = MagicMock()
_cv2.setWindowProperty   = MagicMock()
_cv2.destroyWindow       = MagicMock()

sys.modules.setdefault("cv2",           _cv2)
sys.modules.setdefault("gaze_tracking", MagicMock())

_mss_sct = MagicMock()
_mss_sct.__enter__ = MagicMock(return_value=_mss_sct)
_mss_sct.__exit__  = MagicMock(return_value=False)
_mss_sct.monitors  = [{}, {"width": 1920, "height": 1080, "left": 0, "top": 0}]
_mss_sct.grab      = MagicMock(return_value=np.zeros((1080, 1920, 4), dtype=np.uint8))
_mss_mod           = MagicMock()
_mss_mod.mss       = MagicMock(return_value=_mss_sct)
sys.modules["mss"] = _mss_mod

_winsound = MagicMock()
sys.modules["winsound"] = _winsound

_tk = types.ModuleType("tkinter")
for _attr in ("Tk","Toplevel","Label","Button","Entry","Frame",
              "Canvas","ttk","messagebox","BOTH","LEFT","TOP",
              "StringVar","IntVar","BooleanVar"):
    setattr(_tk, _attr, MagicMock())
sys.modules["tkinter"]            = _tk
sys.modules["tkinter.ttk"]        = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()

_mp = MagicMock()
_mp.solutions.face_mesh.FaceMesh.return_value = MagicMock()
sys.modules["mediapipe"] = _mp
