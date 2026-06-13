import cv2
from gaze_tracking import GazeTracking
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GazeData:
    timestamp: float
    is_left: bool
    is_right: bool
    is_center: bool
    is_blinking: bool


class GazeTracker:
    def __init__(self):
        self.gaze = GazeTracking()
        self.webcam = cv2.VideoCapture(0)  # Инициализация камеры

    def detect_gaze(self) -> Optional[GazeData]:
        ret, frame = self.webcam.read()
        if not ret:
            return None

        self.gaze.refresh(frame)
        frame = self.gaze.annotated_frame()  # Рамка с выделенными глазами

        gaze_data = GazeData(
            timestamp=time.time(),
            is_left=self.gaze.is_left(),
            is_right=self.gaze.is_right(),
            is_center=self.gaze.is_center(),
            is_blinking=self.gaze.is_blinking()
        )

        cv2.imshow("Gaze Tracking", frame)  # Показ кадра с аннотацией
        if cv2.waitKey(1) == 27:  # ESC для выхода
            return None

        return gaze_data

    def release(self):
        self.webcam.release()
        cv2.destroyAllWindows()


class BehaviorAnalyzer:
    def __init__(self, gaze_history_max=10):
        self.gaze_history: List[GazeData] = []
        self.gaze_history_max = gaze_history_max

    def analyze(self, gaze_data: GazeData) -> bool:
        """Анализирует паттерны взгляда, возвращает True, если есть подозрение на списывание."""
        self.gaze_history.append(gaze_data)
        if len(self.gaze_history) > self.gaze_history_max:
            self.gaze_history.pop(0)

        # Подозрительное поведение: слишком частые взгляды в сторону
        suspicious_look_count = sum(1 for g in self.gaze_history if g.is_left or g.is_right)
        cheating_threshold = 5  # Настройка порога

        return suspicious_look_count >= cheating_threshold


class CheatingDetectorApp:
    def __init__(self):
        self.gaze_tracker = GazeTracker()
        self.behavior_analyzer = BehaviorAnalyzer()

    def run(self):
        try:
            while True:
                gaze_data = self.gaze_tracker.detect_gaze()
                if gaze_data is None:
                    break

                is_cheating = self.behavior_analyzer.analyze(gaze_data)
                if is_cheating:
                    print("⚠️ Подозрительное поведение! Возможно, студент списывает.")

        finally:
            self.gaze_tracker.release()


if __name__ == "__main__":
    app = CheatingDetectorApp()
    app.run()