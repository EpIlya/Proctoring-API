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

def load_config(config_path: str = "config.json") -> Dict:
    default_config = {
        "debug": True,
        "debug_window_size": [800, 600],
        "calibration_threshold": 0.10,
        "max_suspicious_actions": 1,
        "logs_dir": "logs",
        "gaze_log_file": "gaze_log.txt",
        "behavior_log_file": "behavior_log.json",
        "calibration_time": 10,
        "analysis_window": 5,
        "sleep_interval": 0.1
    }

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # Объединяем с дефолтными значениями
        return {**default_config, **config}
    except FileNotFoundError:
        print(f"Файл конфигурации {config_path} не найден, используются значения по умолчанию")
        return default_config


CONFIG = load_config()

Path(CONFIG["logs_dir"]).mkdir(parents=True, exist_ok=True)


class GazeTracker:
    def __init__(self, debug: bool = True, calibration_threshold: float = 0.10):
        self.gaze = GazeTracking()
        self.camera = None
        self.debug = CONFIG["debug"] if debug is None else debug
        self.debug_window_size = tuple(CONFIG["debug_window_size"])
        self.calibration_threshold = CONFIG[
            "calibration_threshold"] if calibration_threshold is None else calibration_threshold
        self.horizontal_center = 0.5
        self.vertical_center = 0.5
        self.calibrated = False
        self.calibration_time = CONFIG["calibration_time"]

        # буфер скользящего среднего
        self._h_buffer = deque(maxlen=7)
        self._v_buffer = deque(maxlen=7)

        # гистерезис
        self.threshold_enter = 0.13  # порог входа в "отклонение"
        self.threshold_exit = 0.07  # порог возврата в "center"
        self._h_offcenter = False
        self._v_offcenter = False

    def initialize_camera(self) -> None:
        """Инициализация камеры с калибровкой."""
        self.camera = cv2.VideoCapture(0)
        self.calibrate()
    
    def initialize_camera_without_calibration(self) -> None:
        """Инициализация камеры без автоматической калибровки."""
        self.camera = cv2.VideoCapture(0)

    def detect_gaze(self, show_debug: bool = None) -> Optional[str]:
        """Определение направления взгляда"""
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

            cv2.putText(debug_frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"H: {gaze_info.get('horizontal_ratio', 0):.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"V: {gaze_info.get('vertical_ratio', 0):.2f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Debug: Eye Tracking", debug_frame)
            cv2.waitKey(1)

        return direction if direction != "not calibrated" else None
    
    def get_frame_with_eyes_status(self) -> Tuple[Optional[any], bool]:
        """Получение кадра с информацией о статусе обнаружения глаз."""
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

    def get_eye_position(self) -> Tuple[int, int]:
        """Получение координат глаз."""
        return self.gaze.pupil_left_coords(), self.gaze.pupil_right_coords()

    def release_camera(self) -> None:
        """Освобождение камеры и закрытие окон"""
        if self.camera is not None:
            self.camera.release()
        if self.debug:
            cv2.destroyAllWindows()

    def calibrate(self, auto_start: bool = False) -> None:
        """Калибровка центрального положения глаз."""
        if not auto_start:
            print("Калибровка: направьте глаза в центр экрана и нажмите 'c'")
            while True:
                _, frame = self.camera.read()
                self.gaze.refresh(frame)
                frame = self.gaze.annotated_frame()

                debug_frame = frame.copy()
                debug_frame = cv2.resize(debug_frame, self.debug_window_size)
                cv2.putText(debug_frame, "Calibration: Look straight and press 'c'",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

        # Показываем прогресс калибровки
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
                cv2.putText(progress_frame, progress_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
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
    
    def calibrate_async(self, update_callback=None) -> None:
        """Асинхронная калибровка."""
        start_time = time.time()
        horizontal_values = []
        vertical_values = []
        
        def calibration_step():
            if time.time() - start_time >= self.calibration_time:
                if horizontal_values and vertical_values:
                    self.horizontal_center = sum(horizontal_values) / len(horizontal_values)
                    self.vertical_center = sum(vertical_values) / len(vertical_values)
                    self.calibrated = True
                    print(f"Калибровка завершена. Центр: H={self.horizontal_center:.2f}, V={self.vertical_center:.2f}")
                else:
                    print("Ошибка калибровки. Используются значения по умолчанию.")
                
                if update_callback:
                    update_callback(completed=True)
                return

            ret, frame = self.camera.read()
            if ret:
                self.gaze.refresh(frame)
                frame = self.gaze.annotated_frame()
                
                # Собираем данные для калибровки
                if self.gaze.horizontal_ratio() is not None:
                    horizontal_values.append(self.gaze.horizontal_ratio())
                if self.gaze.vertical_ratio() is not None:
                    vertical_values.append(self.gaze.vertical_ratio())
                
                # Показываем прогресс на кадре
                elapsed = time.time() - start_time
                remaining = self.calibration_time - elapsed
                progress_text = f"Калибровка... Осталось {int(remaining)} секунд. Не двигайте глазами."
                
                h, w = frame.shape[:2]
                text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h // 2
                cv2.putText(frame, progress_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                if update_callback:
                    update_callback(frame=frame, remaining=int(remaining))
            
            # Планируем следующий шаг
            if self.root:
                self.root.after(30, calibration_step)
        
        # Запускаем первый шаг
        if self.root:
            self.root.after(100, calibration_step)

    def get_gaze_direction(self) -> Dict[str, any]:
        """Определение направления взгляда относительно калиброванного центра."""
        if not self.calibrated:
            return {"direction": "not calibrated"}

        horizontal = self.gaze.horizontal_ratio()
        vertical = self.gaze.vertical_ratio()

        if horizontal is None or vertical is None:
            return {"direction": "blink"}

        # сглаживание скользящим средним
        self._h_buffer.append(horizontal)
        self._v_buffer.append(vertical)
        smooth_h = sum(self._h_buffer) / len(self._h_buffer)
        smooth_v = sum(self._v_buffer) / len(self._v_buffer)

        h_diff = smooth_h - self.horizontal_center
        v_diff = smooth_v - self.vertical_center

        direction = []

        # гистерезис по горизонтали
        if not self._h_offcenter and abs(h_diff) > self.threshold_enter:
            self._h_offcenter = True
        elif self._h_offcenter and abs(h_diff) < self.threshold_exit:
            self._h_offcenter = False

        # гистерезис по вертикали
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

        # Углы считаем от сглаженных значений
        h_diff_real = smooth_h - 0.5
        v_diff_real = smooth_v - 0.5
        angle_from_real_center = math.degrees(math.atan2(v_diff_real, h_diff_real))
        angle_from_calibrated_center = math.degrees(math.atan2(v_diff, h_diff))

        return {
            "direction": " ".join(direction),
            "horizontal_ratio": smooth_h,
            "vertical_ratio": smooth_v,
            "horizontal_deviation": h_diff,
            "vertical_deviation": v_diff,
            "total_deviation": (h_diff ** 2 + v_diff ** 2) ** 0.5,
            "angle_from_real_center": angle_from_real_center,
            "angle_from_calibrated_center": angle_from_calibrated_center
        }


class BehaviorAnalyzer:
    def __init__(self, max_suspicious_actions: int = 1):
        self.suspicious_actions = 0
        self.max_suspicious_actions = CONFIG[
            "max_suspicious_actions"] if max_suspicious_actions is None else max_suspicious_actions
        self.gaze_history = []  
        self.analysis_window = CONFIG["analysis_window"]  
        self.window_size = int(self.analysis_window / CONFIG["sleep_interval"])  
        self.min_consecutive_offcenter = int(2.0 / CONFIG["sleep_interval"]) 
        self.offcenter_threshold = 0.5  
        self.last_direction = "center"
        self.consecutive_offcenter = 0
        self.last_offcenter_time = None

    def analyze_gaze_pattern(self, gaze_data: str, gaze_info: Dict[str, any] = None) -> None:
        """Анализ паттернов взгляда с учетом длительности и процента вне центра."""
        timestamp = time.time()
        history_entry = {
            "direction": gaze_data,
            "timestamp": timestamp
        }
        if gaze_info:
            history_entry.update({
                "horizontal_deviation": gaze_info.get("horizontal_deviation"),
                "vertical_deviation": gaze_info.get("vertical_deviation"),
                "total_deviation": gaze_info.get("total_deviation"),
                "horizontal_ratio": gaze_info.get("horizontal_ratio"),
                "vertical_ratio": gaze_info.get("vertical_ratio"),
                "angle_from_real_center": gaze_info.get("angle_from_real_center"),
                "angle_from_calibrated_center": gaze_info.get("angle_from_calibrated_center")
            })
        
        self.gaze_history.append(history_entry)
        if len(self.gaze_history) > self.window_size:
            self.gaze_history = self.gaze_history[-self.window_size:]

        if gaze_data not in ["center", "blink", "not calibrated"]:
            self.consecutive_offcenter += 1
        else:
            self.consecutive_offcenter -= 1

        self.last_direction = gaze_data

        if self.consecutive_offcenter >= self.min_consecutive_offcenter:
            self.suspicious_actions += 1
            self.consecutive_offcenter = 0  

        offcenter_count = sum(1 for entry in self.gaze_history 
                            if isinstance(entry, dict) and entry.get("direction") not in ["center", "blink", "not calibrated"]
                            or (isinstance(entry, tuple) and entry[1] not in ["center", "blink", "not calibrated"]))
        if len(self.gaze_history) >= self.window_size:
            offcenter_ratio = offcenter_count / self.window_size
            if offcenter_ratio > self.offcenter_threshold:
                self.suspicious_actions += 1
                self.gaze_history = self.gaze_history[-self.window_size//2:]

    def detect_cheating(self) -> bool:
        """Проверка на списывание."""
        return self.suspicious_actions >= self.max_suspicious_actions

    def generate_report(self) -> Dict[str, any]:
        """Генерация отчета."""
        formatted_history = []
        for entry in self.gaze_history:
            if isinstance(entry, dict):
                formatted_history.append(entry)
            else:
                # Старый формат конвертируем в dict
                timestamp, direction = entry
                formatted_history.append({
                    "timestamp": timestamp,
                    "direction": direction
                })
        
        report = {
            "suspicious_actions": self.suspicious_actions,
            "gaze_history": formatted_history,
            "current_status": "cheating" if self.detect_cheating() else "normal"
        }
        self.suspicious_actions = 0
        return report

class UIInterface:
    @staticmethod
    def display_gaze_data(gaze_data: str) -> None:
        """Отображение направления взгляда."""
        #(f"Направление взгляда: {gaze_data}")

    @staticmethod
    def show_alert() -> None:
        """Предупреждение о списывании."""
        #print("Внимание! Обнаружено подозрительное поведение!")

    @staticmethod
    def display_report(report: Dict[str, any]) -> None:
        """Отображение отчета."""
        #print("Отчет о поведении:")
        #print(json.dumps(report, indent=2))

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
        self.root = None
        self.button_window = None

    def show_window1(self):
        """Окно 1: Ввод номера участника."""
        self.root = tk.Tk()
        self.window1 = self.root
        self.window1.title("Начало тестирования")
        self.window1.geometry("400x200")
        self.window1.resizable(False, False)
        
        # Центрирование окна
        self.window1.update_idletasks()
        x = (self.window1.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.window1.winfo_screenheight() // 2) - (200 // 2)
        self.window1.geometry(f"400x200+{x}+{y}")
        
        # Поле ввода
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
            gaze_log_entry = f"{timestamp}: Номер участника: {participant}"
            self.main_app.logger.gaze_logs.append(gaze_log_entry)
            
            behavior_log_entry = {
                "timestamp": timestamp,
                "data": {
                    "participant_number": participant,
                    "message": "Начало сеанса, номер участника записан"
                }
            }
            self.main_app.logger.behavior_logs.append(behavior_log_entry)
            self.main_app.logger.save_logs_to_file()
            
            # Закрываем окно 1 и запускаем окно 2
            self.window1.withdraw()
            self.show_window2()
        
        # Кнопка
        button = tk.Button(self.window1, text="Начать тестирование", 
                          command=start_testing, font=("Arial", 11), 
                          bg="#4CAF50", fg="white", padx=20, pady=10)
        button.pack(pady=20)
        
        # Обработка Enter
        entry.bind("<Return>", lambda e: start_testing())
        
        self.window1.mainloop()
    
    def show_window2(self):
        """Окно 2: Полноэкранное окно с камерой."""
        self.window2_active = True
        self.main_app.gaze_tracker.initialize_camera_without_calibration()
        
        # Создаем полноэкранное окно OpenCV
        cv2.namedWindow("Calibration Window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Создаем tkinter окно для кнопки
        self.button_window = tk.Toplevel(self.root)
        self.button_window.overrideredirect(True)
        self.button_window.attributes('-topmost', True)
        self.button_window.configure(bg='black')
        
        # Получаем размер экрана
        screen_width = self.button_window.winfo_screenwidth()
        screen_height = self.button_window.winfo_screenheight()
        
        # Создаем overlay с красной точкой и инструкцией
        self.calibration_overlay = tk.Toplevel(self.root)
        self.calibration_overlay.overrideredirect(True)
        self.calibration_overlay.attributes('-topmost', True)
        self.calibration_overlay.configure(bg='black')
        try:
            self.calibration_overlay.attributes('-alpha', 0.4)
        except:
            pass
        
        # Устанавливаем размер на весь экран
        self.calibration_overlay.geometry(f"{screen_width}x{screen_height}+0+0")

        self.calibration_canvas = tk.Canvas(self.calibration_overlay, width=screen_width, height=screen_height, 
                          bg='black', highlightthickness=0)
        self.calibration_canvas.pack(fill=tk.BOTH, expand=True)

        dot_radius = 25
        center_x = screen_width // 2
        center_y = screen_height // 2
        self.calibration_canvas.create_oval(center_x - dot_radius, center_y - dot_radius,
                          center_x + dot_radius, center_y + dot_radius,
                          fill='#FF0000', outline='white', width=5, tags='dot')
        
        # Текст инструкции
        instruction_frame = tk.Frame(self.calibration_canvas, bg='black')
        instruction_label = tk.Label(
            instruction_frame,
            text="Во время калибровки смотрите на красную точку",
            font=("Arial", 28, "bold"),
            bg='black',
            fg='white'
        )
        instruction_label.pack()
        self.calibration_canvas.create_window(screen_width // 2, 120, window=instruction_frame, anchor='center')
        
        # Текст прогресса
        progress_frame = tk.Frame(self.calibration_canvas, bg='black')
        self.progress_label = tk.Label(
            progress_frame,
            text="",
            font=("Arial", 36, "bold"),
            bg='black',
            fg='#00FF00'
        )
        self.progress_label.pack()
        progress_y = screen_height // 2 + 150
        self.calibration_canvas.create_window(screen_width // 2, progress_y, window=progress_frame, anchor='center')

        self.calibration_overlay.update()

        self.calibration_button = tk.Button(
            self.button_window, 
            text="Начать калибровку",
            command=self.start_calibration,
            font=("Arial", 14),
            bg="#2196F3",
            fg="white",
            padx=30,
            pady=15,
            state="disabled"
        )
        self.calibration_button.pack(pady=20)

        self.button_window.geometry(f"300x100+{screen_width//2 - 150}+{screen_height - 120}")

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
        """Запуск калибровки."""
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
        vertical_values = []
        
        def calibration_loop():
            """Цикл калибровки с обновлением кадров в реальном времени."""
            if not self.calibrating:
                return
            
            elapsed = time.time() - calibration_start_time
            remaining = self.main_app.gaze_tracker.calibration_time - elapsed
            
            if remaining <= 0:
                if horizontal_values and vertical_values:
                    self.main_app.gaze_tracker.horizontal_center = sum(horizontal_values) / len(horizontal_values)
                    self.main_app.gaze_tracker.vertical_center = sum(vertical_values) / len(vertical_values)
                    self.main_app.gaze_tracker.calibrated = True
                    print(f"Калибровка завершена. Центр: H={self.main_app.gaze_tracker.horizontal_center:.2f}, V={self.main_app.gaze_tracker.vertical_center:.2f}")

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    calibration_log_entry = {
                        "timestamp": timestamp,
                        "data": {
                            "participant_number": self.main_app.participant_number or self.participant_number or "unknown",
                            "message": "Калибровка завершена",
                            "horizontal_ratio": self.main_app.gaze_tracker.horizontal_center,
                            "vertical_ratio": self.main_app.gaze_tracker.vertical_center
                        }
                    }
                    self.main_app.logger.behavior_logs.append(calibration_log_entry)
                    self.main_app.logger.save_logs_to_file()
                else:
                    print("Ошибка калибровки. Используются значения по умолчанию.")
                
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

                # Небольшая задержка для стабилизации
                time.sleep(0.2)
                
                # В .exe файле важно пересоздать объект gaze после калибровки
                # чтобы избежать проблем с состоянием dlib
                from gaze_tracking import GazeTracking
                self.main_app.gaze_tracker.gaze = GazeTracking()
                
                # Проверяем доступность камеры и делаем несколько тестовых чтений
                if self.main_app.gaze_tracker.camera is not None:
                    for i in range(3):
                        test_ret, test_frame = self.main_app.gaze_tracker.camera.read()
                        if test_ret and test_frame is not None:
                            self.main_app.gaze_tracker.gaze.refresh(test_frame)
                        time.sleep(0.1)
                    
                    # Финальная проверка
                    test_ret, test_frame = self.main_app.gaze_tracker.camera.read()
                    if test_ret and test_frame is not None:
                        self.main_app.gaze_tracker.gaze.refresh(test_frame)
                        if not self.main_app.gaze_tracker.gaze.pupils_located:
                            # Если глаза не обнаруживаются, переинициализируем камеру
                            print("Переинициализация камеры после калибровки (глаза не обнаружены)...")
                            self.main_app.gaze_tracker.camera.release()
                            time.sleep(0.2)
                            self.main_app.gaze_tracker.camera = cv2.VideoCapture(0)
                            time.sleep(0.3)
                            test_ret, test_frame = self.main_app.gaze_tracker.camera.read()
                            if test_ret and test_frame is not None:
                                self.main_app.gaze_tracker.gaze.refresh(test_frame)
                
                # После калибровки показываем окно 3
                if self.root:
                    self.root.after(0, self.show_window3)
                return
            
            try:
                frame, eyes_detected = self.main_app.gaze_tracker.get_frame_with_eyes_status()
                
                if frame is not None:
                    # Собираем данные для калибровки
                    gaze = self.main_app.gaze_tracker.gaze
                    if gaze.horizontal_ratio() is not None:
                        horizontal_values.append(gaze.horizontal_ratio())
                    if gaze.vertical_ratio() is not None:
                        vertical_values.append(gaze.vertical_ratio())

                    try:
                        self.progress_label.config(text=f"Калибровка... Осталось {int(remaining)} секунд. Не двигайте глазами.")
                    except:
                        pass

                    cv2.imshow("Calibration Window", frame)
                    cv2.waitKey(1)
            except Exception as e:
                print(f"Ошибка в цикле калибровки: {e}")
            
            # Планируем следующий шаг
            if self.calibrating and self.root:
                self.root.after(30, calibration_loop)
        
        # Запускаем цикл калибровки
        if self.root:
            self.root.after(100, calibration_loop)
    
    def show_window3(self):
        """Окно 3: Окно тестирования поверх всех окон."""
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
        
        # Центрирование окна
        self.window3.update_idletasks()
        x = (self.window3.winfo_screenwidth() // 2) - (350 // 2)
        y = (self.window3.winfo_screenheight() // 2) - (150 // 2)
        self.window3.geometry(f"350x150+{x}+{y}")
        
        # Текст
        label = tk.Label(self.window3, text="Тестирование проходит", 
                        font=("Arial", 14, "bold"))
        label.pack(pady=15)
        
        # Фрейм для кнопок
        button_frame = tk.Frame(self.window3)
        button_frame.pack(pady=10)
        
        # Логирование списывания
        button1 = tk.Button(button_frame, text="Отметить списывание", 
                           command=self.log_cheating, font=("Arial", 10),
                           bg="#f44336", fg="white", padx=15, pady=8)
        button1.pack(side=tk.LEFT, padx=10)
        
        # Завершение теста
        button2 = tk.Button(button_frame, text="Завершить тест", 
                           command=self.end_testing, font=("Arial", 10),
                           bg="#4CAF50", fg="white", padx=15, pady=8)
        button2.pack(side=tk.LEFT, padx=10)

        testing_thread = threading.Thread(target=self.run_testing_loop, daemon=True)
        testing_thread.start()
    
    def log_cheating(self):
        """Логирование попытки списывания."""
        winsound.Beep(1000, 300)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.main_app.logger.gaze_logs.append(f"{timestamp}: Отмечена попытка списывания (по нажатию кнопки)")
        self.main_app.logger.behavior_logs.append({
            "timestamp": timestamp,
            "data": {
                "event_type": "manual_cheating_mark",
                "message": "Пользователь отметил попытку списывания по нажатию кнопки"
            }
        })
        self.main_app.logger.save_logs_to_file()
    
    def end_testing(self):
        """Завершение тестирования."""
        self.testing_active = False
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
        time.sleep(0.5)
        
        while self.testing_active:
            try:
                gaze_data = self.main_app.gaze_tracker.detect_gaze(show_debug=False)

                cv2.waitKey(1)
                
                if gaze_data and gaze_data != "not calibrated":
                    # Получаем полную информацию о взгляде с отклонениями
                    gaze_info = self.main_app.gaze_tracker.get_gaze_direction()
                    self.main_app.ui.display_gaze_data(gaze_data)
                    self.main_app.behavior_analyzer.analyze_gaze_pattern(gaze_data, gaze_info)
                    self.main_app.logger.log_gaze_data(gaze_data, gaze_info)
                    
                    if self.main_app.behavior_analyzer.detect_cheating():
                        self.main_app.ui.show_alert()
                        report = self.main_app.behavior_analyzer.generate_report()
                        self.main_app.logger.log_behavior(report)
                        self.main_app.ui.display_report(report)
                elif gaze_data == "not calibrated":
                    print(f"Предупреждение: калибровка не завершена. calibrated={self.main_app.gaze_tracker.calibrated}")
                
                time.sleep(self.main_app.sleep_interval)
            except Exception as e:
                print(f"Ошибка в цикле тестирования: {e}")
                import traceback
                traceback.print_exc()
                break


class DataLogger:
    def __init__(self):
        self.gaze_logs = []
        self.behavior_logs = []
        self.logs_dir = CONFIG["logs_dir"]
        self.gaze_log_file = Path(self.logs_dir) / CONFIG["gaze_log_file"]
        self.behavior_log_file = Path(self.logs_dir) / CONFIG["behavior_log_file"]

    def log_gaze_data(self, gaze_data: str, gaze_info: Dict[str, any] = None) -> None:
        """Логирование данных о взгляде в память."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"{timestamp}: {gaze_data}"

        if gaze_info:
            h_dev = gaze_info.get("horizontal_deviation")
            v_dev = gaze_info.get("vertical_deviation")
            total_dev = gaze_info.get("total_deviation")
            angle_real = gaze_info.get("angle_from_real_center")
            angle_calibrated = gaze_info.get("angle_from_calibrated_center")
            
            if h_dev is not None and v_dev is not None:
                log_entry += f" | H_dev: {h_dev:.3f}, V_dev: {v_dev:.3f}, Total_dev: {total_dev:.3f}"
            
            if angle_real is not None:
                log_entry += f" | Angle_real: {angle_real:.2f}°"
            
            if angle_calibrated is not None:
                log_entry += f" | Angle_calibrated: {angle_calibrated:.2f}°"
        
        self.gaze_logs.append(log_entry)

    def log_behavior(self, behavior_data: Dict[str, any]) -> None:
        """Логирование данных о поведении в память."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = {
            "timestamp": timestamp,
            "data": behavior_data
        }
        self.behavior_logs.append(log_entry)

    def save_logs_to_file(self) -> None:
        """Сохранение логов из памяти в файлы."""
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
        self.gaze_tracker = GazeTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.ui = UIInterface()
        self.logger = DataLogger()
        self.sleep_interval = CONFIG["sleep_interval"]
        self.gui = None
        self.participant_number = None

    def run(self) -> None:
        """Запуск приложения с GUI."""
        try:
            self.gui = GUIInterface(self)
            self.gui.show_window1()

            if self.gui.participant_number:
                self.participant_number = self.gui.participant_number

        except Exception as e:
            print(f"Ошибка при запуске приложения: {e}")
            self.stop()

    def run_legacy(self) -> None:
        """Запуск приложения (старая версия без GUI)."""
        try:
            participant_number = input("Введите номер участника: ")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            gaze_log_entry = f"{timestamp}: Номер участника: {participant_number}"
            self.logger.gaze_logs.append(gaze_log_entry)

            behavior_log_entry = {
                "timestamp": timestamp,
                "data": {
                    "participant_number": participant_number,
                    "message": "Начало сеанса, номер участника записан"
                }
            }
            self.logger.behavior_logs.append(behavior_log_entry)
            self.logger.save_logs_to_file()

            self.gaze_tracker.initialize_camera()
            print("Калибровка завершена. Приложение запущено. Нажмите C для остановки.")

            if self.gaze_tracker.calibrated:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                calibration_log_entry = {
                    "timestamp": timestamp,
                    "data": {
                        "participant_number": participant_number,
                        "message": "Калибровка завершена",
                        "horizontal_ratio": self.gaze_tracker.horizontal_center,
                        "vertical_ratio": self.gaze_tracker.vertical_center
                    }
                }
                self.logger.behavior_logs.append(calibration_log_entry)
                self.logger.save_logs_to_file()

            while True:
                gaze_data = self.gaze_tracker.detect_gaze()
                if gaze_data and gaze_data != "not calibrated":
                    gaze_info = self.gaze_tracker.get_gaze_direction()
                    self.ui.display_gaze_data(gaze_data)
                    self.behavior_analyzer.analyze_gaze_pattern(gaze_data, gaze_info)
                    self.logger.log_gaze_data(gaze_data, gaze_info)

                    if self.behavior_analyzer.detect_cheating():
                        self.ui.show_alert()
                        report = self.behavior_analyzer.generate_report()
                        self.logger.log_behavior(report)
                        self.ui.display_report(report)

                time.sleep(self.sleep_interval)

                key = cv2.waitKey(1)
                if key == ord('x') or key == ord('ч'):
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    self.logger.gaze_logs.append(f"{timestamp}: Отмечена попытка списывания (по нажатию X)")
                    self.logger.behavior_logs.append({
                        "timestamp": timestamp,
                        "data": {
                            "event_type": "manual_cheating_mark",
                            "message": "Пользователь отметил попытку списывания по нажатию X"
                        }
                    })
                    self.logger.save_logs_to_file()
                    print("Попытка списывания отмечена в логах")

                if key == ord('c') or key == ord('с'):
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Остановка приложения."""
        self.gaze_tracker.release_camera()
        self.logger.save_logs_to_file()
        print("Общий лог активности успешно сохранен!")
        print("Лог подозрительной активности успешно сохранен!")
        print("Приложение остановлено.")


if __name__ == "__main__":
    app = MainApp()
    app.run()