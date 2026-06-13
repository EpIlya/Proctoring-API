import cv2
from gaze_tracking import GazeTracking

url = "http://192.168.91.138:8080/video"

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Устанавливаем размер окна (ширина, высота)
window_width = 800
window_height = 600

# Создаем окно с возможностью изменения размера
cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Demo", window_width, window_height)

while True:
    # Получаем кадр с камеры
    _, frame = webcam.read()
    if frame is None:
        print("Не удалось получить кадр. Проверьте подключение.")
        break

    # Анализируем взгляд
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    
    # Добавляем текстовую информацию
    text = ""
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Отображаем координаты зрачков
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, f"Left pupil: {left_pupil}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, f"Right pupil: {right_pupil}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Показываем кадр
    cv2.imshow("Demo", frame)

    # Выход по ESC
    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()