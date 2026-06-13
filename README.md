# Proctoring-API
# Authors and copyright
Supervisor and main author V. A. Parhomenko, co-author I. A. Epishin. Сopyright © V. A. Parhomenko, I. A. Epishin.

# General Description
Программа представляет собой экспериментальный стенд и API. Это программа прокторинга, которая отслеживает данные с веб-камеры, а именно - направление взляда и положенияя головы. В программе имеется аналитический алгоритм на основе этих 2 признаков, а также финальное решение принимает обученная на наших данных модель XGBoost. В данном репозитории представлены 4 версии программы, а также апи версия и тестирование программы. Также присутствуют некоторые дополнительные файлы, по типу диаграмм классов, результатов тестирований и тд.

Актуальная версия экспериментального стенда находится по пути GazeTracking-master/main_with_head_tracking_model.
Актуальная версия api находится по пути GazeTracking-master/proctoring_api.
Также в директории находится обученная модель, это файлы GazeTracking-master/best_model.pkl, GazeTracking-master/feature_names.pkl и GazeTracking-master/preprocessor.pkl.
Тесты программы находятся в директории GazeTracking-master/tests.
conftest - конфигурация тестов, test_behavior_analyzer, test_data_logger, test_gaze_tracker, test_gui_interface, test_head_pose_tracker, test_model_referee, test_session_recorder - модульные тесты.
test_load - нагрузочные тесты системы
test_proctoring_api - нагрузочные тесты API

# Warranty
The contributors provide no warranty for the use of this software. Use it at your own risk.

# License
This project is open for use in educational purposes and is licensed under the MIT License.
