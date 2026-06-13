# Proctoring-API
# Authors and copyright
Supervisor and main author V. A. Parhomenko, co-author I. A. Epishin. Сopyright © V. A. Parhomenko, I. A. Epishin.

# General Description
The program consists of an experimental setup and an API. It is a proctoring program that tracks webcam data, specifically gaze direction and head position. The program includes an analytical algorithm based on these two features, and the final decision is made by an XGBoost model trained on our data. This repository contains four versions of the program, as well as an API version and a test version. Some additional files, such as class diagrams, test results, etc., are also available.

The current version of the experimental setup is located at GazeTracking-master/main_with_head_tracking_model. The current version of the API is located at GazeTracking-master/proctoring_api. The trained model is also located in this directory: GazeTracking-master/best_model.pkl, GazeTracking-master/feature_names.pkl, and GazeTracking-master/preprocessor.pkl. The program tests are located in the GazeTracking-master/tests directory. conftest - test configuration, test_behavior_analyzer, test_data_logger, test_gaze_tracker, test_gui_interface, test_head_pose_tracker, test_model_referee, test_session_recorder - unit tests, test_load - system load tests, test_proctoring_api - API load tests.

# Warranty
The contributors provide no warranty for the use of this software. Use it at your own risk.

# License
This project is open for use in educational purposes and is licensed under the MIT License.
