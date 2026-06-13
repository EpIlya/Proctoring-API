import json
from datetime import datetime
from collections import Counter


def parse_behavior_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        logs = json.load(f)

    # Словарь для хранения результатов
    participants = {}

    # Временные переменные для хранения данных между итерациями
    current_participant = None
    cheating_marks = []
    cheating_attempts = []

    for log in logs:
        timestamp_str = log['timestamp']
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        data = log['data']

        # Тип 1: Начало сессии респондента
        if 'participant_number' in data:
            current_participant = data['participant_number']
            if current_participant not in participants:
                participants[current_participant] = {
                    'manual_cheating_marks': 0,
                    'detected_cheating_attempts': 0,
                    'false_positives': 0,
                    'gaze_directions': {}  # Будем хранить статистику по направлениям взгляда
                }

        # Тип 2: Ручная отметка о списывании
        elif 'event_type' in data and data['event_type'] == 'manual_cheating_mark':
            if current_participant is not None:
                participants[current_participant]['manual_cheating_marks'] += 1
                cheating_marks.append((timestamp, current_participant))

        # Тип 3: Попытка списать (обнаружена системой)
        elif 'suspicious_actions' in data and 'gaze_history' in data:
            if current_participant is not None:
                cheating_attempts.append((timestamp, current_participant))

                # Находим самое частое направление взгляда
                gaze_counter = Counter(data['gaze_history'])
                most_common_direction = gaze_counter.most_common(1)[0][0]

                # Увеличиваем счетчик для этого направления
                if most_common_direction in participants[current_participant]['gaze_directions']:
                    participants[current_participant]['gaze_directions'][most_common_direction] += 1
                else:
                    participants[current_participant]['gaze_directions'][most_common_direction] = 1

    for mark_time, participant in cheating_marks:
        # Ищем автоматические обнаружения в течение 10 секунд после ручной отметки
        detected = False
        for attempt_time, attempt_participant in cheating_attempts:
            if attempt_participant == participant:
                time_diff = (attempt_time - mark_time).total_seconds()
                if 0 <= time_diff <= 10:
                    detected = True
                    break

        if detected:
            participants[participant]['detected_cheating_attempts'] += 1

    for participant in participants:
        # Общее количество автоматических обнаружений для этого участника
        total_attempts = sum(1 for _, p in cheating_attempts if p == participant)

        participants[participant]['false_positives'] = total_attempts - participants[participant][
            'detected_cheating_attempts']

    return participants

def calculate_total_stats(stats_file):
    if isinstance(stats_file, str):
        with open(stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = stats_file

    total_stats = {
        'total_detected_attempts': 0,  # detected_cheating_attempts + false_positives
        'total_manual_marks': 0,  # manual_cheating_marks
        'total_real_attempts': 0,  # detected_cheating_attempts
        'total_false_positives': 0  # false_positives
    }

    for participant_data in data.values():
        total_stats['total_detected_attempts'] += participant_data['detected_cheating_attempts'] + participant_data[
            'false_positives']
        total_stats['total_manual_marks'] += participant_data['manual_cheating_marks']
        total_stats['total_real_attempts'] += participant_data['detected_cheating_attempts']
        total_stats['total_false_positives'] += participant_data['false_positives']

    return total_stats


def print_total_stats(total_stats):

    print("Итоговая статистика:")
    print(f"1. Всего залогировано попыток списываний: {total_stats['total_detected_attempts']}")
    print(f"2. Всего было реальных попыток списать: {total_stats['total_manual_marks']}")
    print(f"3. Всего замечены реальные попытки списать: {total_stats['total_real_attempts']}")
    print(f"4. Всего ложных срабатываний: {total_stats['total_false_positives']}")


result = parse_behavior_log('../../behavior_log2.json')
print(json.dumps(result, indent=2, ensure_ascii=False))

total_stats = calculate_total_stats('../../test_result.json')
print_total_stats(total_stats)