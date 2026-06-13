import os
import sys
import time
import uuid
import argparse
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import requests


# Конфигурация
BASE_URL     = os.environ.get("BASE_URL",     "http://localhost:8085")
LOAD_USERS   = int(os.environ.get("LOAD_USERS",   "20"))2
STRESS_MAX   = int(os.environ.get("STRESS_MAX",   "100"))
SOAK_USERS   = int(os.environ.get("SOAK_USERS",   "10"))
SOAK_MINUTES = int(os.environ.get("SOAK_MINUTES", "5"))
CONCUR_USERS = int(os.environ.get("CONCUR_USERS", "20"))

REQUEST_TIMEOUT           = 15
MAX_ACCEPTABLE_P95_MS     = 1000
MAX_ACCEPTABLE_ERROR_RATE = 0.05



# Структуры данных
@dataclass
class RequestResult:
    endpoint: str
    method:   str
    status:   int
    latency:  float
    error:    Optional[str] = None
    body:     Optional[dict] = None  # сохраняем тело для диагностики

    @property
    def success(self) -> bool:
        return self.error is None and 200 <= self.status < 500


@dataclass
class TestReport:
    name:        str
    results:     List[RequestResult] = field(default_factory=list)
    started_at:  float = field(default_factory=time.time)
    finished_at: float = 0.0

    def finish(self):
        self.finished_at = time.time()

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def errors(self) -> int:
        return self.total - self.successes

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total else 0.0

    @property
    def latencies(self) -> List[float]:
        return [r.latency for r in self.results if r.success]

    def percentile(self, p: float) -> float:
        lats = sorted(self.latencies)
        if not lats:
            return 0.0
        idx = int(len(lats) * p / 100)
        return lats[min(idx, len(lats) - 1)]

    def avg_latency(self) -> float:
        lats = self.latencies
        return statistics.mean(lats) if lats else 0.0

    def rps(self) -> float:
        return self.successes / self.duration if self.duration > 0 else 0.0

    def print_summary(self) -> bool:
        passed   = self.error_rate <= MAX_ACCEPTABLE_ERROR_RATE
        p95_ms   = self.percentile(95) * 1000
        p95_ok   = p95_ms <= MAX_ACCEPTABLE_P95_MS
        verdict  = "PASSED" if (passed and p95_ok) else "FAILED"
        print(f"\n{'='*60}")
        print(f"  {self.name}  -  {verdict}")
        print(f"{'='*60}")
        print(f"  Длительность теста : {self.duration:.1f} с")
        print(f"  Всего запросов     : {self.total}")
        print(f"  Успешных           : {self.successes}")
        print(f"  Ошибок             : {self.errors}  ({self.error_rate*100:.1f}%)")
        print(f"  RPS                : {self.rps():.1f}")
        print(f"  Latency avg        : {self.avg_latency()*1000:.0f} мс")
        print(f"  Latency p50        : {self.percentile(50)*1000:.0f} мс")
        print(f"  Latency p95        : {p95_ms:.0f} мс  (порог {MAX_ACCEPTABLE_P95_MS} мс)")
        print(f"  Latency p99        : {self.percentile(99)*1000:.0f} мс")
        if not passed:
            print(f"  Ошибок слишком много: "
                  f"{self.error_rate*100:.1f}% > {MAX_ACCEPTABLE_ERROR_RATE*100:.0f}%")
        if not p95_ok:
            print(f"  p95 превышает порог: {p95_ms:.0f} мс > {MAX_ACCEPTABLE_P95_MS} мс")

        # Показываем первые несколько ошибок для диагностики
        failed = [r for r in self.results if not r.success][:3]
        if failed:
            print("  --- Примеры ошибок ---")
            for r in failed:
                if r.error:
                    print(f"  {r.method} {r.endpoint}  =>  {r.error}")
                else:
                    detail = ""
                    if r.body:
                        detail = str(r.body).get if callable(str(r.body).get) \
                            else str(r.body)[:120]
                    print(f"  {r.method} {r.endpoint}  =>  HTTP {r.status}  {detail}")
        print()
        return passed and p95_ok


# ===========================================================================
# HTTP-утилиты
# ===========================================================================

def _req(method: str, path: str, **kwargs) -> RequestResult:
    url = BASE_URL + path
    t0  = time.perf_counter()
    try:
        resp    = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        latency = time.perf_counter() - t0
        try:
            body = resp.json()
        except Exception:
            body = None
        return RequestResult(path, method, resp.status_code, latency, body=body)
    except requests.exceptions.ConnectionError as e:
        return RequestResult(path, method, 0, time.perf_counter() - t0,
                             error=f"ConnectionError: {e}")
    except requests.exceptions.Timeout:
        return RequestResult(path, method, 0, time.perf_counter() - t0,
                             error="Timeout")
    except Exception as e:
        return RequestResult(path, method, 0, time.perf_counter() - t0,
                             error=str(e))


def _req_json(method: str, path: str,
              **kwargs) -> Tuple[RequestResult, Optional[dict]]:
    r = _req(method, path, **kwargs)
    return r, r.body


def check_health() -> bool:
    try:
        r = requests.get(BASE_URL + "/health", timeout=5)
        if r.status_code != 200:
            return False
        body = r.json()
        if not body.get("test_mode", False):
            print()
            print("  [WARN] Сервер запущен БЕЗ PROCTORING_TEST_MODE=1 !")
            print("         SessionState будет инициализировать реальную камеру")
            print("         и писать логи на диск при каждом создании сессии.")
            print("         Рекомендуется перезапустить сервер:")
            print("         Windows cmd  : set PROCTORING_TEST_MODE=1 && uvicorn proctoring_api:app --port 8000")
            print("         Windows PS   : $env:PROCTORING_TEST_MODE='1'; uvicorn proctoring_api:app --port 8000")
            print("         Linux/macOS  : PROCTORING_TEST_MODE=1 uvicorn proctoring_api:app --port 8000")
            print()
        return True
    except Exception:
        return False


def create_session(participant: str = None,
                   quiz: str = "quiz-001") -> Tuple[Optional[str], RequestResult]:
    if participant is None:
        participant = f"student_{uuid.uuid4().hex[:8]}"
    r, body = _req_json(
        "POST", "/sessions",
        json={"participant_number": participant, "quiz_id": quiz}
    )
    session_id = body.get("session_id") if body else None
    return session_id, r


# ===========================================================================
# ДИАГНОСТИЧЕСКИЙ ТЕСТ  (запустить первым при любых проблемах)
# ===========================================================================

def run_diag() -> bool:
    """
    Делает по одному запросу к каждому эндпоинту жизненного цикла
    и печатает полный HTTP-статус и тело ответа.
    Позволяет увидеть точную причину ошибок до запуска нагрузочных тестов.
    """
    print("\n" + "="*60)
    print("  ДИАГНОСТИКА  (один проход жизненного цикла)")
    print("="*60)
    ok = True

    def check(label: str, r: RequestResult, expected_status: int = 200) -> bool:
        status_ok = (r.status == expected_status) and r.error is None
        mark = "OK" if status_ok else "FAIL"
        print(f"  [{mark}]  {r.method:6s} {r.endpoint}")
        print(f"          HTTP {r.status}  latency={r.latency*1000:.0f} мс")
        if r.error:
            print(f"          Ошибка: {r.error}")
        if r.body:
            import json
            body_str = json.dumps(r.body, ensure_ascii=False, indent=2)
            # Обрезаем длинные ответы
            if len(body_str) > 300:
                body_str = body_str[:300] + "\n  ... (обрезано)"
            for line in body_str.splitlines():
                print(f"          {line}")
        print()
        return status_ok

    # /health
    r = _req("GET", "/health")
    ok &= check("health", r, 200)

    # GET /sessions (список)
    r = _req("GET", "/sessions")
    ok &= check("list sessions", r, 200)

    # POST /sessions
    participant = f"diag_user_{uuid.uuid4().hex[:6]}"
    r, body = _req_json("POST", "/sessions",
                        json={"participant_number": participant,
                              "quiz_id": "diag-quiz"})
    ok &= check("create session", r, 201)
    session_id = body.get("session_id") if body else None

    if not session_id:
        print("  [FAIL] Не удалось получить session_id - дальнейшая диагностика невозможна")
        print("         Убедитесь, что сервер запущен с PROCTORING_TEST_MODE=1")
        return False

    # GET /sessions/{id}
    r = _req("GET", f"/sessions/{session_id}")
    ok &= check("get session status", r, 200)

    # GET /sessions/{id}/calibrate/status
    r = _req("GET", f"/sessions/{session_id}/calibrate/status")
    ok &= check("calibrate status", r, 200)

    # GET /sessions/{id}/report
    r = _req("GET", f"/sessions/{session_id}/report")
    ok &= check("get report", r, 200)

    # GET /sessions/{id}/verdict
    r = _req("GET", f"/sessions/{session_id}/verdict")
    ok &= check("get verdict", r, 200)

    # GET /sessions/{id}/logs
    r = _req("GET", f"/sessions/{session_id}/logs")
    ok &= check("get logs", r, 200)

    # POST .../cheating-mark
    r = _req("POST", f"/sessions/{session_id}/cheating-mark",
             json={"comment": "diag test mark"})
    ok &= check("cheating mark", r, 200)

    # POST .../stop
    r = _req("POST", f"/sessions/{session_id}/stop")
    ok &= check("stop session", r, 200)

    # DELETE /sessions/{id}
    r = _req("DELETE", f"/sessions/{session_id}")
    ok &= check("delete session", r, 200)

    # GET несуществующей сессии - должен вернуть 404
    r = _req("GET", f"/sessions/nonexistent-id-000")
    ok &= check("404 for unknown id", r, 404)

    # PATCH /config
    r = _req("PATCH", "/config",
             json={"max_suspicious_actions": 3, "debug": False})
    ok &= check("patch config", r, 200)

    verdict = "Все эндпоинты работают корректно" if ok \
        else "Часть эндпоинтов недоступна или возвращает ошибки"
    print(f"  Результат: {verdict}")
    return ok


# ===========================================================================
# 1. НАГРУЗОЧНЫЙ ТЕСТ
# ===========================================================================

def simulate_user_load(user_idx: int, polls: int = 5) -> List[RequestResult]:
    """Один виртуальный пользователь: создать -> polling -> отчёт -> удалить."""
    results = []
    session_id, res = create_session(f"load_user_{user_idx}")
    results.append(res)
    if session_id is None:
        return results
    for _ in range(polls):
        results.append(_req("GET", "/sessions"))
        results.append(_req("GET", f"/sessions/{session_id}"))
    results.append(_req("GET",    f"/sessions/{session_id}/report"))
    results.append(_req("POST",   f"/sessions/{session_id}/stop"))
    results.append(_req("DELETE", f"/sessions/{session_id}"))
    return results


def run_load_test(n_users: int = LOAD_USERS) -> bool:
    """
    Нагрузочный тест: n_users параллельных пользователей,
    рамп-ап по 5 пользователей каждые 2 секунды.
    """
    report = TestReport(name=f"Нагрузочный тест ({n_users} пользователей)")
    print(f"\n[LOAD] Запуск: {n_users} пользователей, 5 человек/2 с")
    lock = threading.Lock()

    def worker(idx):
        res = simulate_user_load(idx, polls=5)
        with lock:
            report.results.extend(res)

    with ThreadPoolExecutor(max_workers=n_users) as executor:
        for i in range(n_users):
            executor.submit(worker, i)
            if (i + 1) % 5 == 0:
                print(f"  [LOAD] Запущено {i+1}/{n_users} пользователей...")
                time.sleep(2.0)

    report.finish()
    return report.print_summary()


# ===========================================================================
# 2. ПИКОВЫЙ ТЕСТ
# ===========================================================================

def run_spike_test() -> bool:
    """
    Пиковый тест: имитирует момент начала экзамена.
    Фаза 1 - базовая (5 пользователей / 10 с)
    Фаза 2 - пик (50 пользователей одновременно)
    Фаза 3 - спад (5 пользователей / 10 с)
    """
    report = TestReport(name="Пиковый тест (spike)")
    lock   = threading.Lock()

    def worker(idx, label):
        session_id, res = create_session(f"spike_{label}_{idx}")
        with lock:
            report.results.append(res)
        if session_id:
            with lock:
                report.results.append(_req("GET", f"/sessions/{session_id}"))
            _req("POST",   f"/sessions/{session_id}/stop")
            _req("DELETE", f"/sessions/{session_id}")

    print("\n[SPIKE] Фаза 1: базовая нагрузка (5 пользователей, 10 с)")
    with ThreadPoolExecutor(max_workers=5) as ex:
        for i in range(5):
            ex.submit(worker, i, "base1")
            time.sleep(2)

    print("[SPIKE] Фаза 2: пик - 50 пользователей одновременно")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=50) as ex:
        for f in as_completed([ex.submit(worker, i, "peak") for i in range(50)]):
            pass
    print(f"  Пик обработан за {time.time()-t0:.1f} с")

    print("[SPIKE] Фаза 3: возврат к базовой нагрузке (5 пользователей, 10 с)")
    with ThreadPoolExecutor(max_workers=5) as ex:
        for i in range(5):
            ex.submit(worker, i, "base2")
            time.sleep(2)

    report.finish()
    return report.print_summary()


# ===========================================================================
# 3. СТРЕСС-ТЕСТ
# ===========================================================================

def run_stress_test(max_users: int = STRESS_MAX, step: int = 10,
                    step_duration: float = 15.0) -> bool:
    """
    Нагрузка растёт шагами по step пользователей каждые step_duration с
    до max_users или точки отказа (error_rate > 10 % или p95 > 5 с).
    """
    report       = TestReport(name=f"Стресс-тест (до {max_users} пользователей)")
    lock         = threading.Lock()
    breakdown_at = None

    def worker(idx, step_num):
        session_id, res = create_session(f"stress_s{step_num}_u{idx}")
        with lock:
            report.results.append(res)
        if session_id:
            for _ in range(3):
                r = _req("GET", f"/sessions/{session_id}")
                with lock:
                    report.results.append(r)
                time.sleep(0.1)
            _req("POST",   f"/sessions/{session_id}/stop")
            _req("DELETE", f"/sessions/{session_id}")

    print(f"\n[STRESS] Шаги по {step} пользователей, шаг {step_duration} с, "
          f"максимум {max_users}")
    current  = 0
    step_num = 0

    while current < max_users:
        current   = min(current + step, max_users)
        step_num += 1
        before    = len(report.results)
        print(f"  [STRESS] Шаг {step_num}: {current} параллельных пользователей")
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=current) as ex:
            for f in as_completed([ex.submit(worker, i, step_num)
                                   for i in range(current)]):
                pass

        step_dur  = time.time() - t0
        step_res  = report.results[before:]
        step_err  = sum(1 for r in step_res if not r.success)
        step_tot  = len(step_res)
        err_rate  = step_err / step_tot if step_tot else 0
        lats      = sorted(r.latency for r in step_res if r.success)
        p95       = lats[int(len(lats) * 0.95)] if lats else 0

        print(f"    {step_tot} запросов, {step_err} ошибок "
              f"({err_rate*100:.1f}%), p95={p95*1000:.0f} мс, "
              f"время шага={step_dur:.1f} с")

        if err_rate > 0.10 or p95 > 5.0:
            breakdown_at = current
            print(f"  Точка отказа обнаружена при {current} пользователях!")
            break

        remaining = step_duration - step_dur
        if remaining > 0 and current < max_users:
            time.sleep(remaining)

    report.finish()
    ok = report.print_summary()
    if breakdown_at:
        print(f"  Точка отказа: {breakdown_at} параллельных пользователей")
    else:
        print(f"  Сервис выдержал максимальную нагрузку {max_users} пользователей")
    return ok


# ===========================================================================
# 4. SOAK-ТЕСТ
# ===========================================================================

def run_soak_test(n_users: int = SOAK_USERS,
                  duration_min: float = SOAK_MINUTES) -> bool:
    """
    n_users параллельных пользователей работают непрерывно duration_min минут.
    Отслеживается деградация latency: первые vs последние 20 % успешных запросов.
    """
    duration_sec      = duration_min * 60
    report            = TestReport(
        name=f"Soak-тест ({n_users} пользователей, {duration_min} мин)"
    )
    stop_event        = threading.Event()
    lock              = threading.Lock()
    iteration_counter = [0]

    def user_loop(user_idx):
        while not stop_event.is_set():
            sid, res = create_session(f"soak_u{user_idx}_i{iteration_counter[0]}")
            with lock:
                report.results.append(res)
                iteration_counter[0] += 1
            if sid:
                for _ in range(3):
                    if stop_event.is_set():
                        break
                    with lock:
                        report.results.append(
                            _req("GET", f"/sessions/{sid}")
                        )
                    time.sleep(0.05)
                if not stop_event.is_set():
                    with lock:
                        report.results.append(
                            _req("POST", f"/sessions/{sid}/stop")
                        )
                        report.results.append(
                            _req("GET",  f"/sessions/{sid}/report")
                        )
                _req("DELETE", f"/sessions/{sid}")
            time.sleep(0.2)

    print(f"\n[SOAK] Запуск: {n_users} пользователей на {duration_min} минут")
    threads = [threading.Thread(target=user_loop, args=(i,), daemon=True)
               for i in range(n_users)]
    for t in threads:
        t.start()

    start = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start
        if elapsed >= duration_sec:
            stop_event.set()
            break
        remaining = duration_sec - elapsed
        with lock:
            n    = len(report.results)
            errs = sum(1 for r in report.results if not r.success)
        print(f"  [SOAK] {elapsed/60:.1f}/{duration_min:.0f} мин | "
              f"запросов: {n} | ошибок: {errs} | осталось: {remaining:.0f} с")
        time.sleep(30)

    for t in threads:
        t.join(timeout=5)
    report.finish()

    ok_res = [r for r in report.results if r.success]
    if len(ok_res) >= 10:
        cut       = max(1, len(ok_res) // 5)
        early_avg = statistics.mean(r.latency for r in ok_res[:cut])  * 1000
        late_avg  = statistics.mean(r.latency for r in ok_res[-cut:]) * 1000
        deg       = ((late_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0
        print(f"  Деградация latency: начало {early_avg:.0f} мс -> "
              f"конец {late_avg:.0f} мс ({deg:+.1f}%)")
        if deg > 50:
            print("  Значительная деградация! Возможна утечка памяти.")

    return report.print_summary()


# ===========================================================================
# 6. ТЕСТ ПАРАЛЛЕЛЬНЫХ СЕССИЙ
# ===========================================================================

def run_concurrency_test(n_users: int = CONCUR_USERS) -> bool:
    """
    n_users сессий создаются одновременно (без рамп-апа).
    Проверяет целостность данных: ответ /sessions/{id} должен вернуть
    participant_number именно того участника, который создал сессию.
    """
    report           = TestReport(
        name=f"Тест параллельных сессий ({n_users} одновременно)"
    )
    lock             = threading.Lock()
    integrity_errors = []

    def worker(user_idx) -> List[RequestResult]:
        participant = f"concur_user_{user_idx}_{uuid.uuid4().hex[:6]}"
        sid, res    = create_session(participant, quiz="concur-quiz")
        results     = [res]
        if sid is None:
            return results

        r, body = _req_json("GET", f"/sessions/{sid}")
        results.append(r)
        if body and body.get("participant_number", "") != participant:
            with lock:
                integrity_errors.append({
                    "session_id": sid,
                    "expected":   participant,
                    "got":        body.get("participant_number"),
                    "phase":      "first_read",
                })

        for _ in range(5):
            r, b = _req_json("GET", f"/sessions/{sid}")
            results.append(r)
            if b and b.get("participant_number", "") != participant:
                with lock:
                    integrity_errors.append({
                        "session_id": sid,
                        "expected":   participant,
                        "got":        b.get("participant_number"),
                        "phase":      "polling",
                    })
            time.sleep(0.05)

        r, rep = _req_json("GET", f"/sessions/{sid}/report")
        results.append(r)
        if rep and rep.get("participant_number", "") != participant:
            with lock:
                integrity_errors.append({
                    "session_id": sid,
                    "expected":   participant,
                    "got":        rep.get("participant_number"),
                    "phase":      "report",
                })

        results.append(_req("POST",   f"/sessions/{sid}/stop"))
        results.append(_req("DELETE", f"/sessions/{sid}"))
        return results

    print(f"\n[CONCUR] {n_users} сессий стартуют одновременно (без рамп-апа)")
    with ThreadPoolExecutor(max_workers=n_users) as executor:
        futures = [executor.submit(worker, i) for i in range(n_users)]
        for f in as_completed(futures):
            with lock:
                report.results.extend(f.result())

    report.finish()
    ok = report.print_summary()

    if integrity_errors:
        print(f"  Обнаружено {len(integrity_errors)} нарушений целостности данных!")
        for err in integrity_errors[:5]:
            print(f"     session={err['session_id']} | "
                  f"ожидалось='{err['expected']}' | "
                  f"получено='{err.get('got')}'  [{err.get('phase','-')}]")
        ok = False
    else:
        print("  Нарушений целостности данных не обнаружено")

    return ok


# ===========================================================================
# ИТОГ
# ===========================================================================

def print_final_summary(results: Dict[str, bool]):
    print("\n" + "="*60)
    print("  ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*60)
    all_passed = True
    for name, passed in results.items():
        mark = "PASSED" if passed else "FAILED"
        print(f"  [{mark}]  {name}")
        if not passed:
            all_passed = False
    print("="*60)
    print("  Все тесты пройдены" if all_passed
          else "  Часть тестов не пройдена - см. детали выше")
    print()


# ===========================================================================
# ТОЧКА ВХОДА
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Нагрузочное тестирование Proctoring API"
    )
    parser.add_argument(
        "--test",
        choices=["diag", "load", "spike", "stress", "soak", "concur", "all"],
        default="all",
        help="Какой тест запустить. Рекомендуется начать с --test diag (default: all)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Proctoring API - тестирование нагрузки")
    print(f"  Сервер : {BASE_URL}")
    print(f"  Время  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    print("\n[CHECK] Проверка доступности сервиса...")
    if not check_health():
        print(f"[ERROR] Сервис недоступен по адресу {BASE_URL}")
        print("        Убедитесь, что сервер запущен.")
        sys.exit(1)
    print("[CHECK] Сервис доступен\n")

    test    = args.test
    results = {}

    if test == "diag":
        results["Диагностика"] = run_diag()
    else:
        if test in ("load",   "all"):
            results["Нагрузочный тест"]         = run_load_test(LOAD_USERS)
        if test in ("spike",  "all"):
            results["Пиковый тест"]             = run_spike_test()
        if test in ("stress", "all"):
            results["Стресс-тест"]              = run_stress_test(STRESS_MAX)
        if test in ("soak",   "all"):
            results["Soak-тест"]                = run_soak_test(SOAK_USERS, SOAK_MINUTES)
        if test in ("concur", "all"):
            results["Тест параллельных сессий"] = run_concurrency_test(CONCUR_USERS)

    print_final_summary(results)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
