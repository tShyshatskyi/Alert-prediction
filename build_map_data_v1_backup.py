"""
build_map_data.py
Зведений датасет для майбутньої веб-мапи тривог: для КОЖНОЇ з 25 областей
рахує базові метрики, бектест baseline-моделей і 30-денний прогноз, і
зберігає все в один reports/map_data.json.

Перевикористовує існуючий пайплайн без дублювання логіки:
  - data.load.REGIONS / load_region          (фільтрація сирих даних)
  - preprocess.build_grid.build_hourly_grid   (погодинна сітка)
  - backtest.engine.run_backtest              (бектест 4 baseline-моделей)
  - models.forecast.forecast_region           (SARIMA-прогноз + погодинний профіль)

Кожна область реєструється в data.load.REGIONS як oblast == <назва
області>, тобто покриває ВСІ рівні тривог (oblast/raion/hromada) у межах
області — на відміну від вузьких фільтрів kyiv/nikopol у data/load.py.

Запуск: python build_map_data.py (з кореня проєкту).
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data.load as L  # noqa: E402
from preprocess.build_grid import build_hourly_grid  # noqa: E402
from backtest.engine import run_backtest  # noqa: E402
from models.forecast import forecast_region  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
OUTPUT_PATH = REPORTS_DIR / "map_data.json"

OBLASTS = [
    "Cherkaska oblast", "Chernihivska oblast", "Chernivetska oblast", "Dnipropetrovska oblast",
    "Donetska oblast", "Ivano-Frankivska oblast", "Kharkivska oblast", "Khersonska oblast",
    "Khmelnytska oblast", "Kirovohradska oblast", "Kyiv City", "Kyivska oblast", "Luhanska oblast",
    "Lvivska oblast", "Mykolaivska oblast", "Odeska oblast", "Poltavska oblast", "Rivnenska oblast",
    "Sumska oblast", "Ternopilska oblast", "Vinnytska oblast", "Volynska oblast", "Zakarpatska oblast",
    "Zaporizka oblast", "Zhytomyrska oblast",
]

FORECAST_HORIZON = 30
RECENT_WINDOW_DAYS = 14


# ---------------------------------------------------------------------------
# Реєстрація областей як регіонів у data.load.REGIONS
# ---------------------------------------------------------------------------

def _register_oblasts() -> None:
    """Додає в L.REGIONS по одному фільтру на кожну область з OBLASTS:
    oblast == <назва>, що покриває всі рівні тривог (oblast/raion/hromada)
    усередині області. Замикання лямбди коректно фіксує назву через
    значення параметра за замовчуванням (n=name), а не через посилання
    на змінну циклу."""
    for name in OBLASTS:
        L.REGIONS[name] = (lambda df, n=name: df["oblast"] == n)


# ---------------------------------------------------------------------------
# Допоміжне: безпечна серіалізація float (NaN/inf -> None для валідного JSON)
# ---------------------------------------------------------------------------

def _safe_float(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


# ---------------------------------------------------------------------------
# Метрики по одній області
# ---------------------------------------------------------------------------

def compute_region_metrics(name: str) -> dict:
    grid = build_hourly_grid(name)

    base_rate = _safe_float(grid["alert_active"].mean())

    cutoff = grid.index.max() - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    recent = grid.loc[grid.index > cutoff, "alert_active"]
    if len(recent) == 0:
        recent = grid["alert_active"]
    recent_rate = _safe_float(recent.mean())

    hourly_profile_series = grid.groupby(grid.index.hour)["alert_active"].mean().reindex(range(24))
    hourly_profile = [_safe_float(v) for v in hourly_profile_series.tolist()]

    # --- Бектест 4 baseline-моделей ---
    bt_summary, _ = run_backtest(grid)
    backtest = {
        model: {
            "brier": _safe_float(row["brier"]),
            "roc_auc": _safe_float(row["roc_auc"]),
        }
        for model, row in bt_summary.iterrows()
    }
    brier_only = bt_summary["brier"].dropna()
    if brier_only.empty:
        raise RuntimeError("Жодна модель не дала валідного Brier у бектесті")
    best_model = brier_only.idxmin()
    best_brier = _safe_float(bt_summary.loc[best_model, "brier"])
    best_auc = _safe_float(bt_summary.loc[best_model, "roc_auc"])

    # --- Прогноз на 30 днів ---
    fc = forecast_region(name, horizons=(FORECAST_HORIZON,))[FORECAST_HORIZON]
    point = fc["sarima_point"]
    low = fc["sarima_low"]
    high = fc["sarima_high"]
    hourly_prob = fc["hourly_prob"]

    forecast_30d = {
        "dates": [d.strftime("%Y-%m-%d") for d in point.index],
        "point": [_safe_float(v) for v in point.tolist()],
        "low": [_safe_float(v) for v in low.tolist()],
        "high": [_safe_float(v) for v in high.tolist()],
        "model_name": fc["model_name"],
    }
    forecast_avg_prob = _safe_float(hourly_prob.mean())
    expected_hours_30d = _safe_float(hourly_prob.sum())

    return {
        "status": "ok",
        "base_rate": base_rate,
        "recent_rate": recent_rate,
        "hourly_profile": hourly_profile,
        "backtest": backtest,
        "best_model": best_model,
        "best_brier": best_brier,
        "best_auc": best_auc,
        "forecast_30d": forecast_30d,
        "forecast_avg_prob": forecast_avg_prob,
        "expected_hours_30d": expected_hours_30d,
    }


# ---------------------------------------------------------------------------
# Прогін по всіх областях
# ---------------------------------------------------------------------------

def build_map_data(oblasts: Optional[list] = None, merge: bool = False) -> dict:
    """Прогнати pipeline по областях.

    Параметри
    ---------
    oblasts : list[str] | None
        Підмножина OBLASTS для обробки (за замовчуванням — усі 25; це
        дозволяє відновлюваний/пакетний запуск великих прогонів частинами
        без зміни поведінки звичайного повного запуску).
    merge : bool
        Якщо True і reports/map_data.json вже існує — підвантажити його й
        оновити лише записи для обраних oblasts, зберігши решту як є.
    """
    _register_oblasts()
    targets = oblasts if oblasts is not None else OBLASTS

    result: dict = {}
    if merge and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            result = json.load(f)

    n = len(targets)
    for i, name in enumerate(targets, start=1):
        t0 = time.time()
        print(f"[{i}/{n}] {name} ... ", end="", flush=True)
        try:
            result[name] = compute_region_metrics(name)
            dt = time.time() - t0
            print(f"OK ({dt:.1f}s, best={result[name]['best_model']}, "
                  f"recent_rate={result[name]['recent_rate']:.3f})")
        except Exception as e:
            dt = time.time() - t0
            print(f"ПОМИЛКА ({dt:.1f}s): {e!r}")
            traceback.print_exc(file=sys.stderr)
            result[name] = {"status": "error", "error": repr(e)}

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def print_final_summary(result: dict) -> None:
    ok = {k: v for k, v in result.items() if v.get("status") == "ok"}
    errors = {k: v for k, v in result.items() if v.get("status") == "error"}

    print("\n" + "=" * 60)
    print(f"Готово: {len(ok)}/{len(result)} областей успішно, {len(errors)} з помилкою.")
    if errors:
        print("Області з помилкою:")
        for name, v in errors.items():
            print(f"  - {name}: {v.get('error')}")

    print("\nТОП-5 областей за recent_rate (середній alert_active за останні "
          f"{RECENT_WINDOW_DAYS} днів):")
    ranked = sorted(
        ((name, v["recent_rate"]) for name, v in ok.items() if v.get("recent_rate") is not None),
        key=lambda t: t[1],
        reverse=True,
    )
    for rank, (name, rate) in enumerate(ranked[:5], start=1):
        print(f"  {rank}. {name}: {rate:.3f}")

    print(f"\nЗбережено: {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Самоперевірка: гарячі області vs західні
# ---------------------------------------------------------------------------

HOT_OBLASTS = ["Donetska oblast", "Khersonska oblast", "Zaporizka oblast", "Sumska oblast", "Kharkivska oblast"]
COLD_OBLASTS = ["Lvivska oblast", "Zakarpatska oblast", "Ternopilska oblast"]


def self_check(result: dict) -> None:
    # 1. JSON валідний (round-trip)
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded == result or set(reloaded.keys()) == set(result.keys()), (
        "reports/map_data.json не пройшов round-trip JSON-перевірку"
    )
    print("OK: reports/map_data.json валідний JSON.")

    hot_rates = {n: result[n]["recent_rate"] for n in HOT_OBLASTS if result.get(n, {}).get("status") == "ok"}
    cold_rates = {n: result[n]["recent_rate"] for n in COLD_OBLASTS if result.get(n, {}).get("status") == "ok"}

    print(f"\nrecent_rate гарячих областей: { {k: round(v, 3) for k, v in hot_rates.items()} }")
    print(f"recent_rate західних областей: { {k: round(v, 3) for k, v in cold_rates.items()} }")

    if hot_rates and cold_rates:
        if min(hot_rates.values()) > max(cold_rates.values()):
            print("OK: гарячі області мають вищий recent_rate, ніж західні, як і очікується.")
        else:
            print("[WARN] Очікувана різниця гарячих/західних областей не підтвердилась повністю — перевір дані.")
    else:
        print("[WARN] Недостатньо успішних областей для порівняння гарячих/західних.")


def _parse_args(argv):
    """Необов'язкові аргументи для пакетного/відновлюваного запуску
    (зручно для запуску великого прогону частинами, напр. у CI з
    обмеженням за часом). Без аргументів — звичайний повний прогін по
    всіх 25 областях, як описано в задачі."""
    start, end = 0, len(OBLASTS)
    merge = False
    i = 0
    while i < len(argv):
        if argv[i] == "--start":
            start = int(argv[i + 1]); i += 2
        elif argv[i] == "--end":
            end = int(argv[i + 1]); i += 2
        elif argv[i] == "--merge":
            merge = True; i += 1
        else:
            i += 1
    return OBLASTS[start:end], merge, (start, end)


if __name__ == "__main__":
    subset, merge_flag, (s, e) = _parse_args(sys.argv[1:])
    is_full_run = (s, e) == (0, len(OBLASTS))

    final_result = build_map_data(oblasts=subset, merge=merge_flag)

    if is_full_run:
        print_final_summary(final_result)
        self_check(final_result)
    else:
        print(f"\nПакетний прогін [{s}:{e}] завершено, збережено у {OUTPUT_PATH}")
