"""Побудова безперервної погодинної сітки повітряних тривог.

Вхід: data/load.py::load_region(region_key) -> DataFrame[started_at, finished_at]
      (інтервали тривог у часовому поясі Europe/Kyiv).

Вихід: parquet-файл data/processed/{region_key}_hourly.parquet з колонками
       [alert_active, alert_minutes, n_starts], індекс — початок години
       (tz-aware, Europe/Kyiv), без дірок.

Кроки обробки (build_hourly_grid):
1. Завантажити інтервали тривог для регіону.
2. Злити перекривні/суміжні інтервали (щоб уникнути подвійного підрахунку
   тривалості, коли, напр., тривога рівня "raion" і тривога рівня "hromada"
   для Нікополя перекриваються в часі).
3. Побудувати безперервну погодинну сітку від початку години найранішої
   тривоги до початку години найпізнішої тривоги (за started_at).
4. Для кожної години рахувати alert_active, alert_minutes, n_starts.
5. Зберегти результат у parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.load import load_region

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TIMEZONE = "Europe/Kyiv"


def merge_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Злити перекривні/суміжні інтервали тривог в один.

    Алгоритм: відсортувати за started_at, потім послідовно проходити
    інтервали — якщо наступний починається раніше або в момент завершення
    поточного (started_at_next <= finished_at_current), об'єднати їх
    (finished = max з двох), інакше почати новий інтервал.

    Параметри
    ---------
    df : pd.DataFrame
        Колонки [started_at, finished_at], tz-aware datetime.

    Повертає
    --------
    pd.DataFrame
        Злиті інтервали, колонки [started_at, finished_at], відсортовані,
        без перекриттів.
    """
    if df.empty:
        return df.copy()

    sorted_df = df.sort_values("started_at").reset_index(drop=True)

    merged_starts = []
    merged_finishes = []

    cur_start = sorted_df.loc[0, "started_at"]
    cur_finish = sorted_df.loc[0, "finished_at"]

    for started_at, finished_at in zip(
        sorted_df["started_at"].iloc[1:], sorted_df["finished_at"].iloc[1:]
    ):
        if started_at <= cur_finish:
            cur_finish = max(cur_finish, finished_at)
        else:
            merged_starts.append(cur_start)
            merged_finishes.append(cur_finish)
            cur_start, cur_finish = started_at, finished_at

    merged_starts.append(cur_start)
    merged_finishes.append(cur_finish)

    return pd.DataFrame({"started_at": merged_starts, "finished_at": merged_finishes})


def _build_hour_grid(min_started_at: pd.Timestamp, max_started_at: pd.Timestamp) -> pd.DatetimeIndex:
    """Побудувати безперервну погодинну сітку (tz Europe/Kyiv).

    Сітка йде від початку години min_started_at до початку години
    max_started_at включно, з кроком freq="h". Параметри nonexistent/
    ambiguous обробляють переходи літнього/зимового часу так, щоб крок
    між сусідніми точками сітки завжди дорівнював рівно 1 годині в
    абсолютному часі (UTC), незалежно від зміни зсуву часового поясу.
    """
    start = min_started_at.floor("h")
    end = max_started_at.floor("h")
    return pd.date_range(
        start=start,
        end=end,
        freq="h",
        tz=TIMEZONE,
        nonexistent="shift_forward",
        ambiguous="infer",
    )


def _accumulate_minutes(
    grid: pd.DatetimeIndex, merged: pd.DataFrame
) -> np.ndarray:
    """Для кожної години сітки рахувати кількість хвилин, покритих тривогою.

    Усі обчислення виконуються через арифметику pd.Timestamp/pd.Timedelta
    (а не через ручне переведення в наносекунди), бо внутрішня одиниця
    зберігання datetime у pandas може відрізнятись між версіями
    (ns/us/ms) — Timedelta-арифметика коректно враховує це сама.

    Параметри
    ---------
    grid : pd.DatetimeIndex
        Погодинна сітка (tz-aware).
    merged : pd.DataFrame
        Злиті інтервали тривог, колонки [started_at, finished_at].

    Повертає
    --------
    np.ndarray
        Масив довжини len(grid) з кількістю хвилин (float, 0..60) покритих
        тривогою в кожній годині.
    """
    n_hours = len(grid)
    alert_minutes = np.zeros(n_hours, dtype=float)

    if merged.empty or n_hours == 0:
        return alert_minutes

    grid_start = grid[0]
    one_hour = pd.Timedelta(hours=1)
    one_unit = pd.Timedelta(nanoseconds=1)

    starts = merged["started_at"]
    finishes = merged["finished_at"]

    idx_start_all = ((starts - grid_start) // one_hour).to_numpy(dtype=np.int64)
    # finish належить наступній годині, якщо рівно на межі — беремо годину,
    # що містить (finish - 1 одиниця), бо інтервал напіввідкритий [start, finish).
    idx_end_all = (((finishes - one_unit) - grid_start) // one_hour).to_numpy(dtype=np.int64)

    for start, finish, idx_start, idx_end in zip(starts, finishes, idx_start_all, idx_end_all):
        idx_start_clipped = max(idx_start, 0)
        idx_end_clipped = min(idx_end, n_hours - 1)

        if idx_start_clipped > idx_end_clipped:
            continue  # інтервал повністю поза межами сітки

        for idx in range(idx_start_clipped, idx_end_clipped + 1):
            bin_start = grid_start + idx * one_hour
            bin_end = bin_start + one_hour
            overlap_start = max(start, bin_start)
            overlap_end = min(finish, bin_end)
            minutes = (overlap_end - overlap_start).total_seconds() / 60
            if minutes > 0:
                alert_minutes[idx] += minutes

    return np.clip(alert_minutes, 0, 60)


def _count_starts(grid: pd.DatetimeIndex, raw_intervals: pd.DataFrame) -> np.ndarray:
    """Рахувати, скільки (оригінальних, до злиття) тривог почалося в кожну годину.

    Лічильник будується на основі НЕзлитих інтервалів, бо злиття об'єднує
    кілька тривог в одну і втрачає інформацію про окремі початки.

    Обчислення індексу години виконується через Timedelta-арифметику
    (started_at - grid_start) // 1h, а не через ручне переведення у
    наносекунди — внутрішня одиниця зберігання datetime у pandas може
    відрізнятись між версіями (ns/us/ms), і Timedelta-арифметика коректно
    враховує це сама.
    """
    n_hours = len(grid)
    n_starts = np.zeros(n_hours, dtype=int)

    if raw_intervals.empty or n_hours == 0:
        return n_starts

    grid_start = grid[0]
    one_hour = pd.Timedelta(hours=1)

    idx = ((raw_intervals["started_at"] - grid_start) // one_hour).to_numpy(dtype=np.int64)
    valid = (idx >= 0) & (idx < n_hours)
    np.add.at(n_starts, idx[valid], 1)

    return n_starts


def build_hourly_grid(region_key: str) -> pd.DataFrame:
    """Побудувати погодинну сітку тривог для регіону і зберегти у parquet.

    Параметри
    ---------
    region_key : str
        Ключ регіону, що приймає load_region (напр. "kyiv", "nikopol").

    Повертає
    --------
    pd.DataFrame
        Індекс — початок години (tz Europe/Kyiv), безперервний, без дірок.
        Колонки:
          - alert_active (int, 0/1): 1, якщо година хоч частково покрита тривогою.
          - alert_minutes (float, 0..60): скільки хвилин години покрито тривогою.
          - n_starts (int): скільки (оригінальних) тривог почалося в цю годину.

    Викидає
    -------
    ValueError
        Якщо для регіону немає жодної тривоги (немає з чого будувати сітку).
    """
    raw = load_region(region_key)
    if raw.empty:
        raise ValueError(
            f"Для регіону '{region_key}' load_region не повернув жодного рядка — "
            "сітку побудувати неможливо."
        )

    merged = merge_intervals(raw)

    grid = _build_hour_grid(raw["started_at"].min(), raw["started_at"].max())

    alert_minutes = _accumulate_minutes(grid, merged)
    alert_active = (alert_minutes > 0).astype(int)
    n_starts = _count_starts(grid, raw)

    result = pd.DataFrame(
        {
            "alert_active": alert_active,
            "alert_minutes": alert_minutes,
            "n_starts": n_starts,
        },
        index=grid,
    )
    result.index.name = "hour"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{region_key}_hourly.parquet"
    result.to_parquet(out_path)

    return result


if __name__ == "__main__":
    for key in ("kyiv", "nikopol"):
        grid_df = build_hourly_grid(key)
        print(f"Регіон: {key}")
        print(f"  Форма таблиці: {grid_df.shape}")
        print(f"  Діапазон дат: {grid_df.index.min()} -> {grid_df.index.max()}")
        print(f"  Середнє alert_active (частка годин під тривогою): {grid_df['alert_active'].mean():.4f}")
        print("  Приклад 5 рядків:")
        print(grid_df.head(5).to_string())
        print()
