"""Завантаження та фільтрація даних про повітряні тривоги.

Модуль читає сирий CSV з тривогами (data/raw/alerts.csv) і дає змогу
вибрати дані для одного з двох регіонів: м. Київ або Нікопольська громада.

ПРИМІТКА щодо джерела даних
----------------------------
У файлі data/raw/alerts.csv ПРИСУТНІЙ рядок заголовка
("oblast,raion,hromada,level,started_at,finished_at,source"), хоча
заздалегідь очікувалось, що заголовка не буде. Це перевірено напряму
(перший рядок файлу = назви колонок, далі йдуть дані з 2022 по 2026 рік).
Тому читання виконується з header=0, а список COLUMNS нижче лише
документує й перевіряє очікувану структуру.

Правила відбору регіонів
-------------------------
- м. Київ: рядки, де oblast == "Kyiv City" (окрема область-сутність,
  не плутати з "Kyivska oblast", яка є сусідньою областю).
- Нікопольська громада: рядки, де oblast == "Dnipropetrovska oblast"
  і raion == "Nikopolskyi raion", і додатково:
    * level == "raion" (тривога на весь район покриває й громаду), АБО
    * hromada == "m. Nikopol ta Nikopolska terytorialna hromada".
  Сусідні громади того ж району (напр. "m. Marhanets ta Marhanetska
  terytorialna hromada") НЕ включаються.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "raw" / "alerts.csv"

COLUMNS = ["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source"]

TIMEZONE = "Europe/Kyiv"

NIKOPOL_HROMADA_NAME = "m. Nikopol ta Nikopolska terytorialna hromada"


def _read_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Прочитати сирий CSV з тривогами як рядки (без парсингу дат).

    Файл фактично містить рядок заголовка, що збігається з COLUMNS,
    тож читаємо header=0 і лише перевіряємо відповідність назв колонок.

    Параметри
    ---------
    path : Path
        Шлях до CSV-файлу.

    Повертає
    --------
    pd.DataFrame
        Сирі дані, усі колонки типу str/object.
    """
    df = pd.read_csv(path, header=0, dtype=str)
    if list(df.columns) != COLUMNS:
        warnings.warn(
            f"Очікувані колонки {COLUMNS}, отримано {list(df.columns)}. "
            "Перевірте структуру файлу data/raw/alerts.csv."
        )
    return df


def _filter_kyiv(df: pd.DataFrame) -> pd.Series:
    """Булева маска рядків для м. Київ.

    м. Київ визначається як oblast == "Kyiv City" (окрема сутність,
    відмінна від "Kyivska oblast").
    """
    return df["oblast"] == "Kyiv City"


def _filter_nikopol(df: pd.DataFrame) -> pd.Series:
    """Булева маска рядків для Нікопольської громади.

    Беремо рядки в межах Nikopolskyi raion (Dnipropetrovska oblast),
    які або є тривогою рівня "raion" (покриває всю громаду), або є
    тривогою рівня "hromada" саме для громади Нікополя. Інші громади
    цього ж району (напр. Marhanets) виключаються свідомо.
    """
    in_raion = (df["oblast"] == "Dnipropetrovska oblast") & (df["raion"] == "Nikopolskyi raion")
    is_raion_level = df["level"] == "raion"
    is_nikopol_hromada = df["hromada"] == NIKOPOL_HROMADA_NAME
    return in_raion & (is_raion_level | is_nikopol_hromada)


REGIONS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "kyiv": _filter_kyiv,
    "nikopol": _filter_nikopol,
}


def load_region(region_key: str, path: Path = DATA_PATH) -> pd.DataFrame:
    """Завантажити та підготувати тривоги для заданого регіону.

    Параметри
    ---------
    region_key : str
        Ключ регіону з REGIONS ("kyiv" або "nikopol").
    path : Path
        Шлях до сирого CSV-файлу.

    Повертає
    --------
    pd.DataFrame
        Колонки [started_at, finished_at] типу datetime з часовим поясом
        "Europe/Kyiv", відсортовано за started_at, індекс перенумеровано.
        Якщо фільтр не знайшов жодного рядка, повертається порожній
        DataFrame з тими ж колонками і виводиться попередження.

    Викидає
    -------
    ValueError
        Якщо region_key не знайдений серед REGIONS.
    """
    if region_key not in REGIONS:
        raise ValueError(f"Невідомий регіон: {region_key!r}. Доступні: {list(REGIONS)}")

    df = _read_raw(path)
    mask = REGIONS[region_key](df)
    subset = df.loc[mask].copy()

    if subset.empty:
        warnings.warn(
            f"Фільтр для регіону '{region_key}' дав 0 рядків. "
            "Перевірте назви oblast/raion/hromada в даних."
        )
        return pd.DataFrame(columns=["started_at", "finished_at"])

    subset["started_at"] = pd.to_datetime(subset["started_at"], utc=True).dt.tz_convert(TIMEZONE)
    subset["finished_at"] = pd.to_datetime(subset["finished_at"], utc=True).dt.tz_convert(TIMEZONE)

    result = (
        subset[["started_at", "finished_at"]]
        .sort_values("started_at")
        .reset_index(drop=True)
    )
    return result


if __name__ == "__main__":
    raw = _read_raw()

    print("Унікальні значення колонки 'oblast':")
    for value in sorted(raw["oblast"].dropna().unique()):
        print(f"  - {value}")
    print()

    for key in REGIONS:
        region_df = load_region(key)
        n = len(region_df)
        print(f"Регіон: {key}")
        print(f"  Кількість тривог: {n}")
        if n == 0:
            print("  УВАГА: 0 рядків! Приклади реальних значень oblast/raion/hromada/level:")
            print(raw[["oblast", "raion", "hromada", "level"]].drop_duplicates().head(10).to_string(index=False))
        else:
            print(f"  Найраніша дата: {region_df['started_at'].min()}")
            print(f"  Найпізніша дата: {region_df['started_at'].max()}")
        print()
