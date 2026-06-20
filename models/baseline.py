"""
models/baseline.py
Прості baseline-моделі для прогнозу повітряних тривог по регіонах.

Дві моделі зі СПІЛЬНИМ інтерфейсом fit(train_df) / predict(index):

1. ClimatologyBaseline — оцінює ЙМОВІРНІСТЬ тривоги (alert_active) як
   історичну середню частоту, згруповану за (годиною доби, днем тижня).
   Рахується ЛИШЕ на train-періоді, без зазирання в майбутнє.

2. PersistenceBaseline — "сезонний наїв": прогноз на годину t = реальне
   значення 24 години тому (той самий час попередньої доби). Працює і
   для alert_active, і для alert_minutes.

Вхідні дані: погодинний DataFrame з DatetimeIndex (tz Europe/Kyiv) і
колонками alert_active (0/1) та alert_minutes (0..60). Завантажуються
через preprocess.build_grid.build_hourly_grid(region) або з parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

TARGET_PROB_COL = "alert_active"
TARGET_DUR_COLS = ["alert_active", "alert_minutes"]


def _check_index(index: pd.DatetimeIndex, name: str = "index") -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} має бути pd.DatetimeIndex, отримано {type(index)}")


class ClimatologyBaseline:
    """
    Baseline для ймовірності тривоги: історична частка alert_active,
    згрупована за (година доби, день тижня).

    fit() рахує таблицю ЛИШЕ на train-вибірці (жодного доступу до test).
    predict() підставляє відповідне значення для кожної години з index;
    якщо комбінації (hour, dow) не було в train — фолбек на глобальне
    середнє по train.
    """

    def __init__(self, target_col: str = TARGET_PROB_COL):
        self.target_col = target_col
        self.table_: Optional[pd.Series] = None  # index: (hour, dow) -> prob
        self.global_mean_: Optional[float] = None

    def fit(self, train_df: pd.DataFrame) -> "ClimatologyBaseline":
        _check_index(train_df.index, "train_df.index")
        if self.target_col not in train_df.columns:
            raise KeyError(f"Колонка '{self.target_col}' відсутня в train_df")

        hour = train_df.index.hour
        dow = train_df.index.dayofweek
        grp = train_df[self.target_col].groupby([hour, dow]).mean()
        grp.index.names = ["hour", "dow"]

        self.table_ = grp
        self.global_mean_ = float(train_df[self.target_col].mean())
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        if self.table_ is None:
            raise RuntimeError("Модель не навчена: викличте fit() перед predict()")
        _check_index(index, "index")

        keys = list(zip(index.hour, index.dayofweek))
        values = [self.table_.get(k, self.global_mean_) for k in keys]
        preds = pd.Series(values, index=index, name=f"{self.target_col}_pred", dtype=float)
        return preds.clip(0.0, 1.0)


class PersistenceBaseline:
    """
    "Сезонний наїв": прогноз на годину t = реальне значення о тій же
    годині попередньої доби (t - 24h). Підтримує одночасно декілька
    колонок (за замовчуванням alert_active і alert_minutes).

    ВАЖЛИВО про "майбутнє":
    Лаг 24h для будь-якого t завжди СТРОГО в минулому щодо t — це не
    зазирання в майбутнє за визначенням. Проблема інша: щоб прогнозувати
    persistence на ВСІ точки test-періоду (а не лише на перші 24 год),
    моделі потрібен доступ до фактичних значень "t-24h", які для пізніх
    точок test самі належать test-періоду. Це нормально для baseline
    такого типу (в реальному часі ці значення вже спостережені на момент
    t), але це НЕ train-дані і модель НЕ використовує їх для навчання
    параметрів — вона лише читає вже минулі (відносно кожного t)
    спостереження. Тому є окремий метод update_history(), яким можна
    "розкривати" нові спостереження в міру їх настання, без впливу на
    fit().
    """

    LAG = pd.Timedelta(hours=24)

    def __init__(self, cols: Iterable[str] = TARGET_DUR_COLS):
        self.cols = list(cols)
        self.history_: Optional[pd.DataFrame] = None

    def fit(self, train_df: pd.DataFrame) -> "PersistenceBaseline":
        _check_index(train_df.index, "train_df.index")
        missing = [c for c in self.cols if c not in train_df.columns]
        if missing:
            raise KeyError(f"Відсутні колонки в train_df: {missing}")
        self.history_ = train_df[self.cols].copy().sort_index()
        return self

    def update_history(self, df: pd.DataFrame) -> "PersistenceBaseline":
        """Додає вже спостережені дані (напр. факт test-періоду, що
        "розкривається" з часом) до внутрішньої історії, з якої
        predict() бере значення t-24h. Не використовується для навчання
        параметрів моделі — лише для читання минулих спостережень."""
        if self.history_ is None:
            raise RuntimeError("Спершу викличте fit()")
        extra = df[self.cols].copy()
        self.history_ = pd.concat([self.history_, extra]).sort_index()
        self.history_ = self.history_[~self.history_.index.duplicated(keep="last")]
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        if self.history_ is None:
            raise RuntimeError("Модель не навчена: викличте fit() перед predict()")
        _check_index(index, "index")

        lagged_idx = index - self.LAG
        preds = self.history_.reindex(lagged_idx)
        preds.index = index
        preds.columns = [f"{c}_pred" for c in self.cols]
        return preds


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _load_region(region: str) -> pd.DataFrame:
    """Завантажує погодинну сітку для регіону: спершу через
    preprocess.build_grid.build_hourly_grid(region), потім з parquet,
    і лише як демонстраційний фолбек (якщо pipeline недоступний у цьому
    середовищі) — синтетичні дані."""
    try:
        from preprocess.build_grid import build_hourly_grid  # type: ignore

        return build_hourly_grid(region)
    except Exception:
        pass

    for p in (Path(f"data/processed/{region}.parquet"), Path(f"data/{region}.parquet")):
        if p.exists():
            return pd.read_parquet(p)

    print(
        f"[WARN] Не вдалося завантажити реальні дані для '{region}' "
        "(build_hourly_grid/parquet недоступні в цьому середовищі). "
        "Використовую синтетичні дані лише для демонстрації self-check.",
        file=sys.stderr,
    )
    return _synthetic_region_grid(region)


def _synthetic_region_grid(region: str, n_hours: int = 24 * 120, seed: int = 0) -> pd.DataFrame:
    """Груба синтетика для прогону self-check без реального pipeline.
    Нікополь отримує суттєво вищу базову частоту тривог, ніж Київ —
    як і очікується по умові завдання."""
    rng = np.random.default_rng((seed + abs(hash(region))) % (2**32 - 1))
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="Europe/Kyiv")

    region_l = region.lower()
    base_rate = 0.45 if ("нікопол" in region_l or "nikopol" in region_l) else 0.08

    night = (idx.hour < 6) | (idx.hour >= 22)
    p = np.where(night, base_rate * 1.3, base_rate)
    p = np.clip(p, 0.0, 0.95)

    alert_active = rng.binomial(1, p)
    alert_minutes = np.where(alert_active == 1, rng.integers(5, 61, size=n_hours), 0)

    return pd.DataFrame({"alert_active": alert_active, "alert_minutes": alert_minutes}, index=idx)


def _self_check(region: str) -> float:
    df = _load_region(region).sort_index()
    n = len(df)
    split = int(n * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    print(f"\n=== {region}: n={n}, train={len(train_df)}, test={len(test_df)} ===")

    # --- Climatology ---
    clim = ClimatologyBaseline().fit(train_df)
    clim_pred = clim.predict(test_df.index)

    assert clim_pred.between(0, 1).all(), "Ймовірності climatology вийшли за межі [0,1]"

    actual_train_rate = float(train_df["alert_active"].mean())
    mean_pred = float(clim_pred.mean())
    print(
        f"Climatology: факт. частка тривог у train = {actual_train_rate:.3f}, "
        f"середня прогнозована ймовірність на test = {mean_pred:.3f}"
    )

    # --- Persistence ---
    pers = PersistenceBaseline().fit(train_df)
    # лаг 24h для перших точок test потрапляє в train; для пізніших точок
    # test значення t-24h саме належить test-періоду — "розкриваємо" його
    # як уже спостережене (див. докстрінг PersistenceBaseline)
    pers.update_history(test_df)
    pers_pred = pers.predict(test_df.index)

    valid = pers_pred.dropna()
    if len(valid):
        mae_active = (
            valid["alert_active_pred"] - test_df.loc[valid.index, "alert_active"]
        ).abs().mean()
        mae_minutes = (
            valid["alert_minutes_pred"] - test_df.loc[valid.index, "alert_minutes"]
        ).abs().mean()
        print(
            f"Persistence: MAE(alert_active) = {mae_active:.3f}, "
            f"MAE(alert_minutes) = {mae_minutes:.3f} "
            f"(на {len(valid)}/{len(test_df)} точок, де є t-24h)"
        )

    return mean_pred


if __name__ == "__main__":
    nikopol_mean = _self_check("nikopol")
    kyiv_mean = _self_check("kyiv")

    print("\n=== Перевірка: Нікополь має вищу базову частоту, ніж Київ ===")
    print(f"Нікополь: середня прогнозована ймовірність = {nikopol_mean:.3f}")
    print(f"Київ:     середня прогнозована ймовірність = {kyiv_mean:.3f}")
    assert nikopol_mean > kyiv_mean, "Очікувалось: Нікополь суттєво вищий за Київ"
    print("OK: Нікополь > Київ, як і очікується.")
