"""
models/classical.py
Класичний підхід до прогнозування ДОБОВОГО ряду тривог (доповнює
погодинний ML з models/baseline.py, models/...): SARIMA /
ExponentialSmoothing проти SeasonalNaiveDaily-еталону.

Ряд: "годин тривог на добу" (0..24), отриманий агрегацією погодинної
сітки build_hourly_grid(region)["alert_active"] по днях.

Запускається і як `python models/classical.py`, і як
`python -m models.classical` (з кореня проєкту Alert-prediction/).
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # без GUI-бекенду — лише збереження у файл
import matplotlib.pyplot as plt  # noqa: E402

from preprocess.build_grid import build_hourly_grid  # noqa: E402

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


REPORTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "reports"

TEST_DAYS = 60
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 0, 1, 7)
SEASONAL_PERIODS = 7


# ---------------------------------------------------------------------------
# Дані
# ---------------------------------------------------------------------------

def build_daily_series(region: str) -> pd.Series:
    """Погодинна сітка -> добовий ряд "годин тривог на добу" (0..24).

    tz знімаємо (Europe/Kyiv -> naive) після агрегації по добах: для
    денного ряду конкретний offset вже не несе інформації, а statsmodels
    стабільніше працює з naive DatetimeIndex і freq="D".
    """
    hourly = build_hourly_grid(region)
    daily = hourly["alert_active"].resample("D").sum()
    daily.index = daily.index.tz_localize(None)
    daily = daily.asfreq("D")
    daily.name = "alert_hours"
    return daily


def chronological_split(series: pd.Series, test_days: int = TEST_DAYS) -> Tuple[pd.Series, pd.Series]:
    """Хронологічний спліт без перемішування: останні test_days днів -> test."""
    if len(series) <= test_days:
        raise ValueError(f"Замало даних ({len(series)} днів) для test_days={test_days}")
    train = series.iloc[:-test_days]
    test = series.iloc[-test_days:]
    assert train.index.max() < test.index.min(), "Витік майбутнього: train перетинається з test"
    return train, test


# ---------------------------------------------------------------------------
# Еталонна модель: SeasonalNaiveDaily
# ---------------------------------------------------------------------------

class SeasonalNaiveDaily:
    """Еталон: прогноз на день t = факт 7 днів тому.

    Для h-кроку наперед (h > 7, наприклад весь test-горизонт) тиражує
    останній спостережений тижневий паттерн з train — і НІЧОГО з test
    не використовує, тож зазирання в майбутнє неможливе за конструкцією.
    """

    PERIOD = 7

    def __init__(self):
        self.last_week_: Optional[np.ndarray] = None
        self.last_train_date_: Optional[pd.Timestamp] = None

    def fit(self, train: pd.Series) -> "SeasonalNaiveDaily":
        if len(train) < self.PERIOD:
            raise ValueError(f"Потрібно щонайменше {self.PERIOD} днів train")
        self.last_week_ = train.iloc[-self.PERIOD :].values
        self.last_train_date_ = train.index.max()
        return self

    def forecast(self, horizon: int) -> pd.Series:
        if self.last_week_ is None:
            raise RuntimeError("Викличте fit() перед forecast()")
        reps = int(np.ceil(horizon / self.PERIOD))
        values = np.tile(self.last_week_, reps)[:horizon]
        idx = pd.date_range(self.last_train_date_ + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(values, index=idx, name="seasonal_naive_pred", dtype=float)


# ---------------------------------------------------------------------------
# Класична модель: SARIMA з фолбеком на ExponentialSmoothing
# ---------------------------------------------------------------------------

def fit_classical_model(train: pd.Series, horizon: int) -> Tuple[str, pd.Series]:
    """Навчає SARIMA на train; якщо вона не сходиться/падає — фолбек на
    Holt-Winters ExponentialSmoothing. Повертає (назва_моделі, прогноз)."""
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels не встановлено. Виконайте: pip install statsmodels"
        )

    freq_idx = pd.date_range(train.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                train,
                order=SARIMA_ORDER,
                seasonal_order=SARIMA_SEASONAL_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            fc = fitted.get_forecast(steps=horizon).predicted_mean
            fc.index = freq_idx
            fc.name = "sarima_pred"
            return "SARIMA(1,1,1)x(1,0,1,7)", fc.clip(lower=0, upper=24)
        except Exception as e:
            print(
                f"[WARN] SARIMA не зійшлась ({e!r}); фолбек на ExponentialSmoothing.",
                file=sys.stderr,
            )

        try:
            ets = ExponentialSmoothing(
                train, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIODS
            ).fit()
            fc = ets.forecast(horizon)
            fc.index = freq_idx
            fc.name = "ets_pred"
            return "ExponentialSmoothing(add,add,7)", fc.clip(lower=0, upper=24)
        except Exception as e:
            raise RuntimeError(f"І SARIMA, і ExponentialSmoothing впали: {e!r}")


# ---------------------------------------------------------------------------
# Метрики та графік
# ---------------------------------------------------------------------------

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true_a, y_pred_a = y_true.align(y_pred, join="inner")
    err = y_true_a.values - y_pred_a.values
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def plot_forecasts(
    region: str,
    train: pd.Series,
    test: pd.Series,
    classical_name: str,
    classical_fc: pd.Series,
    naive_fc: pd.Series,
    out_path: pathlib.Path,
) -> None:
    tail = train.iloc[-30:]  # хвіст train для контексту
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tail.index, tail.values, label="train (хвіст, 30 дн.)", color="grey")
    ax.plot(test.index, test.values, label="факт (test)", color="black", linewidth=2)
    ax.plot(classical_fc.index, classical_fc.values, label=classical_name, linestyle="--")
    ax.plot(naive_fc.index, naive_fc.values, label="SeasonalNaiveDaily", linestyle=":")
    ax.set_title(f"Прогноз годин тривог на добу — {region}")
    ax.set_xlabel("дата")
    ax.set_ylabel("год. тривог/добу (0..24)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Прогін по регіону
# ---------------------------------------------------------------------------

def run_region(region: str, test_days: int = TEST_DAYS) -> pd.DataFrame:
    daily = build_daily_series(region)
    train, test = chronological_split(daily, test_days=test_days)
    horizon = len(test)

    naive = SeasonalNaiveDaily().fit(train)
    naive_fc = naive.forecast(horizon)
    naive_fc.index = test.index  # та сама довжина й послідовні дати, що й test

    classical_name, classical_fc = fit_classical_model(train, horizon)
    classical_fc.index = test.index

    summary = pd.DataFrame(
        {
            "SeasonalNaiveDaily": compute_metrics(test, naive_fc),
            classical_name: compute_metrics(test, classical_fc),
        }
    ).T
    summary.index.name = "model"

    out_path = REPORTS_DIR / f"classical_{region}.png"
    plot_forecasts(region, train, test, classical_name, classical_fc, naive_fc, out_path)
    print(f"[{region}] графік збережено: {out_path}")

    return summary


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not STATSMODELS_AVAILABLE:
        print(
            "[ERROR] statsmodels не встановлено. Виконайте: pip install statsmodels",
            file=sys.stderr,
        )
        sys.exit(1)

    all_summaries = {}
    for region in ["kyiv", "nikopol"]:
        print(f"\n===== {region} =====")
        summary = run_region(region)
        print(summary.round(3).to_string())
        all_summaries[region] = summary

    # самоперевірка: на Нікополі класична модель має побити SeasonalNaiveDaily за MAE
    nik = all_summaries["nikopol"]
    classical_row = [i for i in nik.index if i != "SeasonalNaiveDaily"][0]
    naive_mae = nik.loc["SeasonalNaiveDaily", "mae"]
    classical_mae = nik.loc[classical_row, "mae"]
    print(f"\nНікополь: SeasonalNaiveDaily MAE={naive_mae:.3f}, {classical_row} MAE={classical_mae:.3f}")
    if classical_mae < naive_mae:
        print(f"OK: {classical_row} перевершує SeasonalNaiveDaily на Нікополі.")
    else:
        print(
            "[WARN] Очікувалось, що SARIMA/ETS побʼє SeasonalNaiveDaily на Нікополі "
            "(сильний тренд) — перевір дані/параметри моделі."
        )
