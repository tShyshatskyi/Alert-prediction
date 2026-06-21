"""
models/ml.py
ML-моделі для прогнозу повітряних тривог — доповнюють baseline.py і
classical.py.

ЧАСТИНА A: MLClassifier на ПОГОДИННОМУ ряді (alert_active) — той самий
fit(train_df)/predict(index)/update_history(df)-інтерфейс, що й у
PersistenceBaseline, тож вмикається прямо в backtest.engine.run_backtest()
поряд із 4 baseline-моделями.

ЧАСТИНА B: MLRegressorDaily на ДОБОВОМУ ряді (alert_hours) — рекурсивний
h-кроковий прогноз, чесно порівнюваний з SARIMA і SeasonalNaiveDaily з
models/classical.py (жодна з моделей не бачить test під час прогнозу).

Запускається і як `python models/ml.py`, і як `python -m models.ml`.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from preprocess.build_grid import build_hourly_grid
from models.classical import (
    build_daily_series,
    chronological_split,
    SeasonalNaiveDaily,
    fit_classical_model,
    compute_metrics,
)

TARGET_COL = "alert_active"


# ===========================================================================
# ЧАСТИНА A — ML на погодинному ряді
# ===========================================================================

class MLClassifier:
    """
    HistGradientBoostingClassifier для ймовірності alert_active на
    погодинному ряді. Інтерфейс ідентичний PersistenceBaseline:
    fit(train_df) / update_history(df) / predict(index).

    ОЗНАКИ (усі — лише з минулого відносно години t):
      - hour, dayofweek, month                         (календарні, відомі
                                                          наперед, не залежать
                                                          від факту)
      - lag_24, lag_168                                 (alert_active рівно
                                                          24/168 год тому)
      - roll_24h, roll_7d, roll_14d                      (середній
                                                          alert_active за
                                                          попередні 24/168/336
                                                          год, з shift(1) —
                                                          поточна година НЕ
                                                          входить у вікно)

    Робота з історією:
      fit() зберігає self.history_ = train_df[target_col] і навчає модель
      ЛИШЕ на ознаках, побудованих з train (рядки з NaN-лагами — на самому
      початку ряду — відкидаються перед навчанням).
      update_history(df) дописує спостережені дані (як і в
      PersistenceBaseline: лаг 24/168 для будь-якого t завжди строго в
      минулому щодо t, тож "розкриття" test у history_ не є витоком
      майбутнього).
      predict(index) рахує ознаки виключно з history_ (через shift/lag,
      без жодного доступу до факту в самих точках index) і повертає
      predict_proba(...)[:, 1].
    """

    LAGS = (24, 168)
    ROLL_WINDOWS = {"roll_24h": 24, "roll_7d": 24 * 7, "roll_14d": 24 * 14}

    def __init__(self, target_col: str = TARGET_COL, **hgb_kwargs):
        self.target_col = target_col
        params = {"random_state": 42}
        params.update(hgb_kwargs)
        self.model = HistGradientBoostingClassifier(**params)
        self.history_: Optional[pd.Series] = None
        self.feature_names_: Optional[list] = None

    # -- ознаки -------------------------------------------------------------

    def _build_features(self, index: pd.DatetimeIndex, history: pd.Series) -> pd.DataFrame:
        index = pd.DatetimeIndex(index)
        feats = pd.DataFrame(index=index)
        feats["hour"] = index.hour
        feats["dayofweek"] = index.dayofweek
        feats["month"] = index.month

        # суцільна погодинна сітка від мінімуму історії до максимуму
        # (history або index, що пізніше) — щоб лаги рахувались за
        # фактичну відстань у годинах, а не за позицію в розрідженому ряді.
        tz = history.index.tz
        grid_end = max(history.index.max(), index.max())
        full_grid = pd.date_range(history.index.min(), grid_end, freq="h", tz=tz)
        full = history.reindex(full_grid)

        for lag in self.LAGS:
            feats[f"lag_{lag}"] = full.shift(lag).reindex(index).values

        shifted = full.shift(1)  # "до поточної години", сама година t не входить
        for name, window in self.ROLL_WINDOWS.items():
            feats[name] = shifted.rolling(window, min_periods=1).mean().reindex(index).values

        return feats

    # -- fit / update_history / predict -------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "MLClassifier":
        train_df = train_df.sort_index()
        if self.target_col not in train_df.columns:
            raise KeyError(f"Колонка '{self.target_col}' відсутня в train_df")

        self.history_ = train_df[self.target_col].copy()

        X = self._build_features(train_df.index, self.history_)
        y = train_df[self.target_col].values

        valid = X.notna().all(axis=1)
        X_valid, y_valid = X.loc[valid], y[valid]
        if X_valid.empty:
            raise ValueError("Після відкидання NaN-лагів не лишилось рядків для навчання")

        self.feature_names_ = list(X.columns)
        self.model.fit(X_valid, y_valid)
        return self

    def update_history(self, df: pd.DataFrame) -> "MLClassifier":
        if self.history_ is None:
            raise RuntimeError("Спершу викличте fit()")
        extra = df[self.target_col].copy()
        self.history_ = pd.concat([self.history_, extra]).sort_index()
        self.history_ = self.history_[~self.history_.index.duplicated(keep="last")]
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        if self.history_ is None:
            raise RuntimeError("Модель не навчена: викличте fit() перед predict()")

        X = self._build_features(index, self.history_)
        # фолбек на 0.0, якщо для самих перших точок ряду історії все ще
        # замало (NaN-лаги); реально в бектесті train завжди >> 168 год.
        X = X.fillna(0.0)
        X = X[self.feature_names_]

        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=index, name=f"{self.target_col}_pred", dtype=float)


# ===========================================================================
# ЧАСТИНА B — ML на добовому ряді (рекурсивний прогноз)
# ===========================================================================

class MLRegressorDaily:
    """
    HistGradientBoostingRegressor для добового ряду "годин тривог на
    добу" (0..24). Прогноз на test-горизонт РЕКУРСИВНИЙ: модель не бачить
    жодного факту з test — кожен наступний день прогнозується на основі
    власних попередніх прогнозів (і "хвоста" train на старті), так само,
    як SARIMA/ETS у fit_classical_model() не бачать test.

    ОЗНАКИ (усі — зі shift, тобто з минулого відносно дня t):
      dayofweek, month, lag_1, lag_7, roll_7, roll_14 (середнє за
      попередні 7/14 днів, з shift(1)).
    """

    LAGS = (1, 7)
    ROLL_WINDOWS = {"roll_7": 7, "roll_14": 14}

    def __init__(self, **hgb_kwargs):
        params = {"random_state": 42}
        params.update(hgb_kwargs)
        self.model = HistGradientBoostingRegressor(**params)
        self.history_: Optional[pd.Series] = None
        self.feature_names_: Optional[list] = None

    def _features_for_series(self, series: pd.Series) -> pd.DataFrame:
        feats = pd.DataFrame(index=series.index)
        feats["dayofweek"] = series.index.dayofweek
        feats["month"] = series.index.month
        for lag in self.LAGS:
            feats[f"lag_{lag}"] = series.shift(lag)
        shifted = series.shift(1)
        for name, window in self.ROLL_WINDOWS.items():
            feats[name] = shifted.rolling(window, min_periods=1).mean()
        return feats

    def fit(self, train: pd.Series) -> "MLRegressorDaily":
        train = train.sort_index()
        X = self._features_for_series(train)
        y = train.values

        valid = X.notna().all(axis=1)
        X_valid, y_valid = X.loc[valid], y[valid]
        if X_valid.empty:
            raise ValueError("Після відкидання NaN-лагів не лишилось рядків для навчання")

        self.feature_names_ = list(X.columns)
        self.model.fit(X_valid, y_valid)
        self.history_ = train.copy()
        return self

    def forecast(self, horizon: int) -> pd.Series:
        """Рекурсивний прогноз на horizon днів наперед. НЕ використовує
        жодного значення з test — кожен крок будується або на train, або
        на власних попередніх прогнозах."""
        if self.history_ is None:
            raise RuntimeError("Викличте fit() перед forecast()")

        history = self.history_.copy()
        fallback = float(history.mean())
        preds = []
        future_idx = pd.date_range(
            history.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D"
        )

        for day in future_idx:
            extended = pd.concat([history, pd.Series([np.nan], index=[day])])
            feats_full = self._features_for_series(extended)
            x_today = feats_full.loc[[day], self.feature_names_].fillna(fallback)

            pred = float(self.model.predict(x_today)[0])
            pred = float(np.clip(pred, 0, 24))
            preds.append(pred)

            history.loc[day] = pred  # дописуємо ВЛАСНИЙ прогноз, не факт test

        return pd.Series(preds, index=future_idx, name="ml_daily_pred", dtype=float)


def run_daily_comparison(region: str, test_days: int = 60) -> pd.DataFrame:
    """SeasonalNaiveDaily vs SARIMA/ETS (fit_classical_model) vs
    MLRegressorDaily — MAE/RMSE на test добового ряду region."""
    daily = build_daily_series(region)
    train, test = chronological_split(daily, test_days=test_days)
    horizon = len(test)
    assert train.index.max() < test.index.min(), "Витік майбутнього в Частині B"

    naive = SeasonalNaiveDaily().fit(train)
    naive_fc = naive.forecast(horizon)
    naive_fc.index = test.index

    classical_name, classical_fc = fit_classical_model(train, horizon)
    classical_fc.index = test.index

    ml = MLRegressorDaily().fit(train)
    ml_fc = ml.forecast(horizon)
    ml_fc.index = test.index

    summary = pd.DataFrame(
        {
            "SeasonalNaiveDaily": compute_metrics(test, naive_fc),
            classical_name: compute_metrics(test, classical_fc),
            "MLRegressorDaily": compute_metrics(test, ml_fc),
        }
    ).T
    summary.index.name = "model"
    return summary


# ===========================================================================
# Self-check
# ===========================================================================

if __name__ == "__main__":
    import backtest.engine as eng

    # реєструємо ML поряд із 4 baseline-моделями в самому ж бектест-движку
    eng.MODEL_FACTORIES = {**eng.MODEL_FACTORIES, "MLClassifier": MLClassifier}

    print("=" * 72)
    print("ЧАСТИНА A: ML на погодинному ряді (бектест проти 4 baseline)")
    print("=" * 72)

    hourly_summaries: Dict[str, pd.DataFrame] = {}
    for region in ["kyiv", "nikopol"]:
        print(f"\n----- {region} -----")
        df = build_hourly_grid(region)
        summary_df, _ = eng.run_backtest(df, n_folds=5, test_days=30)
        print(summary_df.round(4).to_string())
        hourly_summaries[region] = summary_df

        winner = summary_df["brier"].idxmin()
        print(f"Переможець за Brier ({region}): {winner} (Brier={summary_df.loc[winner, 'brier']:.4f})")

    print("\n" + "=" * 72)
    print("ЧАСТИНА B: ML на добовому ряді (рекурсивний прогноз vs SARIMA/Naive)")
    print("=" * 72)

    daily_summaries: Dict[str, pd.DataFrame] = {}
    for region in ["kyiv", "nikopol"]:
        print(f"\n----- {region} -----")
        summary_df = run_daily_comparison(region)
        print(summary_df.round(3).to_string())
        daily_summaries[region] = summary_df

        winner = summary_df["mae"].idxmin()
        print(f"Переможець за MAE ({region}): {winner} (MAE={summary_df.loc[winner, 'mae']:.3f})")

    # -----------------------------------------------------------------------
    # Самоперевірки
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("САМОПЕРЕВІРКИ")
    print("=" * 72)

    nik = hourly_summaries["nikopol"]
    ml_brier, recent_brier = nik.loc["MLClassifier", "brier"], nik.loc["RecentRate", "brier"]
    ml_auc, clim_auc = nik.loc["MLClassifier", "roc_auc"], nik.loc["ClimatologyBaseline", "roc_auc"]

    print(f"Нікополь: MLClassifier Brier={ml_brier:.4f} vs RecentRate Brier={recent_brier:.4f}")
    print(f"Нікополь: MLClassifier ROC-AUC={ml_auc:.4f} vs ClimatologyBaseline ROC-AUC={clim_auc:.4f}")
    if ml_brier <= recent_brier and ml_auc >= clim_auc:
        print("OK: MLClassifier поєднує переваги RecentRate (Brier) і Climatology (ROC-AUC) на Нікополі.")
    else:
        print(
            "[Чесний результат] MLClassifier НЕ домінує одночасно над RecentRate і "
            "Climatology на Нікополі за обома метриками — фіксуємо як є, без підкручування."
        )

    kyiv = hourly_summaries["kyiv"]
    ml_auc_kyiv = kyiv.loc["MLClassifier", "roc_auc"]
    clim_auc_kyiv = kyiv.loc["ClimatologyBaseline", "roc_auc"]
    delta = ml_auc_kyiv - clim_auc_kyiv
    print(
        f"\nКиїв: MLClassifier ROC-AUC={ml_auc_kyiv:.4f} vs ClimatologyBaseline ROC-AUC={clim_auc_kyiv:.4f} "
        f"(delta={delta:+.4f}; очікується лише невелике покращення — Київ малопередбачуваний)"
    )
