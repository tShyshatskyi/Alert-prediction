"""
backtest/engine.py
Ковзний (expanding window) бектест для baseline-моделей тривог.

Дизайн:
- time_splits(index, n_folds, test_days) — чисто часові, неперекрапні
  фолди: train для фолда k = усі години ДО початку test-блоку k;
  test = наступні test_days днів. Жодного random split / shuffle.
  Кожен наступний train більший (вікно розширюється), кожен наступний
  test — пізніший і не перетинається з попередніми test-блоками.
- Дві тривіальні baseline-моделі (ConstantRate, RecentRate) з тим самим
  fit(train_df)/predict(index)-інтерфейсом, що й у models.baseline.
- run_backtest(region) ганяє 4 моделі (ConstantRate, RecentRate,
  ClimatologyBaseline, PersistenceBaseline) по всіх фолдах і рахує
  Brier / MAE / ROC-AUC / PR-AUC на цілі alert_active, усереднені по
  фолдах.

Вхідні дані: погодинна сітка регіону через
preprocess.build_grid.build_hourly_grid(region) (або parquet, або —
як демонстраційний фолбек — синтетика з трендом, якщо нічого з цього
не доступне в поточному середовищі).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.baseline import ClimatologyBaseline, PersistenceBaseline  # noqa: E402

TARGET_COL = "alert_active"


# ---------------------------------------------------------------------------
# 1. Часові спліти (expanding window)
# ---------------------------------------------------------------------------

def time_splits(
    index: pd.DatetimeIndex, n_folds: int = 5, test_days: int = 30
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Повертає список (train_idx, test_idx) для expanding-window бектесту.

    - train_idx фолда k = усі години СТРОГО до початку test-блоку k
      (тобто max(train_idx) < min(test_idx) гарантовано за побудовою).
    - test_idx фолда k = наступні test_days*24 годин.
    - Останні n_folds test-блоків розташовані в кінці ряду, послідовно
      й без перекриття; train кожного наступного фолда — більший
      (включає в себе test попередніх фолдів).
    - Жодного random split, жодного перемішування — порядок завжди
      хронологічний.
    """
    index = pd.DatetimeIndex(index).sort_values().unique()
    test_hours = test_days * 24
    n = len(index)
    total_test = n_folds * test_hours

    if total_test >= n:
        raise ValueError(
            f"Замало даних ({n} год.) для {n_folds} фолдів по {test_days} днів "
            f"({total_test} год. test); потрібно більше історії або менше фолдів."
        )

    first_test_start = n - total_test
    splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for k in range(n_folds):
        test_start = first_test_start + k * test_hours
        test_end = test_start + test_hours
        train_idx = index[:test_start]
        test_idx = index[test_start:test_end]
        splits.append((pd.DatetimeIndex(train_idx), pd.DatetimeIndex(test_idx)))

    return splits


# ---------------------------------------------------------------------------
# 2. Дві тривіальні baseline-моделі (той самий fit/predict-інтерфейс)
# ---------------------------------------------------------------------------

class ConstantRate:
    """Прогноз = середній alert_active по ВСЬОМУ train (одне число)."""

    def __init__(self, target_col: str = TARGET_COL):
        self.target_col = target_col
        self.rate_: Optional[float] = None

    def fit(self, train_df: pd.DataFrame) -> "ConstantRate":
        self.rate_ = float(train_df[self.target_col].mean())
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        if self.rate_ is None:
            raise RuntimeError("Викличте fit() перед predict()")
        return pd.Series(self.rate_, index=index, name=f"{self.target_col}_pred", dtype=float)


class RecentRate:
    """Прогноз = середній alert_active за останні window_days днів train
    (реагує на тренд, на відміну від ConstantRate)."""

    def __init__(self, target_col: str = TARGET_COL, window_days: int = 14):
        self.target_col = target_col
        self.window = pd.Timedelta(days=window_days)
        self.rate_: Optional[float] = None

    def fit(self, train_df: pd.DataFrame) -> "RecentRate":
        train_df = train_df.sort_index()
        cutoff = train_df.index.max() - self.window
        recent = train_df.loc[train_df.index > cutoff, self.target_col]
        if len(recent) == 0:
            recent = train_df[self.target_col]
        self.rate_ = float(recent.mean())
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        if self.rate_ is None:
            raise RuntimeError("Викличте fit() перед predict()")
        return pd.Series(self.rate_, index=index, name=f"{self.target_col}_pred", dtype=float)


MODEL_FACTORIES = {
    "ConstantRate": ConstantRate,
    "RecentRate": RecentRate,
    "ClimatologyBaseline": ClimatologyBaseline,
    "PersistenceBaseline": PersistenceBaseline,
}


# ---------------------------------------------------------------------------
# 3. Метрики
# ---------------------------------------------------------------------------

def _extract_prob(pred, target_col: str = TARGET_COL) -> pd.Series:
    """Уніфікує вихід predict(): Series (ConstantRate/RecentRate/Climatology)
    або DataFrame з кількома колонками (PersistenceBaseline) -> Series
    ймовірності/прогнозу для target_col."""
    col = f"{target_col}_pred"
    if isinstance(pred, pd.DataFrame):
        if col in pred.columns:
            return pred[col]
        if target_col in pred.columns:
            return pred[target_col]
        raise KeyError(f"Не знайшов колонку '{col}' у виході predict(): {list(pred.columns)}")
    return pred


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "brier": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "roc_auc": _safe_roc_auc(y_true, y_pred),
        "pr_auc": _safe_pr_auc(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# 4. Бектест-цикл
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame, n_folds: int = 5, test_days: int = 30
) -> Tuple[pd.DataFrame, dict]:
    """Прогонить усі MODEL_FACTORIES по expanding-window фолдах df.
    Повертає (summary_df, per_fold_results)."""
    df = df.sort_index()
    splits = time_splits(df.index, n_folds=n_folds, test_days=test_days)

    per_fold: dict = {name: [] for name in MODEL_FACTORIES}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        # самоперевірка №1: жодного витоку майбутнього
        assert train_idx.max() < test_idx.min(), (
            f"Фолд {fold_i}: max(train)={train_idx.max()} >= min(test)={test_idx.min()}"
        )

        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
        y_true = test_df[TARGET_COL].values

        for name, factory in MODEL_FACTORIES.items():
            model = factory()
            model.fit(train_df)
            # PersistenceBaseline: "розкриваємо" фактичні значення test як
            # уже спостережені — легітимно, бо лаг 24h завжди в минулому
            # відносно кожного t (див. докстрінг у models/baseline.py).
            if hasattr(model, "update_history"):
                model.update_history(test_df)

            pred = model.predict(test_idx)
            prob = _extract_prob(pred).reindex(test_idx)

            metrics = compute_metrics(y_true, prob.values)
            metrics["fold"] = fold_i
            per_fold[name].append(metrics)

    summary = {
        name: pd.DataFrame(rows)[["brier", "mae", "roc_auc", "pr_auc"]].mean(skipna=True)
        for name, rows in per_fold.items()
    }
    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = "model"
    return summary_df, per_fold


# ---------------------------------------------------------------------------
# Завантаження даних (реальний pipeline -> parquet -> синтетика-фолбек)
# ---------------------------------------------------------------------------

def _load_region(region: str) -> pd.DataFrame:
    try:
        from preprocess.build_grid import build_hourly_grid  # type: ignore

        return build_hourly_grid(region)
    except Exception:
        pass

    for p in (Path(f"data/processed/{region}_hourly.parquet"), Path(f"data/{region}_hourly.parquet")):
        if p.exists():
            return pd.read_parquet(p)

    raise RuntimeError(
        f"Не вдалося завантажити дані для регіону '{region}': "
        "ні build_hourly_grid, ні parquet недоступні. "
        f"Перевір preprocess/build_grid.py і наявність data/processed/{region}_hourly.parquet."
    )


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ALWAYS_HALF_BRIER = 0.25  # Brier «завжди прогнозувати 0.5»
    summaries = {}

    for region in ["kyiv", "nikopol"]:
        print(f"\n===== Backtest: {region} =====")
        df = _load_region(region)
        summary_df, _ = run_backtest(df, n_folds=5, test_days=30)
        summaries[region] = summary_df
        print(summary_df.round(4).to_string())

        sensible = summary_df.index[summary_df["brier"] <= ALWAYS_HALF_BRIER].tolist()
        print(
            f"\nМоделі з Brier <= {ALWAYS_HALF_BRIER} "
            f"(не гірші за 'завжди 0.5'): {sensible}"
        )
        assert sensible, f"Жодна модель не побила тривіальний Brier=0.25 для {region}"

        winner = summary_df["brier"].idxmin()
        print(f"Переможець за Brier для {region}: {winner} (Brier={summary_df.loc[winner, 'brier']:.4f})")

    # самоперевірка: на Нікополі RecentRate має бути кращим за ConstantRate
    cr_brier = summaries["nikopol"].loc["ConstantRate", "brier"]
    rr_brier = summaries["nikopol"].loc["RecentRate", "brier"]
    print(
        f"\nНікополь: ConstantRate Brier={cr_brier:.4f}, RecentRate Brier={rr_brier:.4f}"
    )
    assert rr_brier < cr_brier, (
        "Очікувалось: RecentRate < ConstantRate за Brier на Нікополі (висхідний тренд)"
    )
    print("OK: RecentRate перевершує ConstantRate на Нікополі, як і очікується.")
