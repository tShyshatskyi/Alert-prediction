# Air Raid Alert Time-Series Analysis — Kyiv & Nikopol

Analysis and forecasting of air-raid alerts in Ukraine for two contrasting
locations: **Kyiv City** (sparse, irregular alerts) and **Nikopol hromada**
(frequent, near-constant alerts). Trivial baselines, machine learning and
classical statistical models are compared under a rigorous, leakage-free,
rolling-window backtest.

## Problem framing

Predicting *exactly when* the next alert will sound is not feasible — alerts
are driven by external military events. Instead the project answers a
tractable and useful question:

> **What is the probability of an alert in region X during a given hour, and
> what is the expected daily alert load?**

This reframes the task from impossible point-prediction to **probabilistic
risk modeling**, evaluated honestly against simple baselines.

## Data

- Source: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
  (official alerts, start/finish intervals, by region, since 2022-03-15).
- The raw file is **not** committed (too large). Run `python download_data.py`
  to fetch `data/raw/alerts.csv`.
- Timestamps are UTC in the source and converted to `Europe/Kyiv`.

### A data caveat that shaped the project

The alerting **granularity changed over time**: until late 2025 most alerts
were issued at **oblast (region) level**; afterwards at **raion / hromada**
level. Naively filtering Nikopol by `raion == "Nikopolskyi raion"` therefore
**dropped** the early oblast-wide alerts that *did* cover Nikopol — making
2022–2023 look artificially quiet and **inflating the apparent upward trend**.
The Nikopol filter was corrected to also include oblast-level Dnipropetrovska
alerts. (Kyiv City is its own oblast-level entity and needs no such fix.)

## Project structure

```
Alert-prediction/
├── data/
│   ├── load.py            # load_region("kyiv" | "nikopol"), region filters, UTC->Kyiv
│   ├── raw/alerts.csv     # downloaded (gitignored)
│   └── processed/         # *_hourly.parquet (generated, gitignored)
├── preprocess/
│   └── build_grid.py      # intervals -> continuous hourly grid (merge overlaps)
├── models/
│   ├── baseline.py        # ConstantRate*, RecentRate*, Climatology, Persistence
│   ├── classical.py       # daily SARIMA / Holt-Winters + SeasonalNaive
│   └── ml.py              # HistGradientBoosting: hourly classifier + daily regressor
├── backtest/
│   └── engine.py          # expanding-window, time-based backtest
├── reports/               # generated forecast plots
├── download_data.py
├── explore.ipynb          # EDA (seasonality, trends)
├── README.md
└── reflection.txt
```
\* `ConstantRate` / `RecentRate` live in `backtest/engine.py`.

## How to run

```bash
pip install pandas pyarrow numpy scikit-learn statsmodels matplotlib requests
python download_data.py            # fetch raw data
python -m preprocess.build_grid    # build hourly grids -> data/processed/
python models/baseline.py          # sanity-check baselines
python backtest/engine.py          # rolling-window backtest (hourly)
python models/classical.py         # daily SARIMA / Holt-Winters
python models/ml.py                # ML + full comparison (hourly & daily)
```

## Methods

- **Baselines** — ConstantRate, RecentRate (last 14 days), Climatology
  (hour-of-day × day-of-week), Persistence (value 24h ago).
- **Machine learning** — HistGradientBoosting (scikit-learn): hourly
  probability classifier and daily count regressor, with lag and
  rolling-window features. (HistGradientBoosting / statsmodels were chosen
  deliberately to avoid heavy or unavailable wheels — e.g. LightGBM / Prophet —
  on very new Python versions.)
- **Classical** — SARIMA (Holt-Winters fallback) on the daily series.

## Evaluation

Expanding-window, time-based backtest (5 folds × 30 days). No shuffling, no
future leakage: `max(train) < min(test)` is asserted and every feature uses
`shift(1)` / positive lags. Hourly probability is scored with **Brier /
ROC-AUC / PR-AUC**; daily load with **MAE / RMSE**.

## Key results

| Task | Kyiv | Nikopol |
|---|---|---|
| Hourly — best Brier | Climatology ≈ ML (~0.100) | **ML 0.064** |
| Hourly — best ROC-AUC | **ML 0.62** | **ML 0.81** |
| Daily — best MAE | **SARIMA 2.21** | **SARIMA 2.04** |
| Base rate (hours under alert) | ~12% | ~67% |

- **Nikopol** is highly predictable; ML wins the hourly task by combining the
  recent level (calibration) with the daily shape (discrimination).
- **Kyiv** is weakly predictable; ML discriminates slightly better than
  climatology, but alerts are largely irregular — an honest negative result.
- On the **daily** series, classical **SARIMA** wins.
- **No single method dominates everything.**

## Honest caveats

1. **Online vs static regime.** ML and Persistence observe history during the
   test window (one-step-ahead nowcasting); the seasonal baselines are static
   forecasts over the whole test block. Part of ML's edge comes from exploiting
   recent observations — fair for an operational warning system, but stated
   explicitly.
2. **The trend was partly a data artifact** (granularity change), corrected as
   described above.
3. **"Alert" is a coarse target.** An air-raid alert is a *precaution*, not a
   strike. The genuinely valuable (and harder) target would be **actual
   shelling / strikes**, which would require fusing other signals —
   intelligence reports, OSINT, media statements, launch/telemetry data. That
   is the natural next step and where real predictive value lies.

## Methods considered but out of scope

- **Hawkes / self-exciting point processes** — theoretically the best fit
  (alerts cluster in waves), but too heavy for the timeframe.
- **HMM / regime-switching** — "calm" vs "active" states; promising future work.
- **Deep learning (LSTM / Transformers)** — overkill for one or two series and
  prone to overfitting here.
