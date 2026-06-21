# Air Raid Alert Time-Series Analysis — Ukraine

Analysis and forecasting of air-raid alerts in Ukraine (data: alerts.in.ua, 2022–2026).
The project pairs a **detailed study of two contrasting locations** — Kyiv City (sparse,
irregular alerts) vs Nikopol hromada (frequent, near-constant) — with an **all-oblasts
interactive comparison dashboard**.

**▶ Live dashboard (nothing to download):** https://<your-username>.github.io/Alert-prediction/

## Problem framing

Predicting the exact moment of the next alert is infeasible — alerts are driven by external
military events. The project instead answers a tractable, useful question:

> **What is the probability of an alert in region X during a given hour, and what is the
> expected daily alert load?**

This reframes the task from impossible point-prediction to **probabilistic risk modeling**,
evaluated honestly against simple baselines.

## Interactive dashboard

A self-contained `index.html` (alert data + Ukraine GeoJSON embedded, Plotly via CDN), hosted
on GitHub Pages. Click oblasts on the map to compare several at once:

- **Statistics:** probability by hour-of-day, by day-of-week, daily share over the full period,
  monthly trend, distribution of individual alert durations.
- **Models — comparison:** every model with metrics, sorted best→worst, across all selected
  oblasts (hourly: Brier / ROC-AUC; daily: MAE / RMSE).
- **Forecast:** expected alert-hours for 7/14/30 days, hourly probability for the next day,
  and a 30-day daily-load forecast (SARIMA, with 80% interval for a single oblast).

## Data

- **Alerts:** [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
  (official, start/finish intervals by region, since 2022-03-15). Run `python download_data.py`
  (not committed — too large).
- **Geometry:** geoBoundaries ADM1 (Ukraine oblasts). Run `python get_geojson.py`.
- Timestamps are UTC in the source and converted to `Europe/Kyiv`.

### Data caveats that shaped the project

1. **Alerting granularity changed over time** — until late 2025 alerts were mostly issued
   oblast-wide; afterwards at raion/hromada level. Naively filtering Nikopol by raion dropped
   the early oblast-wide alerts that *did* cover it, inflating the apparent upward trend. The
   filter was corrected to include oblast-level alerts.
2. **"Alert in an oblast" = anywhere in the oblast** (union of all its communities). For
   frontline oblasts this is ~90–99% recently — genuine ("at least one community under alert"),
   not the whole oblast under siren.

## How to run

```bash
pip install pandas pyarrow numpy scikit-learn statsmodels matplotlib requests plotly
python download_data.py            # raw alert data  -> data/raw/
python get_geojson.py              # Ukraine oblast geometry -> reports/
python -m preprocess.build_grid    # hourly grids (Kyiv, Nikopol)
python models/baseline.py          # baseline sanity-check
python backtest/engine.py          # rolling-window backtest (hourly)
python models/classical.py         # daily SARIMA / Holt-Winters
python models/ml.py                # ML + full comparison
python models/forecast.py          # multi-horizon forecast (Kyiv, Nikopol)
python build_map_data.py           # per-oblast stats/metrics/forecasts -> reports/map_data.json
python build_map_html.py           # builds index.html (the dashboard)
```

## Methods

- **Baselines** — ConstantRate, RecentRate (last 14 days), Climatology (hour-of-day × day-of-week),
  Persistence (value 24 h ago).
- **Machine learning** — HistGradientBoosting (scikit-learn): hourly probability classifier and
  daily count regressor with lag/rolling features.
- **Classical** — SARIMA (Holt-Winters fallback) on the daily series.

> HistGradientBoosting and statsmodels were chosen deliberately to avoid heavy or unavailable
> native wheels (LightGBM, Prophet) on very new Python versions (3.14).

## Evaluation

Expanding-window, time-based backtest (5 folds × 30 days). No shuffling and no future leakage:
`max(train) < min(test)` is asserted, and every feature uses `shift(1)` / positive lags.
Hourly probability is scored with **Brier / ROC-AUC / PR-AUC**; daily load with **MAE / RMSE**.

## Key results

- **Nikopol / frontline oblasts** are highly predictable; **ML wins**, combining the recent
  level (calibration) with the daily shape (discrimination).
- **Kyiv / quiet oblasts** are weakly predictable — ML discriminates a little better than
  climatology (AUC ≈ 0.64 vs 0.58), but alerts are largely irregular. An honest negative result.
- On the **daily load**, classical **SARIMA** usually wins MAE.
- **Most alerts are short** — even in Kyiv the median alert lasts ≈ 0.7 h.
- **No single method dominates everywhere.**

## Honest caveats

1. **Online vs static regime.** ML and Persistence observe recent history during the test window
   (one-step-ahead nowcasting); the seasonal baselines are static forecasts. Part of ML's edge
   comes from exploiting recent observations — fair for an operational warning system, but stated.
2. **The trend was partly a data artifact** (granularity change), corrected as above.
3. **Two forecast views differ by design:** "expected hours" assumes the current rate holds; the
   30-day SARIMA chart reverts toward the long-run mean — they can disagree when a region's recent
   level differs from its long-run level.
4. **"Alert" is a coarse target** — a precaution, not a strike. The meaningful (and harder) next
   step is predicting **actual shelling**, fusing alerts with other signals (intelligence, OSINT,
   media). That is where real predictive value lies.

## Methods considered, out of scope

Hawkes / self-exciting point processes (alerts cluster in waves — theoretically the best fit),
HMM regime-switching, and deep learning (LSTM / Transformers) — too heavy or overfit-prone for the
two-day timeframe.

## Project structure

```
Alert-prediction/
├── data/load.py             # load_region: region filters, UTC -> Europe/Kyiv
├── preprocess/build_grid.py # intervals -> continuous hourly grid (merge overlaps)
├── models/
│   ├── baseline.py          # Climatology, Persistence
│   ├── classical.py         # daily SARIMA / Holt-Winters + SeasonalNaive
│   ├── ml.py                # HistGradientBoosting: hourly classifier + daily regressor
│   └── forecast.py          # multi-horizon forecast (climatology×recent + SARIMA)
├── backtest/engine.py       # expanding-window backtest + ConstantRate / RecentRate
├── build_map_data.py        # per-oblast stats/metrics/forecasts -> reports/map_data.json
├── build_map_html.py        # builds index.html (the dashboard)
├── download_data.py         # fetch alert dataset
├── get_geojson.py           # fetch Ukraine oblast GeoJSON
├── explore.ipynb            # EDA (seasonality, trends)
├── index.html               # interactive dashboard (served via GitHub Pages)
├── README.md
└── reflection.txt
```

## Environment

Python 3.13+ (works on 3.14). scikit-learn + statsmodels keep the stack free of heavy native
dependencies.
