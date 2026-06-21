# -*- coding: utf-8 -*-
"""Збір повної per-oblast статистики для дашборда -> reports/map_data.json.
Для КОЖНОЇ області рахує:
  - базові частоти, профіль доби (24), профіль дня тижня (7), тренд по місяцях,
    розподіл тривалостей тривог;
  - ПОГОДИННИЙ бектест усіх моделей (4 baseline + MLClassifier): Brier, ROC-AUC;
  - ДОБОВИЙ бектест (SeasonalNaive, SARIMA, MLRegressor): MAE, RMSE;
  - прогноз на 7/14/30 днів (очікувані години) + добовий ряд SARIMA на 30 днів.
Важко рахується (ML+SARIMA x24) — кілька хвилин.
"""
import sys, pathlib, json, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import data.load as L
from preprocess.build_grid import build_hourly_grid
import backtest.engine as eng
from models.ml import MLClassifier, MLRegressorDaily
from models.classical import (build_daily_series, chronological_split,
                              SeasonalNaiveDaily, fit_classical_model, compute_metrics)
from models.forecast import forecast_region
from data.load import load_region
from preprocess.build_grid import merge_intervals
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

_raw=pd.read_csv("data/raw/alerts.csv",header=0,dtype=str)
GLOBAL_MAX=pd.to_datetime(_raw["finished_at"],utc=True,errors="coerce").max().tz_convert("Europe/Kyiv")
RECENT_WIN=pd.date_range(end=GLOBAL_MAX.floor("h"),periods=14*24,freq="h",tz="Europe/Kyiv")

OBLASTS = ["Cherkaska oblast","Chernihivska oblast","Chernivetska oblast","Dnipropetrovska oblast",
 "Donetska oblast","Ivano-Frankivska oblast","Kharkivska oblast","Khersonska oblast","Khmelnytska oblast",
 "Kirovohradska oblast","Kyiv City","Kyivska oblast","Luhanska oblast","Lvivska oblast","Mykolaivska oblast",
 "Odeska oblast","Poltavska oblast","Rivnenska oblast","Sumska oblast","Ternopilska oblast","Vinnytska oblast",
 "Volynska oblast","Zakarpatska oblast","Zaporizka oblast","Zhytomyrska oblast"]

# зареєструвати всі області + додати ML у помапний бектест
for nm in OBLASTS:
    L.REGIONS[nm] = (lambda n: (lambda df: df["oblast"] == n))(nm)
eng.MODEL_FACTORIES = {**eng.MODEL_FACTORIES, "MLClassifier": MLClassifier}

DUR_EDGES = [0,0.5,1,2,3,4,6,9,1e9]
DUR_LABELS = ["<0.5","0.5-1","1-2","2-3","3-4","4-6","6-9","9+"]

def compute_region(name):
    grid = build_hourly_grid(name)
    _full = pd.date_range(grid.index.min(), GLOBAL_MAX.floor("h"), freq="h", tz="Europe/Kyiv")
    grid = grid.reindex(_full, fill_value=0)   # доповнити нулями до спільної "тепер"
    aa = grid["alert_active"]
    idx = grid.index
    base = float(aa.mean())
    recent = float(aa.reindex(RECENT_WIN, fill_value=0).mean())  # спільне вікно "останні 14 днів"
    hourly = [float(x) for x in aa.groupby(idx.hour).mean().reindex(range(24)).fillna(0)]
    dow = [float(x) for x in aa.groupby(idx.dayofweek).mean().reindex(range(7)).fillna(0)]
    mon = aa.groupby(idx.to_period("M")).mean()
    monthly = [{"m": str(p), "rate": float(v)} for p, v in mon.items()]
    # добовий ряд за весь час (частка годин під тривогою на добу)
    ds = aa.resample("D").mean()
    daily_series = {"start": str(ds.index.min().date()), "vals": [round(float(x),3) for x in ds.values]}
    # тривалість ОКРЕМИХ тривог (год) — кожен запис alerts.in.ua, без об'єднання районів
    ivr = load_region(name)
    durs = ((ivr["finished_at"]-ivr["started_at"]).dt.total_seconds()/3600).values
    counts = [int(c) for c in np.histogram(durs, bins=DUR_EDGES)[0]] if len(durs) else [0]*8

    # погодинний бектест (4 baseline + ML)
    bt_summary, _ = eng.run_backtest(grid)
    bt_hourly = {m: {"brier": float(bt_summary.loc[m,"brier"]),
                     "roc_auc": float(bt_summary.loc[m,"roc_auc"])} for m in bt_summary.index}

    # добовий бектест (SeasonalNaive, SARIMA, MLRegressor) на холдауті 60 днів
    daily = build_daily_series(name)
    tr, te = chronological_split(daily, 60)
    sn = SeasonalNaiveDaily().fit(tr); sn_fc = sn.forecast(len(te)); sn_fc.index = te.index
    _, sar_fc = fit_classical_model(tr, len(te)); sar_fc.index = te.index
    mld = MLRegressorDaily().fit(tr); ml_fc = mld.forecast(len(te)); ml_fc.index = te.index
    bt_daily = {"SeasonalNaive": compute_metrics(te, sn_fc),
                "SARIMA": compute_metrics(te, sar_fc),
                "MLRegressor": compute_metrics(te, ml_fc)}

    # ПРОГНОЗ від спільної "тепер" (GLOBAL_MAX), а не від останньої тривоги регіону
    daily_hours = aa.resample("D").sum()
    fdates = pd.date_range(GLOBAL_MAX.floor("D")+pd.Timedelta(days=1), periods=30, freq="D")
    try:
        _r = SARIMAX(daily_hours, order=(1,1,1), seasonal_order=(1,0,1,7),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        _f = _r.get_forecast(30); pt = _f.predicted_mean.clip(0,24)
        _ci = _f.conf_int(alpha=0.2); low = _ci.iloc[:,0].clip(0,24); high = _ci.iloc[:,1].clip(0,24)
    except Exception:
        pt = ExponentialSmoothing(daily_hours, trend="add", seasonal="add", seasonal_periods=7).fit().forecast(30).clip(0,24)
        low = pt; high = pt
    forecast_30d = {"dates":[str(d.date()) for d in fdates],
                    "point":[float(x) for x in np.asarray(pt)],
                    "low":[float(x) for x in np.asarray(low)],
                    "high":[float(x) for x in np.asarray(high)]}
    horizons = {"7": 24*7*recent, "14": 24*14*recent, "30": 24*30*recent}

    bm = min(bt_hourly, key=lambda m: bt_hourly[m]["brier"])
    am = max(bt_hourly, key=lambda m: bt_hourly[m]["roc_auc"])
    return {"status":"ok","base_rate":base,"recent_rate":recent,
            "hourly_profile":hourly,"dow_profile":dow,"monthly_trend":monthly,
            "duration_labels":DUR_LABELS,"duration_counts":counts,"daily_series":daily_series,
            "backtest_hourly":bt_hourly,"backtest_daily":bt_daily,
            "best_brier_model":bm,"best_auc_model":am,
            "forecast_30d":forecast_30d,"forecast_horizons":horizons,
            "expected_hours_30d":horizons["30"]}

if __name__ == "__main__":
    import time
    result, ok, err = {}, 0, []
    for i, nm in enumerate(OBLASTS, 1):
        t=time.time()
        try:
            result[nm] = compute_region(nm); ok += 1
            print(f"[{i}/{len(OBLASTS)}] {nm} ... OK ({time.time()-t:.1f}s, recent={result[nm]['recent_rate']:.2f})")
        except Exception as e:
            result[nm] = {"status":"error","error":repr(e)}; err.append(nm)
            print(f"[{i}/{len(OBLASTS)}] {nm} ... ПОМИЛКА: {e}")
    pathlib.Path("reports").mkdir(exist_ok=True)
    json.dump(result, open("reports/map_data.json","w",encoding="utf-8"), ensure_ascii=False)
    print(f"\nГотово: {ok}/{len(OBLASTS)} ок, помилок: {len(err)} {err}")
    print("Збережено reports/map_data.json")
