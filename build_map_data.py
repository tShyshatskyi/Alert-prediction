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

OBLASTS = ["Cherkaska oblast","Chernihivska oblast","Chernivetska oblast","Dnipropetrovska oblast",
 "Donetska oblast","Ivano-Frankivska oblast","Kharkivska oblast","Khersonska oblast","Khmelnytska oblast",
 "Kirovohradska oblast","Kyiv City","Kyivska oblast","Luhanska oblast","Lvivska oblast","Mykolaivska oblast",
 "Odeska oblast","Poltavska oblast","Rivnenska oblast","Sumska oblast","Ternopilska oblast","Vinnytska oblast",
 "Volynska oblast","Zakarpatska oblast","Zaporizka oblast","Zhytomyrska oblast"]

# зареєструвати всі області + додати ML у помапний бектест
for nm in OBLASTS:
    L.REGIONS[nm] = (lambda n: (lambda df: df["oblast"] == n))(nm)
eng.MODEL_FACTORIES = {**eng.MODEL_FACTORIES, "MLClassifier": MLClassifier}

DUR_EDGES = [0.5,1.5,2.5,3.5,4.5,6.5,8.5,12.5,24.5,1e9]
DUR_LABELS = ["1","2","3","4","5-6","7-8","9-12","13-24","25+"]

def run_lengths(active):
    runs=[]; c=0
    for v in active:
        if v==1: c+=1
        elif c>0: runs.append(c); c=0
    if c>0: runs.append(c)
    return runs

def compute_region(name):
    grid = build_hourly_grid(name)
    aa = grid["alert_active"]
    idx = grid.index
    base = float(aa.mean())
    recent = float(aa[idx > idx.max()-pd.Timedelta(days=14)].mean())
    hourly = [float(x) for x in aa.groupby(idx.hour).mean().reindex(range(24)).fillna(0)]
    dow = [float(x) for x in aa.groupby(idx.dayofweek).mean().reindex(range(7)).fillna(0)]
    mon = aa.groupby(idx.to_period("M")).mean()
    monthly = [{"m": str(p), "rate": float(v)} for p, v in mon.items()]
    runs = run_lengths(aa.values)
    counts = [int(c) for c in np.histogram(runs, bins=DUR_EDGES)[0]] if runs else [0]*9

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

    # прогнози на 7/14/30 днів
    fc = forecast_region(name, horizons=(7,14,30))
    horizons = {str(h): float(fc[h]["hourly_prob"].sum()) for h in (7,14,30)}  # очікувані години
    f30 = fc[30]
    forecast_30d = {"dates":[str(d.date()) for d in f30["sarima_point"].index],
                    "point":[float(x) for x in f30["sarima_point"].values],
                    "low":[float(x) for x in f30["sarima_low"].values],
                    "high":[float(x) for x in f30["sarima_high"].values]}

    bm = min(bt_hourly, key=lambda m: bt_hourly[m]["brier"])
    am = max(bt_hourly, key=lambda m: bt_hourly[m]["roc_auc"])
    return {"status":"ok","base_rate":base,"recent_rate":recent,
            "hourly_profile":hourly,"dow_profile":dow,"monthly_trend":monthly,
            "duration_labels":DUR_LABELS,"duration_counts":counts,
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
