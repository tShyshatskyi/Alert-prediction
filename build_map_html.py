# -*- coding: utf-8 -*-
"""Інтерактивний дашборд: choropleth областей України + порівняння кількох
областей одночасно (статистика, найкраща модель, прогноз на 30 днів).
Самодостатній HTML (geojson і дані вшиті; Plotly з CDN)."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
data = json.load(open(ROOT/"reports"/"map_data.json", encoding="utf-8"))
geo  = json.load(open(ROOT/"reports"/"ukraine_oblasts.min.geojson", encoding="utf-8"))

NAME2SHAPE = {
 "Cherkaska oblast":"Cherkasy Oblast","Chernihivska oblast":"Chernihiv Oblast","Chernivetska oblast":"Chernivtsi Oblast",
 "Dnipropetrovska oblast":"Dnipropetrovsk Oblast","Donetska oblast":"Donetsk Oblast","Ivano-Frankivska oblast":"Ivano-Frankivsk Oblast",
 "Kharkivska oblast":"Kharkiv Oblast","Khersonska oblast":"Kherson Oblast","Khmelnytska oblast":"Khmelnytskyi Oblast",
 "Kirovohradska oblast":"Kirovohrad Oblast","Kyiv City":"Kyiv","Kyivska oblast":"Kyiv Oblast","Luhanska oblast":"Luhansk Oblast",
 "Lvivska oblast":"Lviv Oblast","Mykolaivska oblast":"Mykolaiv Oblast","Odeska oblast":"Odessa Oblast","Poltavska oblast":"Poltava Oblast",
 "Rivnenska oblast":"Rivne Oblast","Sumska oblast":"Sumy Oblast","Ternopilska oblast":"Ternopil Oblast","Vinnytska oblast":"Vinnytsia Oblast",
 "Volynska oblast":"Volyn Oblast","Zakarpatska oblast":"Zakarpattia Oblast","Zaporizka oblast":"Zaporizhia Oblast","Zhytomyrska oblast":"Zhytomyr Oblast"}
SHAPE2NAME = {v:k for k,v in NAME2SHAPE.items()}
LABEL = {"Cherkaska oblast":"Черкаська","Chernihivska oblast":"Чернігівська","Chernivetska oblast":"Чернівецька",
 "Dnipropetrovska oblast":"Дніпропетровська","Donetska oblast":"Донецька","Ivano-Frankivska oblast":"Івано-Франківська",
 "Kharkivska oblast":"Харківська","Khersonska oblast":"Херсонська","Khmelnytska oblast":"Хмельницька","Kirovohradska oblast":"Кіровоградська",
 "Kyiv City":"м. Київ","Kyivska oblast":"Київська","Luhanska oblast":"Луганська","Lvivska oblast":"Львівська","Mykolaivska oblast":"Миколаївська",
 "Odeska oblast":"Одеська","Poltavska oblast":"Полтавська","Rivnenska oblast":"Рівненська","Sumska oblast":"Сумська","Ternopilska oblast":"Тернопільська",
 "Vinnytska oblast":"Вінницька","Volynska oblast":"Волинська","Zakarpatska oblast":"Закарпатська","Zaporizka oblast":"Запорізька","Zhytomyrska oblast":"Житомирська"}

# порядок локацій = усі shapeName з geojson
shapes = [f["properties"]["shapeName"] for f in geo["features"]]
locations, z, custom, hover = [], [], [], []
for s in shapes:
    nm = SHAPE2NAME.get(s)
    locations.append(s)
    if nm and data.get(nm, {}).get("status") == "ok":
        rr = data[nm]["recent_rate"]; z.append(rr); custom.append(nm)
        hover.append(f"<b>{LABEL[nm]}</b><br>під тривогою (14д): {rr*100:.0f}%<br>"
                     f"краща модель: {data[nm]['best_model']} (AUC {data[nm]['best_auc']:.2f})<extra></extra>")
    else:
        z.append(None); custom.append(nm or "")
        title = LABEL.get(nm, s)
        hover.append(f"<b>{title}</b><br>немає даних (постійна тривога / окуповано)<extra></extra>")

fig = {"data":[{"type":"choroplethmapbox","locations":locations,"z":z,"customdata":custom,
  "featureidkey":"properties.shapeName","colorscale":"YlOrRd","zmin":0,"zmax":1,
  "hovertemplate":hover,"marker":{"line":{"color":"#0d141c","width":0.6},"opacity":0.85},
  "colorbar":{"title":{"text":"Під тривогою<br>(14 дн.)","font":{"color":"#cbd5e1","size":11}},
    "tickfont":{"color":"#9fb3c8"},"tickformat":".0%","outlinewidth":0,"len":0.75,"x":0.98}}],
 "layout":{"mapbox":{"style":"carto-darkmatter","center":{"lat":48.45,"lon":31.3},"zoom":4.3},
   "paper_bgcolor":"rgba(0,0,0,0)","margin":{"l":0,"r":0,"t":0,"b":0},"height":620}}

mapdata = {}
for nm,d in data.items():
    if d.get("status")!="ok": continue
    bt=d["backtest"]
    bm=min(bt,key=lambda m:bt[m]["brier"]); am=max(bt,key=lambda m:bt[m]["roc_auc"])
    mapdata[nm]={"label":LABEL[nm],"recent_rate":d["recent_rate"],"base_rate":d["base_rate"],
      "brier_model":bm,"brier_val":bt[bm]["brier"],"auc_model":am,"auc_val":bt[am]["roc_auc"],
      "expected_hours_30d":d["expected_hours_30d"],"hourly_profile":d["hourly_profile"],
      "backtest":d["backtest"],"forecast_30d":d["forecast_30d"]}
default = ["Kyiv City","Dnipropetrovska oblast","Lvivska oblast"]

TPL = """<!DOCTYPE html><html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Повітряні тривоги України — дашборд</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;font-family:'Segoe UI',system-ui,sans-serif;background:#0d141c;color:#e6edf3}
.header{padding:16px 24px;border-bottom:1px solid #1d2935}
.header h1{margin:0;font-size:21px}.header p{margin:4px 0 0;color:#7d93a8;font-size:13px}
.wrap{display:flex;gap:16px;padding:16px 20px;align-items:flex-start;flex-wrap:wrap}
.card{background:#121b25;border:1px solid #1d2935;border-radius:14px;padding:14px}
.mapcard{flex:1 1 480px;min-width:380px}.panel{flex:1 1 560px;min-width:380px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:2px 0 12px;min-height:30px}
.chip{background:#22303d;border:1px solid #33414f;border-radius:20px;padding:5px 12px;font-size:13px;cursor:pointer}
.chip b{margin-left:6px;color:#ff9a7a}
.hint{color:#7d93a8;font-size:12px;margin:8px 2px 0}
h4{margin:16px 0 6px;font-size:12px;color:#9fb3c8;text-transform:uppercase;letter-spacing:.05em}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 9px;border-bottom:1px solid #1d2935;text-align:left}
th{color:#7d93a8}.btnclear{float:right;background:#22303d;border:1px solid #33414f;color:#cbd5e1;border-radius:8px;padding:4px 10px;cursor:pointer;font-size:12px}
</style></head><body>
<div class="header"><h1>Повітряні тривоги України — інтерактивний дашборд</h1>
<p>Натисніть області на мапі, щоб додати їх до порівняння · дані alerts.in.ua 2022–2026</p></div>
<div class="wrap">
 <div class="card mapcard"><div id="map"></div>
   <div class="hint">Колір = частка годин під тривогою за 14 днів. Сірі — немає даних (постійна тривога / окуповано).</div></div>
 <div class="card panel">
   <button class="btnclear" onclick="clearSel()">очистити</button>
   <h4>Обрані області (порівняння)</h4><div class="chips" id="chips"></div>
   <h4>Зведення</h4><div id="tbl"></div>
   <h4>Ймовірність тривоги за годиною доби</h4><div id="profile"></div>
   <h4>Прогноз на 30 днів — годин тривог на добу</h4><div id="forecast"></div>
 </div>
</div>
<script>
const GEOJSON=__GEO__, FIG=__FIG__, MAP_DATA=__DATA__;
const PALETTE=["#ff8c3c","#4ea1ff","#52d273","#e36bd0","#f2d24b","#9b8cff","#ff6b6b","#3fd0c9"];
const LOCS=FIG.data[0].locations, CUST=FIG.data[0].customdata;
FIG.data[0].geojson=GEOJSON;
const DARK={paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:"#cbd5e1",size:11},
 margin:{l:44,r:12,t:8,b:34},height:270,legend:{orientation:"h",y:1.2,font:{size:10}}};
Plotly.newPlot('map',FIG.data,FIG.layout,{responsive:true,displayModeBar:false});
let selected=__DEFAULT__.slice();
document.getElementById('map').on('plotly_click',e=>{
  const nm=e.points[0].customdata;
  if(!nm||!MAP_DATA[nm]){return;}
  const i=selected.indexOf(nm); if(i>=0)selected.splice(i,1); else selected.push(nm);
  if(selected.length>8)selected.shift(); render();});
function clearSel(){selected=[];render();}
function colorOf(nm){return PALETTE[selected.indexOf(nm)%PALETTE.length];}
function render(){
  // підсвітка на мапі
  const w=LOCS.map((l,i)=>selected.includes(CUST[i])?2.6:0.6);
  Plotly.restyle('map',{'marker.line.width':[w],'marker.line.color':[LOCS.map((l,i)=>selected.includes(CUST[i])?'#ffffff':'#0d141c')]});
  // чіпи
  document.getElementById('chips').innerHTML = selected.length? selected.map(nm=>
    `<span class="chip" style="border-color:${colorOf(nm)}" onclick="rm('${nm}')">${MAP_DATA[nm].label}<b>×</b></span>`).join('')
    : '<span class="hint">Нічого не обрано — натисніть область на мапі.</span>';
  // таблиця
  if(selected.length){
    let rows=selected.map(nm=>{const d=MAP_DATA[nm];return `<tr>
      <td style="color:${colorOf(nm)}">●</td><td>${d.label}</td>
      <td>${(d.recent_rate*100).toFixed(0)}%</td><td>${(d.base_rate*100).toFixed(0)}%</td>
      <td>${d.brier_model} (${d.brier_val.toFixed(3)})</td><td>${d.auc_model} (${d.auc_val.toFixed(2)})</td><td>${Math.round(d.expected_hours_30d)}</td></tr>`}).join('');
    document.getElementById('tbl').innerHTML=`<table><tr><th></th><th>область</th><th>14дн</th><th>увесь період</th>
      <th>рівень (Brier↓)</th><th>розрізнення годин (AUC↑)</th><th>год/30дн</th></tr>${rows}</table><p class="hint">Brier — точність самої ймовірності (нижче=краще). AUC — чи модель відрізняє небезпечні години (0.5=ніяк, &gt;0.6 — є сигнал). Константні моделі виграють Brier, але AUC≈0.5, бо не ранжують години.</p>`;
  } else document.getElementById('tbl').innerHTML='';
  // профілі доби (накладені)
  const pr=selected.map(nm=>({x:[...Array(24).keys()],y:MAP_DATA[nm].hourly_profile,mode:'lines',
    name:MAP_DATA[nm].label,line:{color:colorOf(nm),width:2}}));
  Plotly.newPlot('profile',pr,{...DARK,xaxis:{title:'година доби'},yaxis:{tickformat:'.0%',range:[0,1]}},{displayModeBar:false});
  // прогнози (накладені; CI лише якщо одна область)
  let fc=[];
  if(selected.length===1){const d=MAP_DATA[selected[0]],f=d.forecast_30d;
    fc=[{x:f.dates,y:f.high,line:{width:0},showlegend:false,hoverinfo:'skip'},
        {x:f.dates,y:f.low,fill:'tonexty',fillcolor:'rgba(255,140,60,0.2)',line:{width:0},name:'80% інтервал'},
        {x:f.dates,y:f.point,mode:'lines',name:d.label,line:{color:colorOf(selected[0]),width:2}}];}
  else fc=selected.map(nm=>{const f=MAP_DATA[nm].forecast_30d;return {x:f.dates,y:f.point,mode:'lines',
        name:MAP_DATA[nm].label,line:{color:colorOf(nm),width:2}}});
  Plotly.newPlot('forecast',fc,{...DARK,yaxis:{title:'год/добу',range:[0,24]}},{displayModeBar:false});
}
function rm(nm){const i=selected.indexOf(nm);if(i>=0)selected.splice(i,1);render();}
render();
</script></body></html>"""

html = (TPL.replace("__GEO__", json.dumps(geo,separators=(",",":")))
          .replace("__FIG__", json.dumps(fig))
          .replace("__DATA__", json.dumps(mapdata))
          .replace("__DEFAULT__", json.dumps(default)))
out = ROOT/"reports"/"alert_map.html"
out.write_text(html, encoding="utf-8")
print("Збережено:", out, "| розмір:", round(len(html)/1e6,2), "МБ")
