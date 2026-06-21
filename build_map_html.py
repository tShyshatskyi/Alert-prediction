# -*- coding: utf-8 -*-
"""Дашборд повітряних тривог: choropleth + мультивибір областей.
Секції: статистика (профіль доби, день тижня, частка по днях за весь час,
тренд по місяцях, тривалість епізодів) -> порівняння моделей (сортовано) ->
ПРОГНОЗ (очікувані години 7/14/30, ймовірність по годинах наступної доби,
30-денний прогноз). Самодостатній HTML; Plotly з CDN."""
import json, pathlib
ROOT=pathlib.Path(__file__).resolve().parent
data=json.load(open(ROOT/"reports"/"map_data.json",encoding="utf-8"))
geo =json.load(open(ROOT/"reports"/"ukraine_oblasts.min.geojson",encoding="utf-8"))

NAME2SHAPE={"Cherkaska oblast":"Cherkasy Oblast","Chernihivska oblast":"Chernihiv Oblast","Chernivetska oblast":"Chernivtsi Oblast",
 "Dnipropetrovska oblast":"Dnipropetrovsk Oblast","Donetska oblast":"Donetsk Oblast","Ivano-Frankivska oblast":"Ivano-Frankivsk Oblast",
 "Kharkivska oblast":"Kharkiv Oblast","Khersonska oblast":"Kherson Oblast","Khmelnytska oblast":"Khmelnytskyi Oblast",
 "Kirovohradska oblast":"Kirovohrad Oblast","Kyiv City":"Kyiv","Kyivska oblast":"Kyiv Oblast","Luhanska oblast":"Luhansk Oblast",
 "Lvivska oblast":"Lviv Oblast","Mykolaivska oblast":"Mykolaiv Oblast","Odeska oblast":"Odessa Oblast","Poltavska oblast":"Poltava Oblast",
 "Rivnenska oblast":"Rivne Oblast","Sumska oblast":"Sumy Oblast","Ternopilska oblast":"Ternopil Oblast","Vinnytska oblast":"Vinnytsia Oblast",
 "Volynska oblast":"Volyn Oblast","Zakarpatska oblast":"Zakarpattia Oblast","Zaporizka oblast":"Zaporizhia Oblast","Zhytomyrska oblast":"Zhytomyr Oblast"}
SHAPE2NAME={v:k for k,v in NAME2SHAPE.items()}
LABEL={"Cherkaska oblast":"Черкаська","Chernihivska oblast":"Чернігівська","Chernivetska oblast":"Чернівецька",
 "Dnipropetrovska oblast":"Дніпропетровська","Donetska oblast":"Донецька","Ivano-Frankivska oblast":"Івано-Франківська",
 "Kharkivska oblast":"Харківська","Khersonska oblast":"Херсонська","Khmelnytska oblast":"Хмельницька","Kirovohradska oblast":"Кіровоградська",
 "Kyiv City":"м. Київ","Kyivska oblast":"Київська","Luhanska oblast":"Луганська","Lvivska oblast":"Львівська","Mykolaivska oblast":"Миколаївська",
 "Odeska oblast":"Одеська","Poltavska oblast":"Полтавська","Rivnenska oblast":"Рівненська","Sumska oblast":"Сумська","Ternopilska oblast":"Тернопільська",
 "Vinnytska oblast":"Вінницька","Volynska oblast":"Волинська","Zakarpatska oblast":"Закарпатська","Zaporizka oblast":"Запорізька","Zhytomyrska oblast":"Житомирська"}

shapes=[f["properties"]["shapeName"] for f in geo["features"]]
locations,z,custom,hover=[],[],[],[]
for s in shapes:
    nm=SHAPE2NAME.get(s); locations.append(s)
    if nm and data.get(nm,{}).get("status")=="ok":
        rr=data[nm]["recent_rate"]; z.append(rr); custom.append(nm)
        hover.append(f"<b>{LABEL[nm]}</b><br>тривога будь-де в області (14д): {rr*100:.0f}%<extra></extra>")
    else:
        z.append(None); custom.append(nm or "")
        hover.append(f"<b>{LABEL.get(nm,s)}</b><br>немає даних (постійна тривога / окуповано)<extra></extra>")

fig={"data":[{"type":"choroplethmapbox","locations":locations,"z":z,"customdata":custom,
  "featureidkey":"properties.shapeName","colorscale":"YlOrRd","zmin":0,"zmax":1,"hovertemplate":hover,
  "marker":{"line":{"color":"#0d141c","width":0.6},"opacity":0.85},
  "colorbar":{"title":{"text":"Тривога будь-де<br>в області (14 дн.)","font":{"color":"#cbd5e1","size":11}},
    "tickfont":{"color":"#9fb3c8"},"tickformat":".0%","outlinewidth":0,"len":0.75,"x":0.98}}],
 "layout":{"mapbox":{"style":"carto-darkmatter","center":{"lat":48.45,"lon":31.3},"zoom":4.3},
   "paper_bgcolor":"rgba(0,0,0,0)","margin":{"l":0,"r":0,"t":0,"b":0},"height":560}}

md={}
for nm,d in data.items():
    if d.get("status")!="ok": continue
    md[nm]={"label":LABEL[nm],"recent_rate":d["recent_rate"],"base_rate":d["base_rate"],
      "hourly_profile":d["hourly_profile"],"dow_profile":d["dow_profile"],
      "month_m":[x["m"] for x in d["monthly_trend"]],"month_r":[x["rate"] for x in d["monthly_trend"]],
      "daily_start":d["daily_series"]["start"],"daily_vals":d["daily_series"]["vals"],
      "dur_labels":d["duration_labels"],"dur_counts":d["duration_counts"],
      "bt_hourly":d["backtest_hourly"],"bt_daily":d["backtest_daily"],
      "fc_dates":d["forecast_30d"]["dates"],"fc_point":d["forecast_30d"]["point"],
      "fc_low":d["forecast_30d"]["low"],"fc_high":d["forecast_30d"]["high"],
      "h7":d["forecast_horizons"]["7"],"h14":d["forecast_horizons"]["14"],"h30":d["forecast_horizons"]["30"]}
default=["Kyiv City","Dnipropetrovska oblast","Lvivska oblast"]
import datetime
_any=next(iter(md.values()))
FCSTART=_any["fc_dates"][0]; FCEND=_any["fc_dates"][-1]
DATAUNTIL=(datetime.date.fromisoformat(FCSTART)-datetime.timedelta(days=1)).isoformat()

TPL=r"""<!DOCTYPE html><html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Повітряні тривоги України — дашборд</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;font-family:'Segoe UI',system-ui,sans-serif;background:#0d141c;color:#e6edf3}
.header{padding:16px 24px;border-bottom:1px solid #1d2935}.header h1{margin:0;font-size:21px}.header p{margin:4px 0 0;color:#7d93a8;font-size:13px}
.wrap{display:flex;gap:16px;padding:16px 20px;align-items:flex-start;flex-wrap:wrap}
.card{background:#121b25;border:1px solid #1d2935;border-radius:14px;padding:14px}
.mapcard{flex:1 1 460px;min-width:360px;position:sticky;top:16px}.panel{flex:1 1 560px;min-width:380px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:2px 0 10px;min-height:30px}
.chip{background:#22303d;border:1px solid #33414f;border-radius:20px;padding:5px 12px;font-size:13px;cursor:pointer}.chip b{margin-left:6px;color:#ff9a7a}
.hint{color:#7d93a8;font-size:12px;margin:6px 2px 0}
.sec{font-size:13px;color:#8fd3ff;text-transform:uppercase;letter-spacing:.06em;margin:22px 0 2px;border-top:2px solid #25455e;padding-top:12px;font-weight:700}
h4{margin:16px 0 6px;font-size:12px;color:#9fb3c8;text-transform:uppercase;letter-spacing:.05em}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 9px;border-bottom:1px solid #1d2935;text-align:left}th{color:#7d93a8}
tr.best td{color:#ffd27d;font-weight:700}
.bestc{color:#ffd27d;font-weight:700}
.btnclear{float:right;background:#22303d;border:1px solid #33414f;color:#cbd5e1;border-radius:8px;padding:4px 10px;cursor:pointer;font-size:12px}
.cols{display:flex;gap:16px;flex-wrap:wrap}.cols>div{flex:1 1 240px}
</style></head><body>
<div class="header"><h1>Повітряні тривоги України — інтерактивний дашборд</h1>
<p>Натисніть області для порівняння · дані alerts.in.ua до <b>__DATAUNTIL__</b> · прогноз на <b>__FCSTART__ – __FCEND__</b></p></div>
<div class="wrap">
 <div class="card mapcard"><div id="map"></div>
   <div class="hint">Колір = частка годин, коли тривога була <b>хоч десь</b> в області (останні 14 днів). Сірі — немає даних (постійна тривога / окуповано).</div></div>
 <div class="card panel">
   <button class="btnclear" onclick="clearSel()">очистити</button>
   <h4 style="margin-top:0">Обрані області</h4><div class="chips" id="chips"></div>
   <div id="tbl"></div>

   <div class="sec">Статистика (весь період)</div>
   <h4>Ймовірність тривоги за годиною доби</h4><div id="profile"></div>
   <h4>За днем тижня</h4><div id="dow"></div>
   <h4>Частка годин під тривогою по днях (за весь час)</h4><div id="daily"></div>
   <h4>Тренд по місяцях</h4><div id="monthly"></div>
   <h4>Тривалість епізодів тривог (годин)</h4><div id="duration"></div>

   <div class="sec">Моделі — порівняння (усі обрані області)</div>
   <h4>Brier по моделях (стовпчики — області, нижче = краще)</h4><div id="modelbar"></div>
   <h4>Погодинні моделі — Brier ↓ (найкраще підсвічено)</h4><div id="mh"></div>
   <h4>Погодинні моделі — ROC-AUC ↑</h4><div id="mhA"></div>
   <h4>Добові моделі — MAE ↓</h4><div id="md"></div>

   <div class="sec">Прогноз на __FCSTART__ – __FCEND__</div>
   <h4>Очікувані години тривог</h4><div id="fhtab"></div>
   <h4>Ймовірність тривоги по годинах наступної доби</h4><div id="fhour"></div>
   <h4>Прогноз навантаження на 30 днів (годин/добу)</h4><div id="forecast"></div>
 </div>
</div>
<script>
const GEOJSON=__GEO__,FIG=__FIG__,MAP_DATA=__DATA__;
const PAL=["#ff8c3c","#4ea1ff","#52d273","#e36bd0","#f2d24b","#9b8cff"];
const DOW=["Пн","Вт","Ср","Чт","Пт","Сб","Нд"];
const LOCS=FIG.data[0].locations,CUST=FIG.data[0].customdata;FIG.data[0].geojson=GEOJSON;
const DARK={paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:"#cbd5e1",size:11},
 margin:{l:46,r:12,t:8,b:34},height:240,legend:{orientation:"h",y:1.25,font:{size:10}}};
const CFG={displayModeBar:false,responsive:true};
let selected=__DEFAULT__.slice();
function clearSel(){selected=[];render();}
function onMapClick(e){const nm=e.points[0].customdata;
 if(!nm||!MAP_DATA[nm])return;const i=selected.indexOf(nm);if(i>=0)selected.splice(i,1);else selected.push(nm);
 if(selected.length>6)selected.shift();render();}
function colorOf(nm){return PAL[selected.indexOf(nm)%PAL.length];}
function rm(nm){const i=selected.indexOf(nm);if(i>=0)selected.splice(i,1);render();}
function lines(acc,x){return selected.map(nm=>({x:x||[...Array(24).keys()],y:acc(MAP_DATA[nm]),mode:'lines',
 name:MAP_DATA[nm].label,line:{color:colorOf(nm),width:2}}));}
function genDates(start,n){const out=[];let d=new Date(start+'T00:00:00Z');for(let i=0;i<n;i++){out.push(d.toISOString().slice(0,10));d.setUTCDate(d.getUTCDate()+1);}return out;}
function fcHourly(d){const m=d.hourly_profile.reduce((a,b)=>a+b,0)/24||1;return d.hourly_profile.map(p=>Math.min(1,p/m*d.recent_rate));}
function matrix(btKey,metric,better,dg){
 const models=Object.keys(MAP_DATA[selected[0]][btKey]);const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
 models.sort((a,b)=>{const ma=mean(selected.map(nm=>MAP_DATA[nm][btKey][a][metric])),mb=mean(selected.map(nm=>MAP_DATA[nm][btKey][b][metric]));return better==='min'?ma-mb:mb-ma;});
 const bestPer={};selected.forEach(nm=>{const arr=models.map(m=>MAP_DATA[nm][btKey][m][metric]);bestPer[nm]=better==='min'?Math.min(...arr):Math.max(...arr);});
 const head=`<tr><th>модель</th>${selected.map(nm=>`<th style="color:${colorOf(nm)}">${MAP_DATA[nm].label}</th>`).join('')}</tr>`;
 const rows=models.map(m=>`<tr><td>${m}</td>${selected.map(nm=>{const v=MAP_DATA[nm][btKey][m][metric];return `<td${v===bestPer[nm]?' class="bestc"':''}>${v.toFixed(dg)}</td>`;}).join('')}</tr>`).join('');
 return `<table>${head}${rows}</table>`;}
function render(){
 const w=LOCS.map((l,i)=>selected.includes(CUST[i])?2.6:0.6);
 Plotly.restyle('map',{'marker.line.width':[w],'marker.line.color':[LOCS.map((l,i)=>selected.includes(CUST[i])?'#fff':'#0d141c')]});
 document.getElementById('chips').innerHTML=selected.length?selected.map(nm=>
   `<span class="chip" style="border-color:${colorOf(nm)}" onclick="rm('${nm}')">${MAP_DATA[nm].label}<b>×</b></span>`).join('')
   :'<span class="hint">Нічого не обрано — натисніть область на мапі.</span>';
 const ids=['tbl','profile','dow','daily','monthly','duration','modelbar','mh','mhA','md','fhtab','fhour','forecast'];
 if(!selected.length){ids.forEach(i=>document.getElementById(i).innerHTML='');return;}
 // зведення
 let rows=selected.map(nm=>{const d=MAP_DATA[nm];return `<tr><td style="color:${colorOf(nm)}">●</td><td>${d.label}</td>
   <td>${(d.recent_rate*100).toFixed(0)}%</td><td>${(d.base_rate*100).toFixed(0)}%</td></tr>`}).join('');
 document.getElementById('tbl').innerHTML=`<table><tr><th></th><th>область</th><th>14дн (будь-де)</th><th>увесь період</th></tr>${rows}</table>`;
 // статистика
 Plotly.newPlot('profile',lines(d=>d.hourly_profile),{...DARK,xaxis:{title:'година доби'},yaxis:{tickformat:'.0%',range:[0,1]}},CFG);
 Plotly.newPlot('dow',lines(d=>d.dow_profile,DOW),{...DARK,yaxis:{tickformat:'.0%',range:[0,1]}},CFG);
 Plotly.newPlot('daily',selected.map(nm=>{const d=MAP_DATA[nm];return {x:genDates(d.daily_start,d.daily_vals.length),y:d.daily_vals,
   mode:'lines',name:d.label,line:{color:colorOf(nm),width:1}}}),{...DARK,height:260,yaxis:{tickformat:'.0%',range:[0,1]}},CFG);
 Plotly.newPlot('monthly',selected.map(nm=>({x:MAP_DATA[nm].month_m,y:MAP_DATA[nm].month_r,mode:'lines',
   name:MAP_DATA[nm].label,line:{color:colorOf(nm),width:2}})),{...DARK,yaxis:{tickformat:'.0%',range:[0,1]}},CFG);
 Plotly.newPlot('duration',selected.map(nm=>({x:MAP_DATA[nm].dur_labels,y:MAP_DATA[nm].dur_counts,type:'bar',
   name:MAP_DATA[nm].label,marker:{color:colorOf(nm)}})),{...DARK,barmode:'group',xaxis:{title:'тривалість, год'},yaxis:{title:'епізодів'}},CFG);
 // моделі: порівняння всіх обраних областей
 const p0=selected[0];
 const order=Object.keys(MAP_DATA[p0].bt_hourly).sort((a,b)=>MAP_DATA[p0].bt_hourly[a].brier-MAP_DATA[p0].bt_hourly[b].brier);
 Plotly.newPlot('modelbar',selected.map(nm=>({x:order,y:order.map(m=>MAP_DATA[nm].bt_hourly[m].brier),type:'bar',
   name:MAP_DATA[nm].label,marker:{color:colorOf(nm)}})),{...DARK,barmode:'group',yaxis:{title:'Brier ↓'}},CFG);
 document.getElementById('mh').innerHTML=matrix('bt_hourly','brier','min',3);
 document.getElementById('mhA').innerHTML=matrix('bt_hourly','roc_auc','max',2);
 document.getElementById('md').innerHTML=matrix('bt_daily','mae','min',2);
 // ПРОГНОЗ
 let fr=selected.map(nm=>{const d=MAP_DATA[nm];return `<tr><td style="color:${colorOf(nm)}">●</td><td>${d.label}</td>
   <td>${Math.round(d.h7)}</td><td>${Math.round(d.h14)}</td><td>${Math.round(d.h30)}</td></tr>`}).join('');
 document.getElementById('fhtab').innerHTML=`<table><tr><th></th><th>область</th><th>за 7 дн</th><th>за 14 дн</th><th>за 30 дн</th></tr>${fr}</table>`;
 Plotly.newPlot('fhour',lines(d=>fcHourly(d)),{...DARK,xaxis:{title:'година доби'},yaxis:{tickformat:'.0%',range:[0,1],title:'ймовірність'}},CFG);
 let fc;
 if(selected.length===1){const d=MAP_DATA[selected[0]];fc=[
   {x:d.fc_dates,y:d.fc_high,line:{width:0},showlegend:false,hoverinfo:'skip'},
   {x:d.fc_dates,y:d.fc_low,fill:'tonexty',fillcolor:'rgba(255,140,60,0.2)',line:{width:0},name:'80% інтервал'},
   {x:d.fc_dates,y:d.fc_point,mode:'lines',name:d.label,line:{color:colorOf(selected[0]),width:2}}];}
 else fc=selected.map(nm=>({x:MAP_DATA[nm].fc_dates,y:MAP_DATA[nm].fc_point,mode:'lines',name:MAP_DATA[nm].label,line:{color:colorOf(nm),width:2}}));
 Plotly.newPlot('forecast',fc,{...DARK,yaxis:{title:'год/добу',range:[0,24]}},CFG);
}
Plotly.newPlot('map',FIG.data,FIG.layout,{responsive:true,displayModeBar:false}).then(gd=>{gd.on('plotly_click',onMapClick);render();});
</script></body></html>"""
html=(TPL.replace("__GEO__",json.dumps(geo,separators=(",",":")))
        .replace("__FIG__",json.dumps(fig))
        .replace("__DATA__",json.dumps(md))
        .replace("__DEFAULT__",json.dumps(default)).replace("__DATAUNTIL__",DATAUNTIL).replace("__FCSTART__",FCSTART).replace("__FCEND__",FCEND))
out=ROOT/"reports"/"alert_map.html";out.write_text(html,encoding="utf-8")
(ROOT/"index.html").write_text(html,encoding="utf-8")  # для GitHub Pages (відкривається за URL)
print("Збережено:",out,"та index.html | розмір:",round(len(html)/1e6,2),"МБ")
