import urllib.request, json, pathlib

import urllib.request, json, pathlib

# 1) API geoBoundaries -> отримати прямий лінк на geojson
api = "https://www.geoboundaries.org/api/current/gbOpen/UKR/ADM1/"
meta = json.load(urllib.request.urlopen(api))
url = meta["gjDownloadURL"]
print("geojson URL:", url)

# 2) скачати
out = pathlib.Path("reports/ukraine_oblasts.geojson")
out.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, out)

# 3) перевірити й показати назви
gj = json.load(open(out, encoding="utf-8"))
print("features:", len(gj["features"]))
print("props:", list(gj["features"][0]["properties"].keys()))
print("shapeName-и:")
for f in gj["features"]:
    print("  -", f["properties"].get("shapeName"))