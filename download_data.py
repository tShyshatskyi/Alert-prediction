import urllib.request, pathlib

URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
out = pathlib.Path("data/raw/alerts.csv")
out.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(URL, out)
print("Завантажено:", out)
