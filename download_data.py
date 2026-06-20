import requests, pathlib

URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
out = pathlib.Path("data/raw/alerts.csv")
out.parent.mkdir(parents=True, exist_ok=True)

r = requests.get(URL, timeout=180)
r.raise_for_status()
out.write_bytes(r.content)

lines = r.text.strip().splitlines()
print("Розмір:", len(r.content), "байт")
print("Рядків:", len(lines))
print("Перший рядок:", lines[0])
print("Останній рядок:", lines[-1])