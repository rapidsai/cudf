import os
import urllib.request

data_dir = "./data/weather/"
if not os.path.exists(data_dir):
    print("creating weather directory")
    os.system("mkdir -p ./data/weather")

# download weather observations
base_url = "ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/"
years = list(range(2010, 2020))
for year in years:
    fn = str(year) + ".csv.gz"
    if not os.path.isfile(data_dir + fn):
        print(f"Downloading {base_url+fn} to {data_dir+fn}")
        urllib.request.urlretrieve(base_url + fn, data_dir + fn)

# download weather station metadata
station_meta_url = (
    "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
)
if not os.path.isfile(data_dir + "ghcnd-stations.txt"):
    print("Downloading station meta..")
    urllib.request.urlretrieve(
        station_meta_url, data_dir + "ghcnd-stations.txt"
    )
