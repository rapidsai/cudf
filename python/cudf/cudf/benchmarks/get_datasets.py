import getopt
import sys
import os
import shutil

# Update url and dir where datasets need to be copied
datasets = {
    "cuio_dataset" : 
        [
            "https://github.com/rapidsai/cudf/files/5424844/temp_bool_orc.zip",
            "cudf/benchmarks/cuio_data1/"
        ],
}

def delete_dir(path):
    if path == "/" or path == "~":
        raise ValueError("Trying to delete root/home directory")

    #shutil.rmtree(path, ignore_errors=False)

def fetch_datasets(urls, dirs):
    tmp_path = os.getcwd() + "/tmp_benchmark/"
    delete_dir(tmp_path)
    os.mkdir(tmp_path)
    for url, path in zip(urls, dirs):
        path = os.getcwd() + "/" + path

        delete_dir(path)
        os.mkdir(path)

        os.system("wget " + url + " -P " + tmp_path)
        os.system("unzip " + tmp_path + "/" + url.split('/')[-1] + " -d " + path)

    delete_dir(tmp_path)


urls = []
dirs = []

options, remainder = getopt.getopt(sys.argv[1:], "hu:d", ["help", "url=", "dir="] + list(datasets.keys()))

for opt, arg in options:
    if opt in ("-h", "--help"):
        print("""
              python cudf/benchmarks/get_datasets.py --cuio_dataset | -u url -d path
              
              Using this python script you can download existing set of datasets, or provide url of
              a new dataset and path where it needs to be stored.

              By default, all the available datasets will be downloaded.
              Following sets of datasets are available for download: \n"""
              "                " + ",  ".join(list(datasets.keys())) + "\n"
              """
              If a url is provided using -u/--url, then a path must also be provided with option -d/--dir.
              """)
        exit()
    elif opt in ("-u", "--url"):
        urls.append(arg)
    elif opt in ("-d", "--dir"):
        dirs.append(arg)
    elif opt[2:] in datasets.keys():
        urls.append(datasets[opt[2:]][0])
        dirs.append(atasets[opt[2:]][1])

if (len(urls)>len(dirs)):
    raise ValueError("Missed providing path to store dataset from url")
elif(len(urls)<len(dirs)):
    raise ValueError("Missed providing url to fetch dataset to store in path")

if len(urls) == 0:
    for _, val in datasets.items():
        urls.append(val[0])
        dirs.append(val[1])

if len(urls) > 0:
    fetch_datasets(urls, dirs)

