# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import argparse
import os
import shutil
from collections import namedtuple

# Update url and dir where datasets needs to be copied
Dataset = namedtuple("Dataset", ["url", "dir"])
datasets = {
    "cuio_dataset": Dataset(
        "https://data.rapids.ai/cudf/benchmark/avro_json_datasets.zip",
        "cudf/benchmarks/cuio_data/",
    ),
}


def delete_dir(path):
    if path == "/" or path == "~":
        raise ValueError("Trying to delete root/home directory")

    shutil.rmtree(path, ignore_errors=True)


def fetch_datasets(urls, dirs):
    tmp_path = os.path.join(os.getcwd(), "tmp_benchmark/")
    delete_dir(tmp_path)
    os.mkdir(tmp_path)
    for url, path in zip(urls, dirs):
        path = os.path.join(os.getcwd(), path)

        delete_dir(path)
        os.mkdir(path)

        os.system("wget " + url + " -P " + tmp_path)
        os.system(
            "unzip " + tmp_path + "/" + url.split("/")[-1] + " -d " + path
        )

    delete_dir(tmp_path)


urls = []
dirs = []

parser = argparse.ArgumentParser(
    description="""
    Fetches datasets as per given option.
    By default it will download all available datasets
    """
)

parser.add_argument("-u", nargs=1, help="url of a dataset")
parser.add_argument(
    "-d",
    nargs=1,
    help="path where downloaded dataset from given url will be unzipped",
)
parser.add_argument(
    "--datasets",
    nargs="+",
    help="Currently supported datasets are: "
    + ", ".join(list(datasets.keys())),
)
args = parser.parse_args()

if (args.u is None and args.d is not None) or (
    args.u is not None and args.d is None
):
    raise ValueError(
        "option -u and -d should be used together, can't use only one"
    )

if args.u and args.d:
    urls.append(args.u[0])
    dirs.append(args.d[0])

if args.datasets:
    for dataset in args.datasets:
        urls.append(datasets[dataset].url)
        dirs.append(datasets[dataset].dir)

if len(dirs) != len(set(dirs)):
    raise ValueError("Duplicate destination paths are provided")

if len(urls) == 0:
    for _, val in datasets.items():
        urls.append(val.url)
        dirs.append(val.dir)

fetch_datasets(urls, dirs)
