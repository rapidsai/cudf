# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
import numpy as np
import os
import pandas as pd
import shutil
import argparse
from collections import namedtuple

# Update url and dir where datasets needs to be copied
Dataset = namedtuple("Dataset", ["url", "dir"])
datasets = {
    "cuio_dataset": Dataset(
        (
            "https://rapidsai-data.s3.us-east-2.amazonaws.com/cudf/"
            "benchmark/avro_json_datasets.zip"
        ),
        "cudf/benchmarks/cuio_data/",
    ),
}


def create_random_data(dtype, file_type, only_file):
    file_dir = "cudf/benchmarks/cuio_data/"
    file_path = os.path.join(file_dir, "file_data"
                             + str(dtype) + "." + file_type)

    if only_file:
        return None, file_path, None

    if dtype == 'datetime64[s]':
        n_samples = 2**19
    else:
        n_samples = 2**21
    n_features = 2**6
    random_state = 23
    np.random.seed(random_state)
    X = np.random.rand(n_samples, n_features)

    return X, file_path, n_features


def create_cudf_dataset(dtype, file_type, only_file):
    X, file_path, n_features = create_random_data(dtype,
                                                  file_type, only_file)
    if only_file:
        return file_path

    X = cudf.DataFrame(X).astype(dtype)
    X.columns = [str(i) for i in range(n_features)]

    return X, file_path


def create_pandas_dataset(dtype, file_type, only_file):
    X, file_path, n_features = create_random_data(dtype,
                                                  file_type, only_file)
    if only_file:
        return file_path

    X = pd.DataFrame(X).astype(dtype)
    X.columns = [str(i) for i in range(n_features)]

    return X, file_path


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


if __name__ == '__main__':
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
