#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION.

set -e
set -o pipefail

# Update this to add/remove/change a dataset, using the following format:
#
#  comment about the dataset
#  dataset download URL
#  destination dir to untar to
#  blank line separator
#
# FIXME: some test data needs to be extracted to "benchmarks", which is
# confusing now that there's dedicated datasets for benchmarks.
CUIO_BENCHMARK_DATASET_DATA="
# 10GB File
https://rapidsai-data.s3.us-east-2.amazonaws.com/cudf/benchmark/avro_json_datasets.zip
cudf/benchmarks/cuio_data/
"

################################################################################
# Do not change the script below this line if only adding/updating a dataset

NUMARGS=$#
ARGS=$*
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help; then
    echo "$0 [--cuio_benchmark | -d dir_path| -u url]"
    echo " "
    echo "If --cuio_benchmark option is used, it will download standard dataset,"
    echo "If you want to download different data-set, use -u along with url to get it."
    echo "Similarly use -d set directory where this dataset needs to be stored."
    echo " "
    echo "Any predefined data-set options can't be used in conjunction with -d and -u options."

    exit 0
fi

# Select the datasets to install
# Update this as and when new datasets and requirement arrive
if hasArg "-d" && hasArg "-u"; then
    while getopts 'd:u:' flag; do
        case "${flag}" in
            d) DESTDIRS="${OPTARG}" ;;
            u) URLS="${OPTARG}" ;;
            *) echo "${flag} option not found"
               exit 1 ;;
        esac
    done
else
    if hasArg "--cuio_benchmark"; then
        DATASET_DATA="${CUIO_BENCHMARK_DATASET_DATA}"
    else
        DATASET_DATA="${CUIO_BENCHMARK_DATASET_DATA}"
    fi
    URLS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 3) print $0}'))  # extract 3rd fields to a bash array
    DESTDIRS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 0) print $0}'))  # extract 4th fields to a bash array
fi

echo Downloading ...

# Download all tarfiles to a tmp dir
rm -rf tmp
mkdir tmp
cd tmp
for url in ${URLS[*]}; do
   time wget --progress=dot:giga ${url}
done
cd ..

# Setup the destination dirs, removing any existing ones first!
for index in ${!DESTDIRS[*]}; do
    rm -rf ${DESTDIRS[$index]}
done
for index in ${!DESTDIRS[*]}; do
    mkdir -p ${DESTDIRS[$index]}
done

# Iterate over the arrays and untar the nth tarfile to the nth dest directory.
# The tarfile name is derived from the download url.
echo Decompressing ...
for index in ${!DESTDIRS[*]}; do
    tfname=$(basename ${URLS[$index]})
    unzip tmp/${tfname} -d ${DESTDIRS[$index]}
done

rm -rf tmp
