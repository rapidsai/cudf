# Copyright (c) 2018-2019, NVIDIA CORPORATION.

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    # This assumes the script is executed from the root of the repo directory
    ./build.sh -v libcudf --allgpuarch --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
else
    ./build.sh -v libcudf tests --allgpuarch --build_metrics --incl_cache_stats --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
fi
