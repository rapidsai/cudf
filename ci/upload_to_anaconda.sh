#!/bin/bash

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-upload-to-anaconda "${CPP_CHANNEL}"
rapids-upload-to-anaconda "${PYTHON_CHANNEL}"