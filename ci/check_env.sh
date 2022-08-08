#!/bin/bash

gpuci_logger "Get env"
env

gpuci_logger "Check versions"
python --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls
