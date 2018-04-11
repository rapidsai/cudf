#!/bin/bash
printenv
nvidia-smi
conda env create --name pygdf_dev --file conda_environments/testing_py35.yml --force
source activate pygdf_dev
py.test
