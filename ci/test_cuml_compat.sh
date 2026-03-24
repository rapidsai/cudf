#!/bin/bash
set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../
source ./ci/test_python_common.sh test_python_cudf

# Install cuml nightly (gets cuml + its dependencies from RAPIDS nightly channel)
rapids-mamba-retry install -n test -c rapidsai-nightly cuml

rapids-logger "pytest cuml cuDF-compat subset"
timeout 10m python -m pytest --pyargs --cache-clear \
  cuml.tests.test_input_utils \
  cuml.tests.test_array \
  cuml.tests.test_reflection \
  cuml.tests.test_validation \
  cuml.tests.test_label_encoder \
  cuml.tests.test_target_encoder \
  cuml.tests.test_one_hot_encoder \
  cuml.tests.test_ordinal_encoder \
  cuml.tests.test_train_test_split \
  cuml.tests.test_split \
  cuml.tests.test_compose \
  cuml.tests.test_text_feature_extraction \
  cuml.tests.test_kneighbors_classifier \
  cuml.tests.test_kneighbors_regressor \
  cuml.tests.test_nearest_neighbors \
  cuml.tests.test_metrics \
  cuml.tests.test_linear_model \
  cuml.tests.test_sgd \
  cuml.tests.test_svm \
  cuml.tests.test_pickle \
  cuml.tests.test_random_forest
