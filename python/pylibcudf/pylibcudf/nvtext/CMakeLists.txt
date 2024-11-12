# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

set(cython_sources
    edit_distance.pyx generate_ngrams.pyx jaccard.pyx minhash.pyx ngrams_tokenize.pyx normalize.pyx
    replace.pyx stemmer.pyx tokenize.pyx byte_pair_encode.pyx subword_tokenize.pyx
)

set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX pylibcudf_nvtext_ ASSOCIATED_TARGETS cudf
)
