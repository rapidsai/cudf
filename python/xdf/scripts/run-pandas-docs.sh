#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Builds the pandas docs with xdf.
#
# Usage:
#   run-pandas-docs.sh
#
# Examples
#   run-pandas-docs.sh
#   run-pandas-docs.sh
#
# This script creates a `pandas-testing` directory if it doesn't exist
# Assumes you ran `pip install myst-nb rst-to-myst``
set -euo pipefail

# Grab the Pandas source corresponding to the version
# of Pandas installed.
PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)")

mkdir -p pandas-testing
cd pandas-testing

if [ ! -d "pandas" ]; then
    git clone https://github.com/pandas-dev/pandas
    cd pandas && git checkout v$PANDAS_VERSION && cd ../
fi

mkdir -p pandas-docs
cp -rT pandas/doc pandas-docs/doc
cd pandas-docs/

for rst_fle in $(find doc/source/user_guide -iname "*.rst")
do
    # TODO: There are ".. code-block:: ipython" blocks that are not as trival to replace
    sed -i 's/{{ header }}/.. code-block:: python\n\n   import numpy as np\n   import xdf as pd/g' $rst_fle
    sed -i 's/.. ipython::/.. code-block::/g' $rst_fle
    sed -i '/:suppress:/d' $rst_fle
    sed -i '/:okexcept:/d' $rst_fle
    sed -i '/:okwarning:/d' $rst_fle
    rst2myst convert $rst_fle
done

for md_file in $(find doc/source/user_guide -iname "*.md")
do
    # https://myst-nb.readthedocs.io/en/latest/authoring/text-notebooks.html#syntax-for-code-cells
    sed -i 's/```python/```{code-cell} ipython3/g' $md_file
    # https://myst-nb.readthedocs.io/en/latest/authoring/text-notebooks.html#notebook-level-metadata
    sed -i '1s/^/---\nfile_format: mystnb\nkernelspec:\n  name: python3\njupytext:\n  text_representation:\n    extension: .md\n    format_name: myst\n    format_version: "0.13"\n    jupytext_version: 1.14.7\n---\n/' $md_file
    mystnb-to-jupyter $md_file
done
