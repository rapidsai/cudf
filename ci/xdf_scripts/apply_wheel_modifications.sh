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

VERSION=${1}
CUDA_SUFFIX=${2}

# __init__.py versions
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/xdf/xdf/__init__.py

# pyproject.toml versions
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/xdf/pyproject.toml
