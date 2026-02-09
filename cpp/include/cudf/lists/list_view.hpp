/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

/**
 * @file list_view.hpp
 * @brief Class definition for cudf::list_view.
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief A non-owning, immutable view of device data that represents
 * a list of elements of arbitrary type (including further nested lists).
 */
class list_view {};

}  // namespace CUDF_EXPORT cudf
