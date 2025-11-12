# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool

from .types cimport DataType


cpdef bool is_relationally_comparable(DataType typ)
cpdef bool is_equality_comparable(DataType typ)
cpdef bool is_numeric(DataType typ)
cpdef bool is_index_type(DataType typ)
cpdef bool is_unsigned(DataType typ)
cpdef bool is_integral(DataType typ)
cpdef bool is_integral_not_bool(DataType typ)
cpdef bool is_floating_point(DataType typ)
cpdef bool is_boolean(DataType typ)
cpdef bool is_timestamp(DataType typ)
cpdef bool is_fixed_point(DataType typ)
cpdef bool is_duration(DataType typ)
cpdef bool is_chrono(DataType typ)
cpdef bool is_dictionary(DataType typ)
cpdef bool is_fixed_width(DataType typ)
cpdef bool is_compound(DataType typ)
cpdef bool is_nested(DataType typ)
cpdef bool is_bit_castable(DataType source, DataType target)
