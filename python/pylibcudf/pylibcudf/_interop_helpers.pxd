# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.libcudf.interop cimport column_metadata

cdef void _release_schema(object schema_capsule) noexcept

cdef void _release_array(object array_capsule) noexcept

cdef void _release_device_array(object array_capsule) noexcept

cdef column_metadata _metadata_to_libcudf(metadata)
