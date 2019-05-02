# Copyright (c) 2019, NVIDIA CORPORATION.
	
	# cython: profile=False
	# distutils: language = c++
	# cython: embedsignature = True
	# cython: language_level = 3
	
	from cudf.bindings.cudf_cpp cimport *
	#from cudf.bindings.types cimport table as cudf_table

cdef extern from "stream_compaction.hpp" namespace "cudf" nogil:

    cdef gdf_column apply_boolean_mask(gdf_column const *input,
                                       gdf_column const *boolean_mask) except +
