/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file convert_types.h
 *
 * This header contains information related to conversion routines
 *
 */

#pragma once


/**
 * structure for compressed spare row -
 * @see https://en.wikipedia.org/wiki/Sparse_matrix
 */
typedef struct csr_gdf_ {
	void *			A;			/**< on-device:	single array (length nnz) that holds all the valid data fields (based on valid bitmap)	*/
	gdf_size_type *	IA;			/**< on-device:	compressed row indexes (size rows + 1)													*/
	int64_t *		JA;			/**< on-device:	column index (size of nnz)																*/
    gdf_dtype 		dtype;		/**< on-host:	the data type																			*/
	int64_t 		nnz;		/**< on-host:	the number of valid fields (nnz = number non-zero)										*/
	gdf_size_type	rows;		/**< on-host:	the number of rows																		*/
	gdf_size_type	cols;		/**< on-host:	the number of columns																	*/
} csr_gdf;



