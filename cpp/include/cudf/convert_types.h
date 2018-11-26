
#pragma once
//
// Compressed Saprse Row - see https://en.wikipedia.org/wiki/Sparse_matrix
//
typedef struct csr_gdf_ {
	void *			A;			// on-device:	single array (length nnz) that holds all the valid data fields (based on valid bitmap)
	gdf_size_type *	IA;			// on-device:	compressed row indexes (size rows + 1)
	int64_t *		JA;			// on-device:	column index (size of nnz)
    gdf_dtype 		dtype;		// on-host:		the data type
	int64_t 		nnz;		// on-host:		the number of valid fields (nnz = number non-zero)
	gdf_size_type	rows;		// on-host:		the number of rows
	gdf_size_type	cols;		// on-host:		the number of columns
} csr_gdf;



