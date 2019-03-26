#include <cstdlib>
#include <iostream>
#include <assert.h>

// type definition for the aggregation function
typedef int (*custom_agg_funtype)(const int*, const int);

// rolling window function
void cudf_rolling(int *output, const int *input, const int size, const int window, custom_agg_funtype agg);

