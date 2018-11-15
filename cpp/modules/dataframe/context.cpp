#include "cudf.h"

gdf_error gdf_context_view(gdf_context *context, int flag_sorted, gdf_method flag_method,
                           int flag_distinct, int flag_sort_result, int flag_sort_inplace) {
    context->flag_sorted   = flag_sorted;
    context->flag_method   = flag_method;
    context->flag_distinct = flag_distinct;
    context->flag_sort_result = flag_sort_result;
    context->flag_sort_inplace = flag_sort_inplace;
    return GDF_SUCCESS;
}

