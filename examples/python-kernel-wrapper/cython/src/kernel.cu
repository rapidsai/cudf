#include <stdio.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>

static constexpr float mm_to_inches = 0.0393701;

__global__ void kernel_tenth_mm_to_inches(cudf::mutable_column_device_view column)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < column.size()) {
        column.element<int64_t>(i) = column.element<int64_t>(i) * (1/10) * mm_to_inches;
    }
}
