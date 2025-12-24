#include <cudf/utilities/memory_resource.hpp>
#include <cudf/column/column_factories.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

int main() {
    // Test single-argument constructor
    rmm::cuda_memory_resource cuda_mr;
    cudf::memory_resources resources1(&cuda_mr);

    // Test two-argument constructor
    rmm::cuda_memory_resource output_mr;
    rmm::cuda_memory_resource temp_mr;
    cudf::memory_resources resources2(&output_mr, &temp_mr);

    // Test accessors
    auto out_mr = resources2.get_output_mr();
    auto tmp_mr = resources2.get_temporary_mr();

    // Test implicit conversion
    auto col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        10,
        cudf::mask_state::UNALLOCATED,
        cudf::get_default_stream(),
        resources1  // implicit conversion from device_async_resource_ref works
    );

    return 0;
}
