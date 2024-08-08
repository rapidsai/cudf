#include "dbgen.hpp"

int main()
{
  auto region =
    cudf::generate_region(cudf::get_default_stream(), rmm::mr::get_current_device_resource());
}
