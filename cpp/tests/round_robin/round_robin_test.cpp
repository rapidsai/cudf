#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/merge.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <cudf/utilities/legacy/wrapper_types.hpp>

#include <cassert>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <initializer_list>

#include <gtest/gtest.h>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

template <typename T>
class RoundRobinTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(RoundRobinTest, cudf::test::FixedWidthTypes);
