
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>

namespace cudf {

/**
 * @brief Compares the type of two `column_view`s
 *
 * This function returns true if the type of `lhs` equals that of `rhs`.
 * - For fixed point types, the scale is compared.
 * - For dictionary types, the type of the keys are compared if both are
 *   non-empty columns.
 * - For lists types, the type of child columns are compared recursively.
 * - For struct types, the type of each field are compared in order.
 * - For all other types, the `id` of `data_type` is compared.
 *
 * @param lhs The first `column_view` to compare
 * @param rhs The second `column_view` to compare
 * @return boolean
 */
bool column_types_equal(column_view const& lhs, column_view const& rhs);

}  // namespace cudf
