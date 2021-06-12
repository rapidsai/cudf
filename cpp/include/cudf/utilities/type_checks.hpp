
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>

namespace cudf {

/**
 * @brief Compares the type of two `column_view`s
 *
 * For nested columns, this function recursively checks that all
 * children of `lhs` matches the type of `rhs`.
 *
 * @param lhs The first `column_view` to compare
 * @param rhs The second `column_view` to compare
 * @return true `lhs` is not equal to `rhs`
 * @return false `lhs` is equal to `rhs`
 */
bool column_types_equal(column_view const& lhs, column_view const& rhs);

bool scalar_types_equal(scalar const& lhs, scalar const& rhs);

}  // namespace cudf
