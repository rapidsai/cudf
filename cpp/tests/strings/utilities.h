

#include <cudf/column/column.hpp>

#include <vector>

namespace cudf {
namespace test {

/**---------------------------------------------------------------------------*
 * @brief Utility for creating a strings column from a vector of host strings
 *
 * @param h_strings Pointer to null-terminated, UTF-8 encode chars arrays.
 * @return column instance of type STRING
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> create_strings_column( const std::vector<const char*>& h_strings );

}  // namespace test
}  // namespace cudf