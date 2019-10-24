#pragma once

#include <cudf/types.h>
#include <cudf/utilities/error.hpp>

/**---------------------------------------------------------------------------*
 * @brief DEPRECATED error checking macro that verifies a condition evaluates to
 * true or returns an error-code.
 *
 * This macro is considered DEPRECATED and should not be used in any new
 * features.
 *
 * Instead, CUDF_EXPECTS() should be used.
 *
 *---------------------------------------------------------------------------**/
#define GDF_REQUIRE(F, S) \
  if (!(F)) return (S);

/**---------------------------------------------------------------------------*
 * @brief a version of GDF_REQUIRE for expressions of type `gdf_error` rather
 * than booleans
 *
 * This macro is sort-of DEPRECATED.
 *
 *---------------------------------------------------------------------------**/
#define GDF_TRY(_expression) do { \
    gdf_error _gdf_try_result = ( _expression ) ; \
    if (_gdf_try_result != GDF_SUCCESS) return _gdf_try_result ; \
} while(0)

/**---------------------------------------------------------------------------*
 * @brief Try evaluation an expression with a gdf_error type,
 * and throw an appropriate exception if it fails.
 *---------------------------------------------------------------------------**/
#define CUDF_TRY(_gdf_error_expression) do { \
    auto _evaluated = _gdf_error_expression; \
    if (_evaluated == GDF_SUCCESS) { break; } \
    throw cudf::logic_error( \
        ("cuDF error " + std::string(gdf_error_get_name(_evaluated)) + " at " \
       __FILE__ ":"  \
        CUDF_STRINGIFY(__LINE__) " evaluating " CUDF_STRINGIFY(#_gdf_error_expression)).c_str() ); \
} while(0)
