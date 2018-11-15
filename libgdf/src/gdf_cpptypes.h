#ifndef GDF_CPPTYPES_H
#define GDF_CPPTYPES_H

#include <gdf/cffi/types.h>

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Placeholder type definitions for the non-primitive types for use in
 * conjunction with the gdf_type_dispatcher.
 *
 * The purpose of these type definitions is to allow distinguishing columns with
 * different gdf_dtype types, but have the same underlying type. For example,
 * the underlying type of both GDF_DATE32 and GDF_INT32 is int32_t. However, if
 * one wished to specialize a functor invoked with the gdf_type_dispatcher to handle
 * GDF_DATE32 different from GDF_INT32, that would not be possible without these
 * type definitions.
 *
 * In the long term, these types should be updated such that they provide appropriate
 * operators/member functions that implement the desired behavior for operating on
 * elements of these types.
 *
 */
/* ----------------------------------------------------------------------------*/
namespace gdf
{
  struct category
  {
    gdf_category value;
  };

  struct timestamp
  {
    gdf_timestamp value;
  };

  struct date32
  {
    gdf_date32 value;
  };

  struct date64
  {
    gdf_date64 value;
  };

  // TODO Add a type for GDF_STRING?
}

#endif
