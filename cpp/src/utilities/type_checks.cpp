#include <algorithm>

#include <thrust/iterator/counting_iterator.h>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/dictionary.hpp>

#include <cudf/lists/list_view.cuh>
#include <cudf/structs/struct_view.hpp>

namespace cudf {

bool list_types_equal(column_view const& lhs, column_view const& rhs);
bool struct_types_equal(column_view const& lhs, column_view const& rhs);

struct types_equal_functor {
  template <typename T, typename U, typename... Args>
  bool operator()(Args&&...)
  {
    return false;
  }

  // Q: How does overload resolution makes sure this takes precedence over the one above?
  template <typename T,
            typename U,
            CUDF_ENABLE_IF(cudf::is_fixed_width<T>() and cudf::is_fixed_width<U>())>
  bool operator()(column_view const& lhs, column_view const& rhs)
  {
    return lhs.type() == rhs.type();
  }
};

template <>
bool types_equal_functor::operator()<dictionary32, dictionary32>(column_view const& lhs,
                                                                 column_view const& rhs)
{
  return lhs.num_children() > 0 and rhs.num_children() > 0
           ? lhs.child(dictionary_column_view::keys_column_index).type() ==
               rhs.child(dictionary_column_view::keys_column_index).type()
           : true;
}

template <>
bool types_equal_functor::operator()<string_view, string_view>(column_view const& lhs,
                                                               column_view const& rhs)
{
  return true;
}

template <>
bool types_equal_functor::operator()<list_view, list_view>(column_view const& lhs,
                                                           column_view const& rhs)
{
  return list_types_equal(lhs, rhs);
}

template <>
bool types_equal_functor::operator()<struct_view, struct_view>(column_view const& lhs,
                                                               column_view const& rhs)
{
  return struct_types_equal(lhs, rhs);
}

bool list_types_equal(column_view const& lhs, column_view const& rhs)
{
  auto const& lchild = lhs.child(lists_column_view::child_column_index);
  auto const& rchild = rhs.child(lists_column_view::child_column_index);
  return double_type_dispatcher(
    lchild.type(), rchild.type(), types_equal_functor{}, lchild, rchild);
}

bool struct_types_equal(column_view const& lhs, column_view const& rhs)
{
  return lhs.num_children() == rhs.num_children() and
         std::all_of(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(lhs.num_children()),
                     [&](auto i) {
                       auto const& lchild = lhs.child(0);
                       auto const& rchild = rhs.child(0);
                       return double_type_dispatcher(
                         lchild.type(), rchild.type(), types_equal_functor{}, lchild, rchild);
                     });
}

bool column_types_equal(column_view const& lhs, column_view const& rhs)
{
  return double_type_dispatcher(lhs.type(), rhs.type(), types_equal_functor{}, lhs, rhs);
}

// TODO
// bool scalar_types_equal(scalar const& lhs, scalar const& rhs) {
//     return true; //TODO
// }

}  // namespace cudf
