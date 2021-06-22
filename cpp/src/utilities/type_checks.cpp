#include <algorithm>

#include <thrust/iterator/counting_iterator.h>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>

namespace cudf {

bool column_types_equal(column_view const& lhs, column_view const& rhs)
{
  if (lhs.type().id() == type_id::DICTIONARY32 and rhs.type().id() == type_id::DICTIONARY32) {
    auto const kidx = dictionary_column_view::keys_column_index;
    return lhs.num_children() > 0 and rhs.num_children() > 0
             ? lhs.child(kidx).type() == rhs.child(kidx).type()
             : true;
  } else if (lhs.type().id() == type_id::LIST and rhs.type().id() == type_id::LIST) {
    auto const& ci = lists_column_view::child_column_index;
    return column_types_equal(lhs.child(ci), rhs.child(ci));
  } else if (lhs.type().id() == type_id::STRUCT and rhs.type().id() == type_id::STRUCT) {
    return lhs.num_children() == rhs.num_children() and
           std::all_of(thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(lhs.num_children()),
                       [&](auto i) { return column_types_equal(lhs.child(i), rhs.child(i)); });
  } else {
    return lhs.type() == rhs.type();
  }
}

}  // namespace cudf
