#include "pair_rtti.cuh"

template <class IndexT>
PairRTTI<IndexT>::PairRTTI(const SideGroup &        left_side_group,
                           const SideGroup &        right_side_group,
                           const gdf_size_type      size,
                           const std::int8_t *const asc_desc_flags)
    : left_side_group_{left_side_group}, right_side_group_{right_side_group},
      size_{size}, asc_desc_flags_{asc_desc_flags} {}

template class PairRTTI<gdf_size_type>;
