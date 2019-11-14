/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <cudf/legacy/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/compare_column_wrappers.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <tests/groupby/sort/legacy/single_column_groupby_test.cuh>
#include <tests/groupby/common/legacy/type_info.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

static constexpr cudf::groupby::operators var_op{
    cudf::groupby::operators::VARIANCE};

static constexpr cudf::groupby::operators std_op{
    cudf::groupby::operators::STD};

template <typename KV>
struct SingleColumnVarStd : public GdfTest {
  using KeyType = typename KV::Key;
  using ValueType = typename KV::Value;
};

template <typename K>
struct key_accessor {
  std::vector<int> const& keys;
  K operator()(cudf::size_type i) { return K(keys[i]); }
};

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename K, typename V>
struct KV {
  using Key = K;
  using Value = V;
};

using TestingTypes =
    ::testing::Types< KV<int8_t, int8_t>, KV<int32_t, int32_t>,
                      KV<int64_t, int64_t>, KV<int32_t, float>,
                      KV<int32_t, double>, KV<cudf::category, int32_t>,
                      KV<cudf::date32, int8_t>, KV<cudf::date64, double>>;

// TODO: tests for cudf::bool8

using std_args =  cudf::groupby::sort::std_args;

TYPED_TEST_CASE(SingleColumnVarStd, TestingTypes);
 
TYPED_TEST(SingleColumnVarStd, TestVarStdPreSorted) {
  using K = typename SingleColumnVarStd<TypeParam>::KeyType;
  using V = typename SingleColumnVarStd<TypeParam>::ValueType;
  using R = cudf::test::expected_result_t<V, var_op>;
  using T = int;

  std::vector<T>   in_keys{1, 1, 1, 2, 2, 2, 2, 2};
  std::vector<V>   in_vals{0, 1, 2, 3, 4, 5, 6, 7};

                        //{1, 1, 1, 2, 2, 2, 2, 2}
  std::vector<T>  out_keys{1,       2,           };
                        //{0, 1, 2, 3, 4, 5, 6, 7}
  std::vector<R>  out_vals{1,       2.5          };

  int ddof = 1;
  cudf::test::single_column_groupby_test<var_op>({var_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys}),
      column_wrapper<V>(in_vals),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys}),
      column_wrapper<R>(out_vals, [](auto){ return true; })
  );
 
  cudf::test::single_column_groupby_test<std_op>({std_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys}),
      column_wrapper<V>(in_vals),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys}),
      column_wrapper<R>(out_vals.size(), [&](auto i) { return sqrt(out_vals[i]); },
        [](auto){ return true; })
  );
 
}  

TYPED_TEST(SingleColumnVarStd, TestVarStd) {
  using K = typename SingleColumnVarStd<TypeParam>::KeyType;
  using V = typename SingleColumnVarStd<TypeParam>::ValueType;
  using R = cudf::test::expected_result_t<V, var_op>;
  using T = int;

  std::vector<T>   in_keys{3, 2, 1, 1, 2, 3, 3, 2, 1};
  std::vector<V>   in_vals{1, 2, 3, 4, 4, 3, 2, 1, 0};

                        //{1, 1, 1, 2, 2, 2, 3, 3, 3}
  std::vector<T>  out_keys{1,       2,       3      };
                        //{0, 3, 4, 1, 2, 4, 1, 2, 3}
  std::vector<R>  out_vals{13./3,   7./3,    1      };

  int ddof = 1;
  cudf::test::single_column_groupby_test<var_op>({var_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys}),
      column_wrapper<V>(in_vals),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys}),
      column_wrapper<R>(out_vals, [](auto){ return true; }));

  cudf::test::single_column_groupby_test<std_op>({std_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys}),
      column_wrapper<V>(in_vals),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys}),
      column_wrapper<R>(out_vals.size(), [&](auto i) { return sqrt(out_vals[i]); },
        [](auto){ return true; }));
}
 

TYPED_TEST(SingleColumnVarStd, TestVarStdDifferentSizeGroups) {
  using K = typename SingleColumnVarStd<TypeParam>::KeyType;
  using V = typename SingleColumnVarStd<TypeParam>::ValueType;
  using R = cudf::test::expected_result_t<V, var_op>;
  using T = int;

  std::vector<T>   in_keys{1, 2, 3, 3, 2, 1, 0, 3, 0, 1, 0, 2, 3, 0, 3, 3, 2, 1, 0};
  std::vector<V>   in_vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

                        //{0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3}
  std::vector<T>  out_keys{0,             1,          2,          3               };
                        //{0, 5, 6, 8, 8, 0, 1, 5, 9, 1, 2, 4, 7, 2, 3, 3, 4, 6, 7}
  std::vector<R>  out_vals{10.8,          203./12,    7,          113./30         };

  int ddof = 1;
  cudf::test::single_column_groupby_test<var_op>({var_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys}),
      column_wrapper<V>(in_vals),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys}),
      column_wrapper<R>(out_vals, [](auto){ return true; }));

  cudf::test::single_column_groupby_test<std_op>({std_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys}),
      column_wrapper<V>(in_vals),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys}),
      column_wrapper<R>(out_vals.size(), [&](auto i) { return sqrt(out_vals[i]); },
        [](auto){ return true; }));
}
 

TYPED_TEST(SingleColumnVarStd, TestVarStdNullable) {
  using K = typename SingleColumnVarStd<TypeParam>::KeyType;
  using V = typename SingleColumnVarStd<TypeParam>::ValueType;
  using R = cudf::test::expected_result_t<V, var_op>;
  using T = int;

  std::vector<T>    in_keys   {1, 1, 1, 1, 1};
  std::vector<bool> key_valid {1, 0, 1, 1, 1};
  std::vector<V>    in_vals   {0, 1, 2, 3, 4};
  std::vector<bool> vals_valid{1, 1, 0, 1, 1};

                            //{1, -, 1, 1, 1}
  std::vector<T>    out_keys  {1,           };
                            //{0, 1, -, 3, 4}
  std::vector<R>    out_vals  {13./3,       };
  std::vector<bool> out_valids{1,           };

  int ddof = 1;
  cudf::test::single_column_groupby_test<var_op>({var_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys},
        [&](auto index) { return key_valid[index]; }),
      column_wrapper<V>(in_vals,
        [&](auto index) { return vals_valid[index]; }),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys},
        [](auto index) { return true; }),
      column_wrapper<R>(out_vals,
        [&](auto index) { return out_valids[index]; })
  );

  cudf::test::single_column_groupby_test<std_op>({std_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys},
        [&](auto index) { return key_valid[index]; }),
      column_wrapper<V>(in_vals,
        [&](auto index) { return vals_valid[index]; }),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys},
        [](auto index) { return true; }),
      column_wrapper<R>(out_vals.size(),
        [&](auto index) { return sqrt(out_vals[index]); },
        [&](auto index) { return out_valids[index]; })
  );
}

TYPED_TEST(SingleColumnVarStd, TestVarStdNullableZeroGroupSize) {
  using K = typename SingleColumnVarStd<TypeParam>::KeyType;
  using V = typename SingleColumnVarStd<TypeParam>::ValueType;
  using R = cudf::test::expected_result_t<V, var_op>;
  using T = int;

  std::vector<T>    in_keys   {1, 1, 1, 1, 1, 2, 2};
  std::vector<bool> key_valid {1, 0, 1, 1, 1, 1, 1};
  std::vector<V>    in_vals   {0, 1, 2, 3, 4, 5, 6};
  std::vector<bool> vals_valid{1, 1, 0, 1, 1, 0, 0};

                            //{1, -, 1, 1, 1, 2, 2}
  std::vector<T>    out_keys  {1,                2};
                            //{0, 1, -, 3, 4, -, 0}
  std::vector<R>    out_vals  {13./3,         0,  };
  std::vector<bool> out_valids{1,             0,  };

  int ddof = 1;
  cudf::test::single_column_groupby_test<var_op>({var_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys},
        [&](auto index) { return key_valid[index]; }),
      column_wrapper<V>(in_vals,
        [&](auto index) { return vals_valid[index]; }),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys},
        [](auto index) { return true; }),
      column_wrapper<R>(out_vals,
        [&](auto index) { return out_valids[index]; })
  );

  cudf::test::single_column_groupby_test<std_op>({std_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys},
        [&](auto index) { return key_valid[index]; }),
      column_wrapper<V>(in_vals,
        [&](auto index) { return vals_valid[index]; }),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys},
        [](auto index) { return true; }),
      column_wrapper<R>(out_vals.size(),
        [&](auto index) { return sqrt(out_vals[index]); },
        [&](auto index) { return out_valids[index]; })
  );
}

TYPED_TEST(SingleColumnVarStd, TestVarStdNullableZeroDDoFDivisor) {
  using K = typename SingleColumnVarStd<TypeParam>::KeyType;
  using V = typename SingleColumnVarStd<TypeParam>::ValueType;
  using R = cudf::test::expected_result_t<V, var_op>;
  using T = int;

  std::vector<T>    in_keys   {1, 1, 1, 1, 1, 3};
  std::vector<bool> key_valid {1, 0, 1, 1, 1, 1};
  std::vector<V>    in_vals   {0, 1, 2, 3, 4, 7};
  std::vector<bool> vals_valid{1, 1, 0, 1, 1, 1};

                            //{1, -, 1, 1, 1, 3}
  std::vector<T>    out_keys  {1,             3};
                            //{0, 1, -, 3, 4, 7}
  std::vector<R>    out_vals  {13./3,         0};
  std::vector<bool> out_valids{1,             0};

  int ddof = 1;
  cudf::test::single_column_groupby_test<var_op>({var_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys},
        [&](auto index) { return key_valid[index]; }),
      column_wrapper<V>(in_vals,
        [&](auto index) { return vals_valid[index]; }),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys},
        [](auto index) { return true; }),
      column_wrapper<R>(out_vals,
        [&](auto index) { return out_valids[index]; })
  );

  cudf::test::single_column_groupby_test<std_op>({std_op, std::make_unique<std_args>(ddof)},
      column_wrapper<K>(in_keys.size(), key_accessor<K>{in_keys},
        [&](auto index) { return key_valid[index]; }),
      column_wrapper<V>(in_vals,
        [&](auto index) { return vals_valid[index]; }),
      column_wrapper<K>(out_keys.size(), key_accessor<K>{out_keys},
        [](auto index) { return true; }),
      column_wrapper<R>(out_vals.size(),
        [&](auto index) { return sqrt(out_vals[index]); },
        [&](auto index) { return out_valids[index]; })
  );
}
