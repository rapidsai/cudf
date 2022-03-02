#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/sort2.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <vector>

struct NewRowOpTest : public cudf::test::BaseFixture {
};

TEST_F(NewRowOpTest, BasicStructTwoChild)
{
  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  const cudf::size_type n_rows{1 << 4};
  const cudf::size_type n_cols{2};

  // Create columns with values in the range [0,100)
  std::vector<column_wrapper> columns;
  columns.reserve(n_cols);
  std::generate_n(std::back_inserter(columns), n_cols, [&]() {
    auto elements = cudf::detail::make_counting_transform_iterator(
      0, [&](auto row) { return distribution(generator); });
    auto valids = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return i % 4 == 0 ? false : true; });
    return column_wrapper(elements, elements + n_rows, valids);
  });

  std::vector<std::unique_ptr<cudf::column>> cols;
  std::transform(columns.begin(), columns.end(), std::back_inserter(cols), [](column_wrapper& col) {
    return col.release();
  });

  auto make_struct = [&](std::vector<std::unique_ptr<cudf::column>> child_cols, int nullfreq) {
    // std::vector<bool> struct_validity;
    std::uniform_int_distribution<int> bool_distribution(0, 10 * (nullfreq));
    // std::generate_n(
    //   std::back_inserter(struct_validity), n_rows, [&]() { return bool_distribution(generator);
    //   });
    auto null_iter = cudf::detail::make_counting_transform_iterator(
      0, [&](int i) { return bool_distribution(generator); });

    cudf::test::structs_column_wrapper struct_col(std::move(child_cols));
    auto struct_ = struct_col.release();
    struct_->set_null_mask(cudf::test::detail::make_null_mask(null_iter, null_iter + n_rows));
    return struct_;
  };

  std::vector<std::unique_ptr<cudf::column>> s2_children;
  s2_children.push_back(std::move(cols[0]));
  s2_children.push_back(std::move(cols[1]));
  auto s2 = make_struct(std::move(s2_children), 1);

  cudf::test::print(s2->view());

  // // Create table view
  // auto input = cudf::table_view({struct_col});
  auto input = cudf::table_view({s2->view()});

  auto result1 = cudf::sorted_order(input);
  cudf::test::print(result1->view());
  auto result2 = cudf::detail::experimental::sorted_order2(input);
  cudf::test::print(result2->view());
  cudf::test::expect_columns_equal(result1->view(), result2->view());
}

TEST_F(NewRowOpTest, DeepStruct)
{
  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);

  const cudf::size_type n_rows{1 << 6};
  const cudf::size_type n_cols{1};
  const cudf::size_type depth{8};

  // Create columns with values in the range [0,100)
  std::vector<column_wrapper> columns;
  columns.reserve(n_cols);
  std::generate_n(std::back_inserter(columns), n_cols, [&, n_rows]() {
    auto elements = cudf::detail::make_counting_transform_iterator(
      0, [&](auto row) { return distribution(generator); });
    auto valids = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return i % 3 == 0 ? false : true; });
    return column_wrapper(elements, elements + n_rows, valids);
  });

  std::vector<std::unique_ptr<cudf::column>> cols;
  std::transform(columns.begin(), columns.end(), std::back_inserter(cols), [](column_wrapper& col) {
    return col.release();
  });

  std::vector<std::unique_ptr<cudf::column>> child_cols = std::move(cols);
  // Lets add some layers
  for (int i = 0; i < depth; i++) {
    std::vector<bool> struct_validity;
    std::uniform_int_distribution<int> bool_distribution(0, 10 * (i + 1));
    std::generate_n(
      std::back_inserter(struct_validity), n_rows, [&]() { return bool_distribution(generator); });
    cudf::test::structs_column_wrapper struct_col(std::move(child_cols), struct_validity);
    child_cols = std::vector<std::unique_ptr<cudf::column>>{};
    child_cols.push_back(struct_col.release());
  }

  cudf::test::print(child_cols[0]->view());

  // // Create table view
  // auto input = cudf::table_view({struct_col});
  auto input = cudf::table(std::move(child_cols));

  // auto sliced_input = cudf::slice(input, {7, input.num_rows() - 12});

  auto result1 = cudf::sorted_order(input);
  cudf::test::print(result1->view());
  auto result2 = cudf::detail::experimental::sorted_order2(input);
  cudf::test::print(result2->view());
  cudf::test::expect_columns_equal(result1->view(), result2->view());
}

TEST_F(NewRowOpTest, SampleStructTest)
{
  using Type           = int;
  using column_wrapper = cudf::test::fixed_width_column_wrapper<Type>;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 10);

  const cudf::size_type n_rows{1 << 6};
  const cudf::size_type n_cols{6};

  // Create columns with values in the range [0,100)
  std::vector<column_wrapper> columns;
  columns.reserve(n_cols);
  std::generate_n(std::back_inserter(columns), n_cols, [&]() {
    auto elements = cudf::detail::make_counting_transform_iterator(
      0, [&](auto row) { return distribution(generator); });
    int start   = distribution(generator);
    auto valids = cudf::detail::make_counting_transform_iterator(
      0, [&](auto i) { return (i + start) % 7 == 0 ? false : true; });
    return column_wrapper(elements, elements + n_rows, valids);
  });

  std::vector<std::unique_ptr<cudf::column>> cols;
  std::transform(columns.begin(), columns.end(), std::back_inserter(cols), [](column_wrapper& col) {
    return col.release();
  });

  auto make_struct = [&](std::vector<std::unique_ptr<cudf::column>> child_cols, int nullfreq) {
    std::vector<bool> struct_validity;
    std::uniform_int_distribution<int> bool_distribution(0, 10 * (nullfreq));
    auto null_iter = cudf::detail::make_counting_transform_iterator(
      0, [&](int i) { return bool_distribution(generator); });

    cudf::test::structs_column_wrapper struct_col(std::move(child_cols));
    auto struct_ = struct_col.release();
    struct_->set_null_mask(cudf::test::detail::make_null_mask(null_iter, null_iter + n_rows));
    return struct_;
  };

  std::vector<std::unique_ptr<cudf::column>> s2_children;
  s2_children.push_back(std::move(cols[0]));
  s2_children.push_back(std::move(cols[1]));
  auto s2 = make_struct(std::move(s2_children), 1);

  std::vector<std::unique_ptr<cudf::column>> s1_children;
  s1_children.push_back(std::move(s2));
  s1_children.push_back(std::move(cols[2]));
  auto s1 = make_struct(std::move(s1_children), 2);

  cudf::test::print(s1->view());

  std::vector<std::unique_ptr<cudf::column>> s22_children;
  s22_children.push_back(std::move(cols[3]));
  s22_children.push_back(std::move(cols[4]));
  auto s22 = make_struct(std::move(s22_children), 1);

  std::vector<std::unique_ptr<cudf::column>> s12_children;
  s12_children.push_back(std::move(cols[5]));
  s12_children.push_back(std::move(s22));
  auto s12 = make_struct(std::move(s12_children), 2);

  cudf::test::print(s12->view());

  // // Create table view
  // auto input = cudf::table_view({struct_col});
  auto input = cudf::table_view({s1->view(), s12->view()});

  auto result1 = cudf::sorted_order(input);
  cudf::test::print(result1->view());
  auto result2 = cudf::detail::experimental::sorted_order2(input);
  cudf::test::print(result2->view());
  cudf::test::expect_columns_equal(result1->view(), result2->view());

  std::vector<cudf::order> col_order       = {cudf::order::DESCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_order = {cudf::null_order::BEFORE, cudf::null_order::AFTER};
  result1                                  = cudf::sorted_order(input, col_order, null_order);
  result2 = cudf::detail::experimental::sorted_order2(input, col_order, null_order);
  cudf::test::print(result1->view());
  cudf::test::print(result2->view());
  cudf::test::expect_columns_equal(result1->view(), result2->view());
}

TEST_F(NewRowOpTest, List)
{
  using lcw = cudf::test::lists_column_wrapper<uint64_t>;
  lcw col{
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
    {{1, 2, 3}, {}, {4, 5}, {0, 6, 0}},
    {{1, 2}, {3}, {4, 5}, {0, 6, 0}},
    {{7, 8}, {}},
    lcw{lcw{}, lcw{}, lcw{}},
    lcw{lcw{}},
    {lcw{10}},
    lcw{},
  };

  auto expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{8, 6, 5, 3, 0, 1, 2, 4, 7};
  auto result = cudf::detail::experimental::sorted_order2(cudf::table_view({col}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, *result);
}

CUDF_TEST_PROGRAM_MAIN()
