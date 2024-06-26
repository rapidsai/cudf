
#include <iostream>
#include <memory>
#include <vector>
#include <cudf/io/parquet.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/groupby.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/sorting.hpp>
#include <cudf/reduction.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include "utils.hpp"

/*
select
    sum(l_extendedprice * l_discount) as revenue
from
    lineitem
where
        l_shipdate >= date '1994-01-01'
    and l_shipdate < date '1995-01-01'
    and l_discount >= 0.05 
    and l_discount <= 0.07
    and l_quantity < 24;
*/

std::unique_ptr<cudf::table> scan_filter_project() {
    std::string lineitem = "/home/jayjeetc/tpch_sf1/lineitem/part-0.parquet";
    auto source = cudf::io::source_info(lineitem);
    auto builder = cudf::io::parquet_reader_options_builder(source);

    std::vector<std::string> projection_cols = {
        "l_extendedprice",
        "l_discount",
        "l_shipdate",
        "l_quantity" 
    };

    auto l_extendedprice = cudf::ast::column_reference(0);
    auto l_discount = cudf::ast::column_reference(1);
    auto l_shipdate = cudf::ast::column_reference(2);
    auto l_quantity = cudf::ast::column_reference(3);

    auto date_scalar_a = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
    auto date_literal_a = cudf::ast::literal(date_scalar_a);

    auto date_scalar_b = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
    auto date_literal_b = cudf::ast::literal(date_scalar_b);

    auto pred_a = cudf::ast::operation(
        cudf::ast::ast_operator::GREATER_EQUAL,
        l_shipdate,
        date_literal_a
    );

    auto pred_b = cudf::ast::operation(
        cudf::ast::ast_operator::LESS,
        l_shipdate,
        date_literal_b
    );

    auto scalar_a = cudf::numeric_scalar<double>(0.05);
    auto literal_a = cudf::ast::literal(scalar_a);
    auto scalar_b = cudf::numeric_scalar<double>(0.07);
    auto literal_b = cudf::ast::literal(scalar_b);
    auto scalar_c = cudf::numeric_scalar<double>(24.0);
    auto literal_c = cudf::ast::literal(scalar_c);

    auto pred_c = cudf::ast::operation(
        cudf::ast::ast_operator::GREATER_EQUAL,
        l_discount,
        literal_a
    );

    auto pred_d = cudf::ast::operation(
        cudf::ast::ast_operator::LESS_EQUAL,
        l_discount,
        literal_b
    );

    auto pred_e = cudf::ast::operation(
        cudf::ast::ast_operator::LESS,
        l_quantity,
        literal_c
    );

    auto pred_ab = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        pred_a,
        pred_b
    );

    auto pred_cd = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        pred_c,
        pred_d
    );

    auto pred_abcd = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        pred_ab,
        pred_cd
    );

    builder.columns(projection_cols);
    // builder.filter(pred_ab);

    auto options = builder.build();
    auto result = cudf::io::read_parquet(options);
    return std::move(result.tbl);
}

std::unique_ptr<cudf::table> compute_result_table(std::unique_ptr<cudf::table>& table) {
    auto l_extendedprice = table->view().column(0);
    auto l_discount = table->view().column(1);

    auto extendedprice_mul_discount = cudf::binary_operation(
        l_extendedprice,
        l_discount,
        cudf::binary_operator::MUL,
        cudf::data_type{cudf::type_id::DECIMAL64}
    );
    
    auto const sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    auto sum = cudf::reduce(extendedprice_mul_discount->view(), *sum_agg, extendedprice_mul_discount->type());

    auto decimal_sum = static_cast<cudf::scalar_type_t<numeric::decimal64>*>(sum.get());
    double decimal_sum_value = decimal_sum->value();

    std::cout << "Sum: " << decimal_sum_value << std::endl;

    cudf::data_type type = cudf::data_type{cudf::type_id::DECIMAL64};
    cudf::size_type len = 1;
    rmm::device_buffer data_buffer = get_device_buffer_from_value(decimal_sum_value);
    rmm::device_buffer null_mask_buffer = get_empty_device_buffer();
    auto col = std::make_unique<cudf::column>(
        type, len, std::move(data_buffer), std::move(null_mask_buffer), 0);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(col));
    auto result_table = std::make_unique<cudf::table>(std::move(columns));
    return result_table;
}

int main() {
    auto t1 = scan_filter_project();
    auto result_table = compute_result_table(t1);
    auto result_metadata = create_table_metadata({"revenue"});
    std::string result_filename = "q6.parquet";
    write_parquet(result_table, result_metadata, result_filename);
    return 0;
}
