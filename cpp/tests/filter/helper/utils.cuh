
#ifndef GDF_TEST_UTILS
#define GDF_TEST_UTILS

#include <iostream>
#include <cudf.h>
#include <cudf/functions.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <string>
#include <functional>
#include <vector>
#include <tuple>
#include <rmm/rmm.h>

template <typename gdf_type>
inline gdf_dtype gdf_enum_type_for()
{
    return GDF_invalid;
}

template <>
inline gdf_dtype gdf_enum_type_for<int8_t>()
{
    return GDF_INT8;
}

template <>
inline gdf_dtype gdf_enum_type_for<int16_t>()
{
    return GDF_INT16;
}

template <>
inline gdf_dtype gdf_enum_type_for<int32_t>()
{
    return GDF_INT32;
}

template <>
inline gdf_dtype gdf_enum_type_for<int64_t>()
{
    return GDF_INT64;
}

template <>
inline gdf_dtype gdf_enum_type_for<float>()
{
    return GDF_FLOAT32;
}

template <>
inline gdf_dtype gdf_enum_type_for<double>()
{
    return GDF_FLOAT64;
}

inline auto get_number_of_bytes_for_valid (size_t column_size) -> size_t {
    return sizeof(gdf_valid_type) * (column_size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
}

auto print_binary(gdf_valid_type n, int size = 8) -> void ;

auto chartobin(gdf_valid_type n, int size = 8) -> std::string;

gdf_size_type count_zero_bits(gdf_valid_type *valid, size_t column_size);

auto delete_gdf_column(gdf_column * column) -> void; 
 
auto gen_gdf_valid(size_t column_size, size_t init_value) -> gdf_valid_type *;

gdf_valid_type * get_gdf_valid_from_device(gdf_column* column) ;

std::string gdf_valid_to_str(gdf_valid_type *valid, size_t column_size);

template <typename RawType, typename PointerType>
auto init_device_vector(gdf_size_type num_elements) -> std::tuple<RawType *, thrust::device_ptr<PointerType>>
{
    RawType *device_pointer;
    rmmError_t rmm_error = RMM_ALLOC((void **)&device_pointer, sizeof(PointerType) * num_elements, 0);
    EXPECT_TRUE(rmm_error == RMM_SUCCESS);
    thrust::device_ptr<PointerType> device_wrapper = thrust::device_pointer_cast((PointerType *)device_pointer);
    return std::make_tuple(device_pointer, device_wrapper);
}


template <typename ValueType = int8_t>
ValueType* get_gdf_data_from_device(gdf_column* column) {
    ValueType* host_out = new ValueType[column->size];
    cudaMemcpy(host_out, column->data, sizeof(ValueType) * column->size, cudaMemcpyDeviceToHost);
    return host_out;
}

template <typename ValueType = int8_t>
std::string gdf_data_to_str(void *data, size_t column_size)
{
    std::string response;
    for (size_t i = 0; i < column_size; i++)
    {
        auto result = std::to_string(*((ValueType*)(data) + i));
        response += std::string(result);
    }
    return response;
}


template <typename ValueType = int8_t>
gdf_column convert_to_device_gdf_column (gdf_column *column) {
    size_t column_size = column->size;
    char *raw_pointer;
    thrust::device_ptr<ValueType> device_pointer;
    std::tie(raw_pointer, device_pointer) = init_device_vector<char, ValueType>(column_size);
 
    void* host_out = column->data;
    cudaMemcpy(raw_pointer, host_out, sizeof(ValueType) * column->size, cudaMemcpyHostToDevice);

    gdf_valid_type *host_valid = column->valid;
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);

    gdf_valid_type *valid_value_pointer;
    RMM_ALLOC((void **)&valid_value_pointer, n_bytes, 0);
    cudaMemcpy(valid_value_pointer, host_valid, n_bytes, cudaMemcpyHostToDevice);

    gdf_column output;
    gdf_column_view_augmented(&output, (void *)raw_pointer, valid_value_pointer, column_size, column->dtype, column->null_count);
    return output;
}

template <typename ValueType = int8_t>
gdf_column convert_to_host_gdf_column (gdf_column *column) {
    auto host_out = get_gdf_data_from_device(column);
    auto host_valid_out = get_gdf_valid_from_device(column);
    
    auto output = *column;
    output.data = host_out;
    output.valid = host_valid_out;
    return output;
}


template <typename ValueType = int8_t>
auto print_column(gdf_column * column) -> void {
    auto host_out = get_gdf_data_from_device<ValueType>(column);
    auto host_valid_out = get_gdf_valid_from_device(column);
    std::cout<<"Printing Column\t null_count:" << column->null_count << "\t type " << column->dtype <<  std::endl;
    int  n_bytes =  sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
    // for(int i = 0; i < column->size; i++) {
    //     int col_position =  i / 8;
    //     int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
    //     int bit_offset =  (length_col - 1) - (i % 8);
    //     if (sizeof(ValueType) == 1) {
    //         std::cout << "host_out[" << i << "] = " << ((int)host_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;

    //     } else {
    //         std::cout << "host_out[" << i << "] = " << ((ValueType)host_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
    //     }
    // }
    for (int i = 0; i < n_bytes; i++) {
        int length = n_bytes != i+1 ? GDF_VALID_BITSIZE : column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        print_binary(host_valid_out[i], length);
    }
    delete[] host_out;
    delete[] host_valid_out;
    std::cout<<std::endl<<std::endl;
}
template <typename ValueType = int8_t>
gdf_column gen_gdb_column(size_t column_size, ValueType init_value)
{
    char *raw_pointer;
    auto gdf_enum_type_value =  gdf_enum_type_for<ValueType>();
    thrust::device_ptr<ValueType> device_pointer;
   // std::cout << "0. gen_gdb_column\n";     
    std::tie(raw_pointer, device_pointer) = init_device_vector<char, ValueType>(column_size);
   // std::cout << "1. gen_gdb_column\n"; 
    
    using thrust::detail::make_normal_iterator;
    thrust::fill(make_normal_iterator(device_pointer), make_normal_iterator(device_pointer + column_size), init_value);
    //std::cout << "2. gen_gdb_column\n"; 
    
    gdf_valid_type *host_valid = gen_gdf_valid(column_size, init_value);
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);

    gdf_valid_type *valid_value_pointer;
    RMM_ALLOC((void **)&valid_value_pointer, n_bytes, 0);
    cudaMemcpy(valid_value_pointer, host_valid, n_bytes, cudaMemcpyHostToDevice);
   // std::cout << "3. gen_gdb_column\n"; 
    
    gdf_column output;
    auto zero_bits = output.null_count = count_zero_bits(host_valid, column_size);

    gdf_column_view_augmented(&output,
                             (void *)raw_pointer, valid_value_pointer,
                             column_size,
                             gdf_enum_type_value,
                             zero_bits);
    //std::cout << "4. gen_gdb_column\n"; 
    
    delete []host_valid;
    return output;
}

template <typename LeftValueType = int8_t, typename RightValueType = int8_t>
void check_column_for_stencil_operation(gdf_column *column, gdf_column *stencil, gdf_column *output_op) {
    gdf_column host_column = convert_to_host_gdf_column(column);
    gdf_column host_stencil = convert_to_host_gdf_column(stencil);
    gdf_column host_output_op = convert_to_host_gdf_column(output_op);

    EXPECT_EQ(host_column.size, host_stencil.size);
    //EXPECT_EQ(host_column.dtype == host_output_op.dtype);  // it must have the same type

    
    int  n_bytes =  sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
    std::vector<int> indexes;
    for(gdf_size_type i = 0; i < host_stencil.size; i++) {
        int col_position =  i / 8;
        int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : column->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        int bit_offset =  (length_col - 1) - (i % 8);
        bool valid = ((host_stencil.valid[col_position] >> bit_offset ) & 1) != 0;
         if ( (int)( ((int8_t *)host_stencil.data)[i] ) == 1 && valid ) {
             indexes.push_back(i);
         }
    }
    
    for(size_t i = 0; i < indexes.size(); i++) 
    {
        int index = indexes[i];
        LeftValueType value = ((LeftValueType *)(host_column.data))[index];
        std::cout << "filtered values: " << index  << "** "  << "\t value: " << (int)value << std::endl;
        EXPECT_EQ( ((RightValueType*)host_output_op.data)[i], value);
        
        int col_position =  i / 8;
        int length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : output_op->size - GDF_VALID_BITSIZE * (n_bytes - 1);
        int bit_offset =  (length_col - 1) - (i % 8);
        bool valid = ((host_output_op.valid[col_position] >> bit_offset ) & 1) != 0;
        EXPECT_EQ(valid, true);
    }
}

template <typename LeftValueType, typename RightValueType>
void check_column_for_comparison_operation(gdf_column *lhs, gdf_column *rhs, gdf_column *output, gdf_comparison_operator gdf_operator)
{
    {
        auto lhs_valid = get_gdf_valid_from_device(lhs);
        auto rhs_valid = get_gdf_valid_from_device(rhs);
        auto output_valid = get_gdf_valid_from_device(output);
        
        size_t n_bytes = get_number_of_bytes_for_valid(output->size);

        EXPECT_EQ(lhs->size, rhs->size); 
        
        for(gdf_size_type i = 0; i < output->size; i++) {
            size_t col_position =  i / 8;
            size_t length_col = n_bytes != col_position+1 ? GDF_VALID_BITSIZE : output->size - GDF_VALID_BITSIZE * (n_bytes - 1);
            size_t bit_offset =  (length_col - 1) - (i % 8);
            
            EXPECT_EQ( ((lhs_valid[col_position] >> bit_offset ) & 1) & ((rhs_valid[col_position] >> bit_offset ) & 1),
            ((output_valid[col_position] >> bit_offset ) & 1) );
        }
        
        delete[] lhs_valid;
        delete[] rhs_valid;
        delete[] output_valid;    
    }

    {
        auto lhs_data = get_gdf_data_from_device<LeftValueType>(lhs);
        auto rhs_data = get_gdf_data_from_device<RightValueType>(rhs);
        auto output_data = get_gdf_data_from_device<int8_t>(output);

        EXPECT_EQ(lhs->size, rhs->size); 
        for(gdf_size_type i = 0; i < lhs->size; i++)
        {
            EXPECT_EQ(lhs_data[i] == rhs_data[i] ? 1 : 0,  output_data[i]);              
        }
        
        delete[] lhs_data;
        delete[] rhs_data;
        delete[] output_data;    
    }
    
}

template <typename ValueType = int8_t>
void check_column_for_concat_operation(gdf_column *lhs, gdf_column *rhs, gdf_column *output)
{
    {
        auto lhs_valid = get_gdf_valid_from_device(lhs);
        auto rhs_valid = get_gdf_valid_from_device(rhs);
        auto output_valid = get_gdf_valid_from_device(output);
        
        auto computed = gdf_valid_to_str(output_valid, output->size);
        auto expected = gdf_valid_to_str(lhs_valid, lhs->size) + gdf_valid_to_str(rhs_valid, rhs->size);

        //std::cout << "computed: " <<  computed << std::endl;
        //std::cout << "expected: " << expected << std::endl;

        delete[] lhs_valid;
        delete[] rhs_valid;
        delete[] output_valid;    
        EXPECT_EQ(computed, expected);
    }

    {
        auto lhs_data = get_gdf_data_from_device<ValueType>(lhs);
        auto rhs_data = get_gdf_data_from_device<ValueType>(rhs);
        auto output_data = get_gdf_data_from_device<ValueType>(output);

        auto computed = gdf_data_to_str<ValueType>(output_data, output->size);
        auto expected = gdf_data_to_str<ValueType>(lhs_data, lhs->size) + gdf_data_to_str<ValueType>(rhs_data, rhs->size);
        delete[] lhs_data;
        delete[] rhs_data;
        delete[] output_data;    
        EXPECT_EQ(computed, expected);    
    }
    
}


#endif // GDF_TEST_UTILS
