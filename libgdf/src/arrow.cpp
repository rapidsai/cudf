#include <gdf/gdf.h>
#include <gdf/errorutils.h>
#include <arrow/api.h>
#include <gdf/arrow.hpp>
#include <arrow/gpu/cuda_api.h>

using namespace arrow;
using namespace arrow::gpu;

gdf_dtype arrow_dtype_to_gdf_dtype(std::shared_ptr<DataType> type){
  gdf_dtype result;
  switch(type->id())
    {
    case arrow_type::INT8:
      result =  gdf_dtype::GDF_INT8;
      break;
    case arrow_type::INT16:
      result =  gdf_dtype::GDF_INT16;
      break;
    case arrow_type::INT32:
      result =  gdf_dtype::GDF_INT32;
      break;
    case arrow_type::INT64:
      result =  gdf_dtype::GDF_INT64;
      break;
    case arrow_type::FLOAT:
      result =  gdf_dtype::GDF_FLOAT32;
      break;
    case arrow_type::DOUBLE:
      result =  gdf_dtype::GDF_FLOAT64;
      break;
    default:
      //TODO throw exception if we don't switch?
      //TODO handle other dtypes
      std::cout << "BAD\n";
    }
  return result;
}
int32_t byte_width(std::shared_ptr<DataType> type){
  int32_t result = -1;
  switch(type->id())
    {
    case arrow_type::INT8:
      result = 1;
      break;
    case arrow_type::INT16:
      result =  2;
      break;
    case arrow_type::INT32:
      result =  4;
      break;
    case arrow_type::INT64:
      result =  8;
      break;
    case arrow_type::FLOAT:
      result =  4;
      break;
    case arrow_type::DOUBLE:
      result =  8;
      break;
    default:
      //TODO throw exception if we don't switch?
      //TODO handle other dtypes
      std::cout << "BAD\n";
    }
  return result;
}
std::shared_ptr<DataType> gdf_dtype_to_arrow_dtype(gdf_dtype type){
  std::shared_ptr<DataType> result;
  switch(type)
    {
    case gdf_dtype::GDF_INT8:
      result = int8();
      break;
    case gdf_dtype::GDF_INT16:
      result = int16();
      break;
    case gdf_dtype::GDF_INT32:
      result = int32();
      break;
    case gdf_dtype::GDF_INT64:
      result = int64();
      break;
    case gdf_dtype::GDF_FLOAT32:
      result = float32();
      break;
    case gdf_dtype::GDF_FLOAT64:
      result = float64();
      break;            
    default:
      //TODO throw exception if we don't switch?
      //TODO handle other dtypes
      std::cout << "BAD\n";
    }
  return result;
}

void* arrow_to_gdf(arrow::PrimitiveArray *array, gdf_column* result){
  std::shared_ptr<arrow::ArrayData> data = array->data();
  std::shared_ptr<DataType> base = data->type;
  std::shared_ptr<DataType> arrow_dtype = std::dynamic_pointer_cast<DataType>(base);
  
  gdf_dtype dtype = arrow_dtype_to_gdf_dtype(arrow_dtype);
  result->data = (void *)array->values()->data();
  const uint8_t * null_bitmap_data = array->null_bitmap_data();
  if (null_bitmap_data != NULLPTR){
    result->valid = (gdf_valid_type*) null_bitmap_data;
  }else{
    result->valid = NULL;
  }
  result->size = array->length();
  result->dtype = dtype;
  result->null_count = array->null_count();
}


std::shared_ptr<arrow::PrimitiveArray> gdf_to_arrow(gdf_column* column){
  CudaDeviceManager* manager_;
  CudaDeviceManager::GetInstance(&manager_);
  std::shared_ptr<CudaContext> context_;
  manager_->GetContext(0, &context_);
  
  std::shared_ptr<DataType> arrow_dtype = gdf_dtype_to_arrow_dtype(column->dtype);
  int32_t width = byte_width(arrow_dtype);
  
  
  std::shared_ptr<CudaBuffer>  buffer = std::make_shared<CudaBuffer>((uint8_t*)column->data,
                                                                     column->size * width,
                                                                     context_,
                                                                     false,
                                                                     false);
  std::shared_ptr<PrimitiveArray> result;
  if (column->valid == NULL){
    auto array = std::make_shared<PrimitiveArray>(arrow_dtype, int64_t(column->size), buffer);
    return array;
  }else{
    auto null_bitmap = std::make_shared<CudaBuffer>((uint8_t *)column->valid,
                                                    column->size,
                                                    context_,
                                                    false,
                                                    false);
    auto array = std::make_shared<PrimitiveArray>(arrow_dtype,
                                                  int64_t(column->size),
                                                  buffer,
                                                  null_bitmap);
    return array;    
  }
}
