#include <memory>
#include <iostream>
#include "device_column.cuh"
#include "gdf_type_dispatcher.cuh"

// Anonymous namespace
namespace{

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief Kernel to construct a TypedDeviceColumn<T> on the device.
   * 
   * @Param[in,out] p A pointer to a device allocation large enough to hold a 
   * DeviceColumn*. When the kernel completes, this allocation will contain
   * a pointer to the newly allocated TypedDeviceColumn.
   * @Param[in] col The gdf_column that will be used to initialize the TypedDeviceColumn
   */
  /* ----------------------------------------------------------------------------*/
  template <typename T>
  __global__ 
  void allocator_kernel(DeviceColumn ** p, gdf_column col){
    if(0 == threadIdx.x + blockIdx.x * blockDim.x)
      *p = static_cast<DeviceColumn*>(new TypedDeviceColumn<T>(col));
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief  Kernel to destroy a DeviceColumn constructed on the device.
   * 
   * @Param p Pointer to a device allocation that contains a pointer to the 
   * device-constructed DeviceColumn that will be destroyed
   */
  /* ----------------------------------------------------------------------------*/
  __global__ 
  void deallocator_kernel(DeviceColumn ** p){
    if(0 == threadIdx.x + blockIdx.x * blockDim.x)
      delete *p;
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief  Functor that constructs on the device a TypedDeviceColumn<T> with 
   * the type T determined by the gdf_dtype enum in a gdf_column.
   */
  /* ----------------------------------------------------------------------------*/
  struct device_column_allocator{

    /* --------------------------------------------------------------------------*/
    /** 
     * @brief  Templated operator() that constructs the TypedColumn<T> on the device.
     * 
     * This operator() is invoked by the gdf_type_dispatcher with the appropriate 
     * T determined by the gdf_column's gdf_dtype enum.
     * 
     * @Param p 
     * @Param col
     */
    /* ----------------------------------------------------------------------------*/
    template<typename T>
    __host__
    DeviceColumn ** operator()(gdf_column const& col){
      
      DeviceColumn** p;

      cudaMalloc(&p, sizeof(DeviceColumn*));
      
      auto status = cudaGetLastError();
      if(cudaSuccess != status)
        std::cerr << "Failed to allocate DeviceColumn pointer.\n";
      
      cudaEvent_t sync{};
      cudaEventCreate(&sync);

      // Invokes kernel to construct the TypedColumn<T> on the device
      allocator_kernel<T><<<1,1>>>(p, col);
      cudaEventRecord(sync);

      // Use events to ensure kernel is complete and object is constructed before
      // returning
      cudaEventSynchronize(sync);

      status = cudaGetLastError();
      if(cudaSuccess != status)
        std::cerr << "Failed to allocate DeviceColumn.\n";

      return p;
    }
  };

} // end anonymous namespace

/* --------------------------------------------------------------------------*/
/** 
 * @brief  TypedColumn factory function.
 *
 * This factory function is used to construct (on the device!) a TypedDeviceColumn<T>
 * with the appropriate T based on the gdf_dtype enum in a gdf_column.
 * 
 * @Param col The gdf_column whose gdf_dtype will determine the type T of the
 * TypedDeviceColumn<T> that is constructed to wrap the gdf_column and 
 * reconstruct the type of the gdf_column's type-erased data buffer.
 * 
 * @Returns A unique pointer that contains a device allocated base-class pointer 
 * to the newly constructed TypedColumn<T>. NOTE: Because the base-class pointer 
 * is allocated in device memory, it should NOT be dereferenced on the host.
 */
/* ----------------------------------------------------------------------------*/
DeviceColumnPointer DeviceColumn::make_device_column(gdf_column col)
{
  auto device_column_deleter = [](DeviceColumn ** d_col)
                                 {
                                   deallocator_kernel<<<1,1>>>(d_col);
                                   cudaFree(d_col);
                                 };

  DeviceColumnPointer p(gdf_type_dispatcher(col.dtype, device_column_allocator{}, col),
                        device_column_deleter);

  return p;
}
