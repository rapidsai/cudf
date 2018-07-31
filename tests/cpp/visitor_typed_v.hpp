#pragma once

#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>

#include <cstdint>

#define EXE_POL __host__ __device__

template<typename T, typename Allocator, template<typename, typename> class Vector>
EXE_POL
void print_v(const Vector<T, Allocator>& v, std::ostream& os)
{
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(os,","));
  os<<"\n";
}

template<typename T>
using Vector = thrust::device_vector<T>;

template<typename T>
struct VectorT;

struct BaseVisitor
{
  EXE_POL
  virtual ~BaseVisitor(void){}
  
  EXE_POL
  virtual void Visit(VectorT<int8_t>& ) = 0;

  EXE_POL
  virtual void Visit(VectorT<int16_t>& ) = 0;

  EXE_POL
  virtual void Visit(VectorT<int32_t>& ) = 0;

  EXE_POL
  virtual void Visit(VectorT<int64_t>& ) = 0;

  EXE_POL
  virtual void Visit(VectorT<float>& ) = 0;

  EXE_POL
  virtual void Visit(VectorT<double>& ) = 0;
};

template<typename IndexT>
struct SumVisitor: BaseVisitor
{
  EXE_POL
  SumVisitor(IndexT col_indx,
             thrust::tuple<IndexT*,void**,IndexT**> tptrs):
    d_sorted_indx_(thrust::get<0>(tptrs)),
    d_agg_out_(thrust::get<1>(tptrs)[col_indx]),
    d_kout_(thrust::get<2>(tptrs)[col_indx])
  {
  }

  EXE_POL
  virtual void Visit(VectorT<int8_t>& dv) override
  {
    using DType = int8_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<int16_t>& dv) override
  {
    using DType = int16_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }
  

  EXE_POL
  virtual void Visit(VectorT<int32_t>& dv) override
  {
    using DType = int32_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<int64_t>& dv) override
  {
    using DType = int64_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<float>& dv) override
  {
    using DType = float;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<double>& dv) override
  {
    using DType = double;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }
  
  ///protected:  
  template<typename DataType>
  void action(DataType* , VectorT<DataType>& );
private:
  IndexT* d_sorted_indx_;
  void* d_agg_out_;
  IndexT* d_kout_;
};


template<typename IndexT>
struct MinVisitor: BaseVisitor
{
  EXE_POL
  MinVisitor(IndexT col_indx,
             thrust::tuple<IndexT*,void**,IndexT**> tptrs):
    d_sorted_indx_(thrust::get<0>(tptrs)),
    d_agg_out_(thrust::get<1>(tptrs)[col_indx]),
    d_kout_(thrust::get<2>(tptrs)[col_indx])
  {
  }

  EXE_POL
  virtual void Visit(VectorT<int8_t>& dv) override
  {
    using DType = int8_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<int16_t>& dv) override
  {
    using DType = int16_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }
  

  EXE_POL
  virtual void Visit(VectorT<int32_t>& dv) override
  {
    using DType = int32_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<int64_t>& dv) override
  {
    using DType = int64_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<float>& dv) override
  {
    using DType = float;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<double>& dv) override
  {
    using DType = double;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }
  
  ///protected:  
  template<typename DataType>
  void action(DataType* , VectorT<DataType>& );
private:
  IndexT* d_sorted_indx_;
  void* d_agg_out_;
  IndexT* d_kout_;
};

template<typename IndexT>
struct MaxVisitor: BaseVisitor
{
  EXE_POL
  MaxVisitor(IndexT col_indx,
             thrust::tuple<IndexT*,void**,IndexT**> tptrs):
    d_sorted_indx_(thrust::get<0>(tptrs)),
    d_agg_out_(thrust::get<1>(tptrs)[col_indx]),
    d_kout_(thrust::get<2>(tptrs)[col_indx])
  {
  }

  EXE_POL
  virtual void Visit(VectorT<int8_t>& dv) override
  {
    using DType = int8_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<int16_t>& dv) override
  {
    using DType = int16_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }
  

  EXE_POL
  virtual void Visit(VectorT<int32_t>& dv) override
  {
    using DType = int32_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<int64_t>& dv) override
  {
    using DType = int64_t;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<float>& dv) override
  {
    using DType = float;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }

  EXE_POL
  virtual void Visit(VectorT<double>& dv) override
  {
    using DType = double;
    DType* d_out = static_cast<DType*>(d_agg_out_);
    action(d_out, dv);
  }
  
  ///protected:  
  template<typename DataType>
  void action(DataType* , VectorT<DataType>& );
private:
  IndexT* d_sorted_indx_;
  void* d_agg_out_;
  IndexT* d_kout_;
};



struct BaseVector
{
  EXE_POL
  virtual ~BaseVector(void){}

  EXE_POL
  virtual void Apply(BaseVisitor* ) = 0;

private:
  void* p_data_;
};

template<typename T>
struct VectorT: BaseVector
{
  template<typename IndexT>
  friend class SumVisitor;

  template<typename IndexT>
  friend class MinVisitor;

  template<typename IndexT>
  friend class MaxVisitor;
  
  EXE_POL
  VectorT(void* p, size_t sz):
    p_data_(static_cast<T*>(p)),
    sz_(sz)
  {
  }

  EXE_POL
  void Apply(BaseVisitor* v) override
  {
    v->Visit(*this);
  }

  EXE_POL
  const T* get_data(void) const
  {
    return p_data_;
  }

  EXE_POL
  size_t size(void) const
  {
    return sz_;
  }
private:
  T* p_data_;
  size_t sz_;
};



template<typename IndexT>
template<typename DataType>
EXE_POL
void SumVisitor<IndexT>::action(DataType* d_out, VectorT<DataType>& dv)
{
  auto r = [] EXE_POL (DataType op1, DataType op2){
    return op1 + op2;
  };
  thrust::reduce_by_key(thrust::device,
                        d_sorted_indx_, d_sorted_indx_ + dv.size(),
                        dv.p_data_, //aggregation source;
                        d_kout_,
                        d_out,//aggregation target;
                        thrust::equal_to<IndexT>(),//TODO: change to multi-column equal operator
                        r);
}

template<typename IndexT>
template<typename DataType>
EXE_POL
void MinVisitor<IndexT>::action(DataType* d_out, VectorT<DataType>& dv)
{
  auto r = [] EXE_POL (DataType op1, DataType op2){
    return op1 < op2 ? op1 : op2;
  };
  thrust::reduce_by_key(thrust::device,
                        d_sorted_indx_, d_sorted_indx_ + dv.size(),
                        dv.p_data_, //aggregation source;
                        d_kout_,
                        d_out,//aggregation target;
                        thrust::equal_to<IndexT>(),//TODO: change to multi-column equal operator
                        r);
}

template<typename IndexT>
template<typename DataType>
EXE_POL
void MaxVisitor<IndexT>::action(DataType* d_out, VectorT<DataType>& dv)
{
  auto r = [] EXE_POL (DataType op1, DataType op2){
    return op1 > op2 ? op1 : op2;
  };
  thrust::reduce_by_key(thrust::device,
                        d_sorted_indx_, d_sorted_indx_ + dv.size(),
                        dv.p_data_, //aggregation source;
                        d_kout_,
                        d_out,//aggregation target;
                        thrust::equal_to<IndexT>(),//TODO: change to multi-column equal operator
                        r);
}

template<typename IndexT,
         typename TupleArgs>
struct VisitorFactory
{
  VisitorFactory(size_t ncols,
                 TupleArgs& tpl_args,
                 std::vector<int>& h_agg_types):
    d_visitors_(ncols, nullptr),
    d_agg_types_(h_agg_types)
  {
    construct(tpl_args);
  }

  ~VisitorFactory(void)
  {
    destroy();
  }


  Vector<BaseVisitor*>& get_visitors(void)
  {
    return d_visitors_;
  }

  void construct(TupleArgs& tpl_args)
  {
    size_t ncols = d_agg_types_.size();
    int* ptr_d_agg_t = d_agg_types_.data().get();
    
    thrust::transform(thrust::device,
                    thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(ncols),
                    d_visitors_.begin(),
                      [ncols, tpl_args, ptr_d_agg_t] EXE_POL (size_t col_indx){
                        switch( ptr_d_agg_t[col_indx] )
                          {
                          case  GDF_SUM:
                            {
                              return dynamic_cast<BaseVisitor*>(new SumVisitor<IndexT>(col_indx, tpl_args));
                              break;
                            }
                          case GDF_MIN:
                            {
                              return  dynamic_cast<BaseVisitor*>(new MinVisitor<IndexT>(col_indx, tpl_args));
                              break;
                            }
                          case GDF_MAX:
                            {
                              return  dynamic_cast<BaseVisitor*>(new MaxVisitor<IndexT>(col_indx, tpl_args));
                              break;
                            }
                          default:
                            assert(false);//unsopported, yet
                          }
                    });

  }

  void destroy(void)
  {
    //free device space for visitors and typed_columns:
    //
    thrust::for_each(d_visitors_.begin(), d_visitors_.end(),
                     [] EXE_POL (BaseVisitor*& ptr){
                       delete ptr;
                       ptr = nullptr;
                     });

  }
               
private:
  Vector<BaseVisitor*> d_visitors_;
  Vector<int> d_agg_types_;
};

struct VectorFactory
{
  VectorFactory(size_t nrows,
                Vector<void*>& d_cols,
                Vector<int>& d_types):
    d_columns_(d_cols.size(), nullptr)
  {
    construct(nrows, d_cols, d_types);
  }

  ~VectorFactory(void)
  {
    destroy();
  }

  Vector<BaseVector*>& get_columns(void)
  {
    return d_columns_;
  }

  void construct(size_t nrows,
                 Vector<void*>& d_cols,
                 Vector<int>& d_types)
  {
    size_t ncols = d_cols.size();
    assert( ncols == d_types.size() );

    void** p_ptr_d_cols = d_cols.data().get();
    int* ptr_d_types    = d_types.data().get();
    
    thrust::transform(thrust::device,
                    thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(ncols),
                    d_columns_.begin(),
                    [nrows, p_ptr_d_cols, ptr_d_types] EXE_POL (size_t col_indx){
                     void* ptr_d_agg_in  = p_ptr_d_cols[col_indx];

                     switch( ptr_d_types[col_indx] )
                       {
                       case GDF_INT8:
                         {
                           using VType = int8_t;
                           return dynamic_cast<BaseVector*>(new VectorT<VType>(ptr_d_agg_in, nrows));
                         }
                       case GDF_INT16:
                         {
                           using VType = int16_t;
                           return dynamic_cast<BaseVector*>(new VectorT<VType>(ptr_d_agg_in, nrows));
                         }
                       case GDF_INT32:
                         {
                           using VType = int32_t;
                           return dynamic_cast<BaseVector*>(new VectorT<VType>(ptr_d_agg_in, nrows));
                         }
                       case GDF_INT64:
                         {
                           using VType = int64_t;
                           return dynamic_cast<BaseVector*>(new VectorT<VType>(ptr_d_agg_in, nrows));
                         }
                       case GDF_FLOAT32:
                         {
                           using VType = float;
                           return dynamic_cast<BaseVector*>(new VectorT<VType>(ptr_d_agg_in, nrows));
                         }
                       case GDF_FLOAT64:
                         {
                           using VType = double;
                           return dynamic_cast<BaseVector*>(new VectorT<VType>(ptr_d_agg_in, nrows));
                         }
                       default:
                         assert( false );//unsopported, yet
                         ///return nullptr;//not implemented yet
                       }
                    });
  }

   void destroy(void)
  {
    //free device space for visitors and typed_columns:
    //
    thrust::for_each(d_columns_.begin(), d_columns_.end(),
                     [] EXE_POL (BaseVector*& ptr){
                       delete ptr;
                       ptr = nullptr;
                     });
  }
  
private:
  size_t n_cols_;
  Vector<BaseVector*> d_columns_;              
};
