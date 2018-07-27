#pragma once

#include "device_functors.hpp"

struct VectorInt;
struct VectorDouble;

#define EXE_POL __host__ __device__

struct BaseVisitor
{
  EXE_POL
  virtual ~BaseVisitor(void){}
  
  EXE_POL
  virtual void VisitVInt(VectorInt& ) = 0;

  EXE_POL
  virtual void VisitVDouble(VectorDouble& ) = 0;
};

template<typename IndexT>
struct ReductionVisitor: BaseVisitor
{
  EXE_POL
  ReductionVisitor(IndexT* ptr_sorted_indx,
                   void* ptr_agg_out,
                   IndexT* ptr_d_kout,
                   GnctrTypeErased g):
    d_sorted_indx_(ptr_sorted_indx),
    d_agg_out_(ptr_agg_out),
    d_kout_(ptr_d_kout),
    g_(g)
  {
  }

  EXE_POL
  void VisitVInt(VectorInt& dv) override;

  EXE_POL
  void VisitVDouble(VectorDouble& dv) override;
  

private:
  IndexT* d_sorted_indx_;
  void* d_agg_out_;
  IndexT* d_kout_;
  GnctrTypeErased g_;
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

struct VectorInt: BaseVector
{
  template<typename T>
  friend EXE_POL void ReductionVisitor<T>::VisitVInt(VectorInt& );
  
  EXE_POL
  VectorInt(void* p, size_t sz):
    p_data_(static_cast<int*>(p)),
    sz_(sz)
  {
  }

  EXE_POL
  void Apply(BaseVisitor* v) override
  {
    v->VisitVInt(*this);
  }

  EXE_POL
  const int* get_data(void) const
  {
    return p_data_;
  }

  EXE_POL
  size_t size(void) const
  {
    return sz_;
  }
private:
  int* p_data_;
  size_t sz_;
};

struct VectorDouble: BaseVector
{
  template<typename T>
  friend EXE_POL void ReductionVisitor<T>::VisitVDouble(VectorDouble& );
  
  EXE_POL
  VectorDouble(void* p, size_t sz):
    p_data_(static_cast<double*>(p)),
    sz_(sz)
  {
  }

  EXE_POL
  void Apply(BaseVisitor* v) override
  {
    v->VisitVDouble(*this);
  }

  EXE_POL
  const double* get_data(void) const
  {
    return p_data_;
  }

  EXE_POL
  size_t size(void) const
  {
    return sz_;
  }
private:
  double* p_data_;
  size_t sz_;
};

template<typename T>
EXE_POL
void ReductionVisitor<T>::VisitVInt(VectorInt& dv)
{
  using AggType = int;
  thrust::reduce_by_key(thrust::device,
                        d_sorted_indx_, d_sorted_indx_ + dv.size(),
                        dv.p_data_, //aggregation source;
                        d_kout_,
                        static_cast<AggType*>(d_agg_out_),//aggregation target;
                        thrust::equal_to<T>(),
                        g_);
}


template<typename T>
EXE_POL
void ReductionVisitor<T>::VisitVDouble(VectorDouble& dv)
{
  using AggType = double;
  thrust::reduce_by_key(thrust::device,
                        d_sorted_indx_, d_sorted_indx_ + dv.size(),
                        dv.p_data_, //aggregation source;
                        d_kout_,
                        static_cast<AggType*>(d_agg_out_),//aggregation target;
                        thrust::equal_to<T>(),
                        g_);
}

template<typename IndexT>
struct ReductionFactory
{
  ReductionFactory(size_t nrows,
                   Vector<GnctrTypeErased>& d_fte,
                   Vector<IndexT>& d_keys,
                   Vector<void*>& d_cols,
                   Vector<IndexT*>& dkout,
                   Vector<void*>& dvout,
                   Vector<short>& d_types):
    n_cols_(d_fte.size()),
    d_columns_(n_cols_, nullptr),
    d_visitors_(n_cols_, nullptr)
  {
    construct(nrows, d_fte, d_keys, d_cols, dkout, dvout, d_types);
  }

  ~ReductionFactory(void)
  {
    destroy();
  }

  Vector<BaseVector*>& get_columns(void)
  {
    return d_columns_;
  }

  Vector<BaseVisitor*>& get_visitors(void)
  {
    return d_visitors_;
  }

  void construct(size_t nrows,
                 Vector<GnctrTypeErased>& d_fte,
                 Vector<IndexT>& d_keys,
                 Vector<void*>& d_cols,
                 Vector<IndexT*>& dkout,
                 Vector<void*>& dvout,
                 Vector<short>& d_types)
  {
    GnctrTypeErased* ptr_d_binary_ops = d_fte.data().get();
    IndexT* ptr_sorted_indx           = d_keys.data().get();
    void** p_ptr_d_cols               = d_cols.data().get();
    IndexT** p_ptr_d_kout             = dkout.data().get();
    void** p_ptr_d_vout               = dvout.data().get();
    short* ptr_d_types                = d_types.data().get();

    thrust::transform(thrust::device,
                    thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(n_cols_),
                    d_visitors_.begin(),
                    [ptr_d_binary_ops, ptr_sorted_indx, p_ptr_d_kout, p_ptr_d_vout] EXE_POL (size_t col_indx){
                      void* ptr_d_agg_out = p_ptr_d_vout[col_indx];
                      IndexT* ptr_d_kout  = p_ptr_d_kout[col_indx];
                      GnctrTypeErased g   = ptr_d_binary_ops[col_indx];
                     
                      return new ReductionVisitor<IndexT>(ptr_sorted_indx, ptr_d_agg_out, ptr_d_kout,g);
                    });

    thrust::transform(thrust::device,
                    thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(n_cols_),
                    d_columns_.begin(),
                    [nrows, p_ptr_d_cols, ptr_d_types] EXE_POL (IndexT col_indx){
                     void* ptr_d_agg_in  = p_ptr_d_cols[col_indx];

                     switch( ptr_d_types[col_indx] )
                       {
                       case Types::INT:
                         {
                           return dynamic_cast<BaseVector*>(new VectorInt(ptr_d_agg_in, nrows));
                         }
                       case Types::DOUBLE:
                         {
                           return dynamic_cast<BaseVector*>(new VectorDouble(ptr_d_agg_in, nrows));
                         }
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

    thrust::for_each(d_columns_.begin(), d_columns_.end(),
                     [] EXE_POL (BaseVector*& ptr){
                       delete ptr;
                       ptr = nullptr;
                     });
  }
               
private:
  size_t n_cols_;
  Vector<BaseVector*> d_columns_;
  Vector<BaseVisitor*> d_visitors_;
};


