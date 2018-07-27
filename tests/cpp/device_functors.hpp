#pragma once

template<typename T, typename Allocator, template<typename, typename> class Vector>
__host__ __device__
void print_v(const Vector<T, Allocator>& v, std::ostream& os)
{
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(os,","));
  os<<"\n";
}

template<typename T>
using Vector = thrust::device_vector<T>;

namespace FctrPtrsDefs
{
  template<typename T>
  __host__ __device__
  T sum(T x1, T x2)
  {
    return x1 + x2;
  }

  template<typename T>
  __host__ __device__
  T min(T x1, T x2)
  {
    return x1<x2?x1:x2;
  }

  template<typename T>
  __host__ __device__
  T max(T x1, T x2)
  {
    return x1>x2?x1:x2;
  }


  template<typename T>
  __host__ __device__
  T nothing(T x1, T x2) // see below why this is _NECESSARY_ 
  {
    return T(0);
  }

  template<typename T>
  __host__ __device__
  T* convertor(void* ptr)//T* ptr //won't help...passed argument is still void* and compiler
  //doesn't know how to resolve to T* based on return parameter alone
  {
    return static_cast<T*>(ptr);
  }

  template<class T>
  constexpr __device__ T (*d_ptr_sum)(T, T) = sum<T>;

  template<class T>
  constexpr __device__ T (*d_ptr_min)(T, T) = min<T>;

  template<class T>
  constexpr __device__ T (*d_ptr_max)(T, T) = max<T>;

  template<class T>
  constexpr __device__ T* (*d_conv)(void* ) = convertor<T>;

  // template<class T>
  // constexpr __device__ T* (*d_conv)(T* ) = convertor<T>;
  //won't help...passed argument is still void* and compiler
  //doesn't know how to resolve to T* based on return parameter alone
}//end namespace

enum Types{DOUBLE=1, INT};
enum OpTypes{SUM = 0, MIN, MAX};

typedef void* (*FTV)(void*, void*);

struct GnctrTypeErased
{
  __host__ __device__
  GnctrTypeErased(void)
  {
  }
  
  __host__ __device__
  GnctrTypeErased(FTV doer, short dtype):
    act_(doer),
    dtype_(dtype)
  {
  }

   __host__ __device__
  GnctrTypeErased(const GnctrTypeErased& other) = default;

  __host__ __device__
  GnctrTypeErased& operator = (const GnctrTypeErased& other) = default;


  template<typename T> 
  __host__ __device__
  T operator()(T x1, T x2) const
  {
    switch( dtype_ )
      {
      case Types::DOUBLE://double
        {
          typedef double (*FTD)(double, double);
          FTD actd = reinterpret_cast<FTD>(act_);
          return actd(x1, x2);
        }
      case Types::INT://int
        {
          typedef int (*FTI)(int, int);
          FTI acti = reinterpret_cast<FTI>(act_);
          return acti(x1, x2);
        }
      default:
        return T();
      }
  }
private:
  FTV act_;
  short dtype_;
};

struct AggregationManager
{
  AggregationManager(const std::vector<short>& v_types,
                     const std::vector<short>& v_ops):
    n_cols_(v_types.size()),
    d_fte_(n_cols_)
  {
    ///TODO: try copying the whole set H to D...
    ///std::vector<GnctrTypeErased> h_fte;//<-tried: segfaults
    for(int i=0;i<n_cols_;++i)
    {
      FTV h_functr = nullptr;
      switch( v_types[i] )
        {
        case Types::DOUBLE:
          {
            using Type = double;
            op_dispatch<Type>(v_ops[i], h_functr);
            break;
          }
        case Types::INT:
          {
            using Type = int;
            op_dispatch<Type>(v_ops[i], h_functr);
            break;
          }
        }
      d_fte_[i] = GnctrTypeErased(h_functr, v_types[i]);//PROBLEM: expensive; 
      ///h_fte[i] =  GnctrTypeErased(h_functr, v_types[i]);//<-tried: segfaults
    }
    ///d_fte_ = h_fte;//<-tried: segfaults
  }

  Vector<GnctrTypeErased>& get_vfctrs(void)
  {
    return d_fte_;
  }
protected:
  template<typename T>
  static void op_dispatch(int op, FTV& h_functr)
  {
    switch(op)
      {
      case OpTypes::SUM:
        cudaMemcpyFromSymbol(&h_functr, FctrPtrsDefs::d_ptr_sum<T>, sizeof(void *));
        break;
      case OpTypes::MAX:
        cudaMemcpyFromSymbol(&h_functr, FctrPtrsDefs::d_ptr_max<T>, sizeof(void *));
        break;
      case OpTypes::MIN:
        cudaMemcpyFromSymbol(&h_functr, FctrPtrsDefs::d_ptr_min<T>, sizeof(void *));
        break;
      }
  }
private:
  size_t n_cols_;
  Vector<GnctrTypeErased> d_fte_;
};
