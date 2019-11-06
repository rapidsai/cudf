
//

#include <cuda_runtime.h>
#include <rmm/device_buffer.hpp>
#include "./regex.cuh"
#include "./regcomp.h"

#include <memory.h>
#include <rmm/rmm.hpp>
#include <rmm/rmm_api.h>


namespace cudf
{
namespace strings
{
namespace detail
{

dreprog* dreprog::create_from(const char32_t* pattern, const uint8_t* codepoint_flags )
{
    // compile pattern into host object
    Reprog* prog = Reprog::create_from(pattern);
    // compute size to hold prog
    auto insts_count = prog->inst_count();
    auto classes_count = prog->classes_count();
    auto starts_count = prog->starts_count();
    auto insts_size = insts_count * sizeof(Reinst);
    auto sids_size = starts_count * sizeof(int32_t);
    auto classes_size = classes_count * sizeof(int32_t); // offsets
    for( int32_t idx=0; idx < classes_count; ++idx )
        classes_size += static_cast<int>((prog->class_at(idx).chrs.size())*sizeof(char32_t)) + sizeof(int32_t);
    // allocate memory to store prog
    size_t memsize = sizeof(dreprog) + insts_size + sids_size + classes_size;
    std::vector<u_char> h_buffer(memsize);
    // put everything into a flat host buffer first
    u_char* buffer = h_buffer.data(); // running pointer
    dreprog* rtn = reinterpret_cast<dreprog*>(buffer);
    buffer += sizeof(dreprog);       // point to the end
    // copy the instructions array first (fixed-size structs)
    Reinst* insts = reinterpret_cast<Reinst*>(buffer);
    memcpy( insts, prog->insts_data(), insts_size);
    buffer += insts_size; // next section
    // copy the startinst_ids next (ints)
    int32_t* startinst_ids = reinterpret_cast<int*>(buffer);
    memcpy( startinst_ids, prog->starts_data(), sids_size );
    buffer += sids_size; // next section
    // classes are variable size so create offsets array
    int32_t* offsets = reinterpret_cast<int*>(buffer);
    buffer += classes_count * sizeof(int32_t);
    char32_t* classes = (char32_t*)buffer;
    int32_t offset = 0;
    for( int32_t idx=0; idx < classes_count; ++idx )
    {
        Reclass& cls = prog->class_at(idx);
        memcpy( classes++, &(cls.builtins), sizeof(int32_t) );
        auto cls_length = cls.chrs.size();
        memcpy( classes, cls.chrs.c_str(), cls_length*sizeof(char32_t) );
        offset += 1 + cls_length;
        offsets[idx] = offset;
        classes += cls_length;
    }
    // initialize the rest of the elements
    rtn->startinst_id = prog->get_start_inst();
    rtn->num_capturing_groups = prog->groups_count();
    rtn->insts_count = insts_count;
    rtn->starts_count = starts_count;
    rtn->classes_count = classes_count;
    rtn->codepoint_flags = codepoint_flags;
    rtn->relists_mem = nullptr;
    rtn->stack_mem1 = nullptr;
    rtn->stack_mem2 = nullptr;

    // compiled prog copied into flat memory
    delete prog;

    // copy flat prog to device memory
    dreprog* d_rtn = 0;
    RMM_TRY(RMM_ALLOC(&d_rtn,memsize,0));
    CUDA_TRY(cudaMemcpy(d_rtn,rtn,memsize,cudaMemcpyHostToDevice));
    return d_rtn;
}

void dreprog::destroy(dreprog* dprog)
{
    dprog->free_relists();
    RMM_TRY(RMM_FREE(dprog,0));
}

bool dreprog::alloc_relists( size_t count )
{
    auto insts = inst_counts();
    auto new_size = Relist::alloc_size(insts);
    size_t rlmsz = new_size*2L*count; // Reljunk has 2 Relist ptrs
    void* new_mem_ptr = 0;
    size_t freeSize=0, totalSize=0;
    rmmGetInfo(&freeSize,&totalSize,0);
    if( rlmsz > freeSize ) // do not allocate more than we have
        return false;      // otherwise, this is unrecoverable
    RMM_TRY(RMM_ALLOC(&new_mem_ptr,rlmsz,0));
    // store the memory pointer
    CUDA_TRY(cudaMemcpy(&relists_mem,&new_mem_ptr,sizeof(void*),cudaMemcpyHostToDevice));
    return true;
}

void dreprog::free_relists()
{
    void* cptr = 0; // copy pointer to this variable
    CUDA_TRY(cudaMemcpy(&cptr,&relists_mem,sizeof(void*),cudaMemcpyDeviceToHost));
    if( cptr )
        RMM_TRY(RMM_FREE(cptr,0));
}

int32_t dreprog::inst_counts()
{
    int32_t count = 0;
    CUDA_TRY(cudaMemcpy(&count,&insts_count,sizeof(int),cudaMemcpyDeviceToHost));
    return count;
}

int32_t dreprog::group_counts()
{
    int32_t count = 0;
    CUDA_TRY(cudaMemcpy(&count,&num_capturing_groups,sizeof(int),cudaMemcpyDeviceToHost));
    return count;
}

} // namespace detail
} // namespace strings
} // namespace cudf
