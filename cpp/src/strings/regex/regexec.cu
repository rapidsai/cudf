
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

// Copy Reprog primitive values
Reprog_device::Reprog_device(Reprog* prog)
{
    startinst_id = prog->get_start_inst();
    num_capturing_groups = prog->groups_count();
    insts_count = prog->inst_count();
    starts_count = prog->starts_count();
    classes_count = prog->classes_count();
    relists_mem = nullptr;
    stack_mem1 = nullptr;
    stack_mem2 = nullptr;
}

// Create instance that can be used inside a device kernel
std::unique_ptr<Reprog_device, std::function<void(Reprog_device*)>> Reprog_device::create(const char32_t* pattern, const uint8_t* codepoint_flags )
{
    // compile pattern into host object
    Reprog* h_prog = Reprog::create_from(pattern);
    std::unique_ptr<Reprog> u_prog(h_prog);
    // compute size to hold all the member data
    auto insts_count = h_prog->inst_count();
    auto classes_count = h_prog->classes_count();
    auto starts_count = h_prog->starts_count();
    auto insts_size = insts_count * sizeof(insts[0]);
    auto startids_size = starts_count * sizeof(startinst_ids[0]);
    auto classes_size = classes_count * sizeof(classes[0]);
    for( int32_t idx=0; idx < classes_count; ++idx )
        classes_size += static_cast<int32_t>((h_prog->class_at(idx).chrs.size())*sizeof(char32_t));
    // allocate memory to store prog data
    size_t memsize = insts_size + startids_size + classes_size;
    std::vector<u_char> h_buffer(memsize);
    u_char* h_ptr = h_buffer.data(); // running pointer
    u_char* d_buffer = 0;
    RMM_TRY(RMM_ALLOC(&d_buffer,memsize,0));
    u_char* d_ptr = d_buffer;        // running device pointer
    // put everything into a flat host buffer first
    Reprog_device* d_prog = new Reprog_device(h_prog);
    // copy the instructions array first (fixed-size structs)
    Reinst* insts = reinterpret_cast<Reinst*>(h_ptr);
    memcpy( insts, h_prog->insts_data(), insts_size);
    h_ptr += insts_size; // next section
    d_prog->insts = reinterpret_cast<Reinst*>(d_ptr);
    d_ptr += insts_size;
    // copy the startinst_ids next (ints)
    int32_t* startinst_ids = reinterpret_cast<int32_t*>(h_ptr);
    memcpy( startinst_ids, h_prog->starts_data(), startids_size );
    h_ptr += startids_size; // next section
    d_prog->startinst_ids = reinterpret_cast<int32_t*>(d_ptr);
    d_ptr += startids_size;
    // copy classes into flat memory: [class1,class2,...][char32 arrays]
    Reclass_device* classes = reinterpret_cast<Reclass_device*>(h_ptr);
    d_prog->classes = reinterpret_cast<Reclass_device*>(d_ptr);
    // get pointer to the end to handle variable length data
    u_char* h_end = h_ptr + (classes_count * sizeof(Reclass_device));
    u_char* d_end = d_ptr + (classes_count * sizeof(Reclass_device));
    // place each class and append the variable length data
    for( int32_t idx=0; idx < classes_count; ++idx )
    {
        Reclass& h_class = h_prog->class_at(idx);
        Reclass_device d_class;
        d_class.builtins = h_class.builtins;
        d_class.count = h_class.chrs.size();
        d_class.chrs = reinterpret_cast<char32_t*>(d_end);
        memcpy( classes++, &d_class, sizeof(d_class) );
        memcpy( h_end, h_class.chrs.c_str(), h_class.chrs.size()*sizeof(char32_t) );
        h_end += h_class.chrs.size()*sizeof(char32_t);
        d_end += h_class.chrs.size()*sizeof(char32_t);
    }
    // initialize the rest of the elements
    d_prog->insts_count = insts_count;
    d_prog->starts_count = starts_count;
    d_prog->classes_count = classes_count;
    d_prog->codepoint_flags = codepoint_flags;
    // compiled prog copied into flat memory
    //delete h_prog;

    // copy flat prog to device memory
    CUDA_TRY(cudaMemcpy(d_buffer,h_buffer.data(),memsize,cudaMemcpyHostToDevice));
    //
    auto deleter = [](Reprog_device*t) {t->destroy();};
    return std::unique_ptr<Reprog_device, std::function<void(Reprog_device*)>>(d_prog,deleter);
}

void Reprog_device::destroy()
{
    if( relists_mem )
        RMM_FREE(relists_mem,0);
    RMM_FREE(insts,0);
    delete this;
}

// allocate extra memory for executing large regex patterns
bool Reprog_device::alloc_relists( size_t count )
{
    auto insts = inst_counts();
    auto new_size = Relist::alloc_size(insts);
    size_t rlmsz = new_size*2L*count; // Reljunk has 2 Relist ptrs
    size_t freeSize=0, totalSize=0;
    rmmGetInfo(&freeSize,&totalSize,0);
    if( rlmsz > freeSize ) // do not allocate more than we have
        return false;      // otherwise, this is unrecoverable
    RMM_TRY(RMM_ALLOC(&relists_mem,rlmsz,0));
    return true;
}

} // namespace detail
} // namespace strings
} // namespace cudf
