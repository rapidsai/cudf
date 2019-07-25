//
#include <memory.h>
#include <cuda_runtime.h>
#include <rmm/rmm.h>
extern "C" {
#include <rmm/rmm_api.h>
}
#include "regex.cuh"
#include "regcomp.h"

dreprog* dreprog::create_from(const char32_t* pattern, unsigned char* uflags )
{
    // compile pattern
    Reprog* prog = Reprog::create_from(pattern);
    // compute size to hold prog
    int insts_count = (int)prog->inst_count();
    int classes_count = (int)prog->classes_count();
    int starts_count = (int)prog->starts_count();
    int insts_size = insts_count * sizeof(Reinst);
    int sids_size = starts_count * sizeof(int);
    int classes_size = classes_count * sizeof(int); // offsets
    for( int idx=0; idx < classes_count; ++idx )
        classes_size += (int)((prog->class_at(idx).chrs.size())*sizeof(char32_t)) + (int)sizeof(int);
    // allocate memory to store prog
    size_t memsize = sizeof(dreprog) + insts_size + sids_size + classes_size;
    u_char* buffer = (u_char*)malloc(memsize);
    dreprog* rtn = (dreprog*)buffer;
    buffer += sizeof(dreprog);       // point to the end
    // copy the insts array first (fixed-size structs)
    Reinst* insts = (Reinst*)buffer;
    memcpy( insts, prog->insts_data(), insts_size);
    buffer += insts_size; // next section
    // copy the startinst_ids next (ints)
    int* startinst_ids = (int*)buffer;
    memcpy( startinst_ids, prog->starts_data(), sids_size );
    buffer += sids_size; // next section
    // classes are variable size so create offsets array
    int* offsets = (int*)buffer;
    buffer += classes_count * sizeof(int);
    char32_t* classes = (char32_t*)buffer;
    int offset = 0;
    for( int idx=0; idx < classes_count; ++idx )
    {
        Reclass& cls = prog->class_at(idx);
        memcpy( classes++, &(cls.builtins), sizeof(int) );
        int len = (int)cls.chrs.size();
        memcpy( classes, cls.chrs.c_str(), len*sizeof(char32_t) );
        offset += 1 + len;
        offsets[idx] = offset;
        classes += len;
    }
    // initialize the rest of the elements
    rtn->startinst_id = prog->get_start_inst();
    rtn->num_capturing_groups = prog->groups_count();
    rtn->insts_count = insts_count;
    rtn->starts_count = starts_count;
    rtn->classes_count = classes_count;
    rtn->unicode_flags = uflags;
    rtn->relists_mem = nullptr;
    rtn->stack_mem1 = nullptr;
    rtn->stack_mem2 = nullptr;

    // compiled prog copied into flat memory
    delete prog;

    // copy flat prog to device memory
    dreprog* d_rtn = 0;
    RMM_ALLOC(&d_rtn,memsize,0);//cudaMalloc(&d_rtn,memsize);
    cudaMemcpy(d_rtn,rtn,memsize,cudaMemcpyHostToDevice);
    free(rtn);
    return d_rtn;
}

void dreprog::destroy(dreprog* prog)
{
    prog->free_relists();
    RMM_FREE(prog,0);//cudaFree(prog);
}

bool dreprog::alloc_relists( size_t count )
{
    int insts = inst_counts();
    int rsz = Relist::alloc_size(insts);
    size_t rlmsz = rsz*2L*count; // Reljunk has 2 Relist ptrs
    void* rmem = 0;
    size_t freeSize=0, totalSize=0;
    rmmGetInfo(&freeSize,&totalSize,0);
    if( rlmsz > freeSize )
        return false;
    RMM_ALLOC(&rmem,rlmsz,0);
    //rtn->relists_mem = rmem;
    cudaMemcpy(&relists_mem,&rmem,sizeof(void*),cudaMemcpyHostToDevice);
    return true;
}

void dreprog::free_relists()
{
    void* cptr = 0; // this magic works but only as member function
    cudaMemcpy(&cptr,&relists_mem,sizeof(void*),cudaMemcpyDeviceToHost);
    if( cptr )
        RMM_FREE(cptr,0);//cudaFree(cptr);
}

int dreprog::inst_counts()
{
    int count = 0;
    cudaMemcpy(&count,&insts_count,sizeof(int),cudaMemcpyDeviceToHost);
    return count;
}

int dreprog::group_counts()
{
    int count = 0;
    cudaMemcpy(&count,&num_capturing_groups,sizeof(int),cudaMemcpyDeviceToHost);
    return count;
}

