#ifndef _MEMORY_POOL_H
#define _MEMORY_POOL_H

#include <forward_list>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <mutex>

#include "cuda_runtime.h"
#include "common.h"

// #define MEMORY_POOL_DEBUG

namespace needle {
namespace cuda {
namespace utilities{


const size_t DEFAULT_FREELIST_SIZE = (0x1lu << 30);
const size_t MAX_FREELIST_LARGE_SIZE = (0x1lu << 31 | 0x1lu << 30);
const size_t MAX_FREELIST_MINOR_SIZE = (0x1lu << 31);

struct GpuMemoryBlock{
    GpuMemoryBlock() : data(nullptr), size(0x0), cookies(0){}
    GpuMemoryBlock(const GpuMemoryBlock& gpuMemoryBlock_) : data(gpuMemoryBlock_.data), size(gpuMemoryBlock_.size), cookies(gpuMemoryBlock_.cookies){}
    GpuMemoryBlock& operator=(const GpuMemoryBlock& gpuMemoryBlock_){
        data = gpuMemoryBlock_.data;
        size = gpuMemoryBlock_.size;
        cookies = gpuMemoryBlock_.cookies;
        return *this;
    }
    ~GpuMemoryBlock(){}

    unsigned char* data;
    size_t size;
    /*
    > 100 MB, MB_LARGE, using naive freelist;
    (512 KB, 100 MB], index of _freelist_small, using freelist pool;
    <= 512 KB, MB_KB_512, using naive freelist
    */
    size_t cookies;
};

class MemoryFreeList{
public:
    MemoryFreeList() = delete;
    MemoryFreeList(size_t size_, size_t _pitch_granularity_, size_t _pitch_shfit_);
    ~MemoryFreeList();

    MemoryFreeList(const MemoryFreeList&) = delete;
    MemoryFreeList& operator=(const MemoryFreeList&) = delete;
    void mallocGpuMemoryBlock(size_t, GpuMemoryBlock&);
    void freeGpuMemoryBlock(GpuMemoryBlock&);
    
private:
    void _initializeFreelist();

private:
    size_t _freelist_size;
    std::forward_list<GpuMemoryBlock> _freelist;
    unsigned char* _freelist_begin = nullptr;
    unsigned char* _freelist_end = nullptr;

    size_t _pitch_granularity;
    size_t _pitch_shfit;

};


class GpuMemoryPool{
public:
    GpuMemoryPool();
    GpuMemoryPool(size_t large_size_, size_t minor_size_);
    ~GpuMemoryPool();

    // avoid copy
    GpuMemoryPool(const GpuMemoryPool&) = delete;   
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;

    void mallocBlock(size_t size, GpuMemoryBlock& gpuMemoryBlock);
    void freeBlock(GpuMemoryBlock& gpuMemoryBlock);

private:
    void _mallocLargeBlock(size_t size, GpuMemoryBlock& gpuMemoryBlock);
    void _freeLargeBlock(GpuMemoryBlock& gpuMemoryBlock);
    void _mallocSmallBlock(size_t size, GpuMemoryBlock& gpuMemoryBlock);
    void _freeSmallBlock(GpuMemoryBlock& gpuMemoryBlock);

private:
    std::forward_list<GpuMemoryBlock> _freelist_small[BLOCK_SIZE_INDEX::MB_LARGE];
    std::forward_list<GpuMemoryBlock> _closelist_small[BLOCK_SIZE_INDEX::MB_LARGE];

    size_t _minor_freelist_size;
    size_t _large_freelist_size;
    MemoryFreeList* _large_freelist;
    MemoryFreeList* _minor_freelist;
};

}   // utilities
}   // cuda
}   // needle


#endif
