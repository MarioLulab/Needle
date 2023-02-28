#include "memory_pool.h"
#include "common.h"

namespace needle {
namespace cuda {
namespace utilities{

GpuMemoryPool::GpuMemoryPool() : _large_freelist_size(MAX_FREELIST_LARGE_SIZE), _minor_freelist_size(MAX_FREELIST_MINOR_SIZE){
    _large_freelist = new MemoryFreeList(_large_freelist_size, PITCH_GRANULARITY_LARGE, PITCH_SHIFT_LARGE);
    _minor_freelist = new MemoryFreeList(_minor_freelist_size, PITCH_GRANULARITY_MINOR, PITCH_SHIFT_MINOR);
};

GpuMemoryPool::GpuMemoryPool(size_t large_size_, size_t minor_size_) : _large_freelist_size(large_size_), _minor_freelist_size(minor_size_){
    _large_freelist = new MemoryFreeList(_large_freelist_size, PITCH_GRANULARITY_LARGE, PITCH_SHIFT_LARGE);
    _minor_freelist = new MemoryFreeList(_minor_freelist_size, PITCH_GRANULARITY_MINOR, PITCH_SHIFT_MINOR);
};

void GpuMemoryPool::mallocBlock(size_t size, GpuMemoryBlock& gpuMemoryBlock){
    if (size > MAX_BYTES){
        _mallocLargeBlock(size, gpuMemoryBlock);
    }
    else{
        _mallocSmallBlock(size, gpuMemoryBlock);
    }
}

void GpuMemoryPool::_mallocLargeBlock(size_t size, GpuMemoryBlock& gpuMemoryBlock){
    _large_freelist->mallocGpuMemoryBlock(size, gpuMemoryBlock);
}

void GpuMemoryPool::_freeLargeBlock(GpuMemoryBlock& gpuMemoryBlock){
    _large_freelist->freeGpuMemoryBlock(gpuMemoryBlock);
}

void GpuMemoryPool::_mallocSmallBlock(size_t size, GpuMemoryBlock& gpuMemoryBlock){
    auto cookies = _getBlockCookies(size);
    if (cookies == BLOCK_SIZE_INDEX::MB_LARGE){
        throw std::runtime_error("GPU Memory Pool Error: _getSmallBlockCookies retval error. small block cookies Expected, large got.");
    }

    if (cookies == BLOCK_SIZE_INDEX::KB_512){
        // allocate minor memory block
        _minor_freelist->mallocGpuMemoryBlock(size, gpuMemoryBlock);
    }
    else{
        if (_freelist_small[cookies].empty()){
            // there is no memory cache in freelist, allocate one
            unsigned char* newdata;
            cudaError_t code = cudaMalloc((void**)&newdata, CONST_VALUE::BYTES_VEC[cookies]);
            if (code != cudaSuccess){
                throw std::runtime_error(std::string("_mallocSmallBlock() ") + cudaGetErrorString(code));
            }
            gpuMemoryBlock.data = newdata;
            gpuMemoryBlock.cookies = cookies;
            gpuMemoryBlock.size = CONST_VALUE::BYTES_VEC[cookies];
            // insert into closelist
            _closelist_small[cookies].insert_after(_closelist_small[cookies].before_begin(), gpuMemoryBlock);
        }
        else{
            // there is memory cache in freelist, use it
            gpuMemoryBlock = _freelist_small[cookies].front();
            _freelist_small[cookies].pop_front();
            // insert into closelist
            _closelist_small[cookies].insert_after(_closelist_small[cookies].before_begin(), gpuMemoryBlock);
        }
    }
}

void GpuMemoryPool::_freeSmallBlock(GpuMemoryBlock& gpuMemoryBlock){
    auto currentCookies = gpuMemoryBlock.cookies;
    assert(currentCookies < BLOCK_SIZE_INDEX::MB_LARGE);

    if (currentCookies == BLOCK_SIZE_INDEX::KB_512){
        _minor_freelist->freeGpuMemoryBlock(gpuMemoryBlock);
        return;
    }

    // get memory block from close list
    auto current = _closelist_small[currentCookies].begin();
    auto previous = _closelist_small[currentCookies].before_begin();
    while(current != _closelist_small[currentCookies].end()){
        if (current->data == gpuMemoryBlock.data && current->size == gpuMemoryBlock.size){
            _freelist_small[currentCookies].insert_after(_freelist_small[currentCookies].before_begin(), *current);
            _closelist_small[currentCookies].erase_after(previous);
            gpuMemoryBlock.data = nullptr;
            gpuMemoryBlock.size = 0;
            break;
        }

        ++current;
        ++previous;
    }

    if (current == _closelist_small[currentCookies].end()){
        throw std::runtime_error(std::string("_freeSmallBlock() GPU Memory Pool Error: can't not find the memory block." ));
    }
}


void GpuMemoryPool::freeBlock(GpuMemoryBlock& gpuMemoryBlock){
    if (gpuMemoryBlock.cookies == BLOCK_SIZE_INDEX::MB_LARGE){
        _freeLargeBlock(gpuMemoryBlock);
    }
    else{
        _freeSmallBlock(gpuMemoryBlock);
    }
}



GpuMemoryPool::~GpuMemoryPool(){
    // free minor and large memory
    delete _minor_freelist;
    delete _large_freelist;

    // free small freelist and closelist memory
    for (int i = 0; i < BLOCK_SIZE_INDEX::MB_LARGE;i++){
        auto current = _freelist_small[i].begin();
        while(current != _freelist_small[i].end()){
            if (current->data != nullptr){
                cudaError_t code = cudaFree(current->data);
                if (code != cudaSuccess){
                    throw std::runtime_error(std::string("~GpuMemoryPool() ") + cudaGetErrorString(code));
                }
            }
            current++;
        }
        _freelist_small[i].clear();

        current = _closelist_small[i].begin();
        while(current != _closelist_small[i].end()){
            if (current->data != nullptr){
                cudaError_t code = cudaFree(current->data);
                if (code != cudaSuccess){
                    throw std::runtime_error(std::string("~GpuMemoryPool() ") + cudaGetErrorString(code));
                }
            }
            current++;
        }
        _closelist_small[i].clear();
    }
}


MemoryFreeList::MemoryFreeList(size_t size_, size_t _pitch_granularity_, size_t _pitch_shfit_) : _freelist_size(size_), _pitch_granularity(_pitch_granularity_), _pitch_shfit(_pitch_shfit_){
    _initializeFreelist();
}

MemoryFreeList::~MemoryFreeList(){
    if (_freelist_begin != nullptr){
        cudaError_t code = cudaFree(_freelist_begin);
        if (code != cudaSuccess){
            throw std::runtime_error(std::string("~MemoryFreeList() ") + cudaGetErrorString(code));
        }
        _freelist.clear();
    }
}

void MemoryFreeList::_initializeFreelist(){
    cudaError_t code = cudaMalloc((void**)&_freelist_begin, _freelist_size);
    if (code != cudaSuccess){
        throw std::runtime_error(std::string("_initializeFreelist() ") + cudaGetErrorString(code) + std::string("_freelist_size = ") + std::to_string(_freelist_size));
    }
    _freelist_end = _freelist_begin + _freelist_size;
}

void MemoryFreeList::mallocGpuMemoryBlock(size_t size, GpuMemoryBlock& gpuMemoryBlock){
    size_t malloc_size = ROUNDUP(size, _pitch_granularity, _pitch_shfit);

    auto cookies = _getBlockCookies(malloc_size);
    if (_freelist.empty()){
        if (_freelist_begin + malloc_size <= _freelist_end){
            gpuMemoryBlock.data = _freelist_begin;
            gpuMemoryBlock.size = malloc_size;
            gpuMemoryBlock.cookies = cookies;
            _freelist.insert_after(_freelist.before_begin(), gpuMemoryBlock);
            return ;
        }
        else{
            throw std::runtime_error("GPU Memory Pool Error: No Sufficient Memory Left");
        }
    }

    auto previous = _freelist.before_begin();
    auto current = _freelist.begin();
    if (_freelist_begin + malloc_size <= current->data){
        // there is Sufficient memory between [_freelist_begin, first_block_start]
        gpuMemoryBlock.data = _freelist_begin;
        gpuMemoryBlock.size = malloc_size;
        gpuMemoryBlock.cookies = cookies;
        _freelist.insert_after(previous, gpuMemoryBlock);
        return ;
    }

    ++current;
    ++previous;
    unsigned char* previous_end;
    if (current == _freelist.end()){
        previous_end = previous->data + previous->size;
        if (previous_end + malloc_size <= _freelist_end){
            gpuMemoryBlock.data = previous_end;
            gpuMemoryBlock.size = malloc_size;
            gpuMemoryBlock.cookies = cookies;
            _freelist.insert_after(previous, gpuMemoryBlock);
        }
        else{
            throw std::runtime_error("GPU Memory Pool Error: No Sufficient Memory Left");
        }
        return ;
    }

    while(current != _freelist.end()){
        previous_end = previous->data + previous->size;
        if (previous_end + malloc_size <= current->data){
            gpuMemoryBlock.data = previous_end;
            gpuMemoryBlock.size = malloc_size;
            gpuMemoryBlock.cookies = cookies;
            _freelist.insert_after(previous, gpuMemoryBlock);
            return ;
        }
        ++current;
        ++previous;
    }


    previous_end = previous->data + previous->size;
    if (previous_end + malloc_size <= _freelist_end){
        gpuMemoryBlock.data = previous_end;
        gpuMemoryBlock.size = malloc_size;
        gpuMemoryBlock.cookies = cookies;
        _freelist.insert_after(previous, gpuMemoryBlock);
        return ;
    }

    throw std::runtime_error("[mallocGpuMemoryBlock END] GPU Memory Pool Error: No Sufficient Memory Left");
}


void MemoryFreeList::freeGpuMemoryBlock(GpuMemoryBlock& gpuMemoryBlock){
    auto current = _freelist.begin();
    auto previous = _freelist.before_begin();
    while(current != _freelist.end()){
        if (gpuMemoryBlock.data == current->data){

            gpuMemoryBlock.data = nullptr;
            gpuMemoryBlock.size = 0;
            _freelist.erase_after(previous);
            return;
        }
        ++previous;
        ++current;
    }
    throw std::runtime_error("GPU Memory Pool Error: No Sufficient Memory Left");
}
}   // utilities
}   // cuda
}   // needle
