#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include "utilities/memory_pool.h"
#include "utilities/singleton.h"
namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
#define L 8
#define S 4

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;

struct CudaArray {
  CudaArray(const size_t size) {
    needle::singleton::Singleton<needle::cuda::utilities::GpuMemoryPool>::instance().mallocBlock(size * ELEM_SIZE, m_gpuMemoryBlock);
    this->size = size;
    // ptr = (scalar_t*)((void*)(m_gpuMemoryBlock.data));
    ptr = reinterpret_cast<scalar_t*>(m_gpuMemoryBlock.data);

    #ifdef MEMORY_POOL_DEBUG
    printf("CudaArray() : m_gpuMemoryBlock.data = 0x%lx, m_gpuMemoryBlock.size = 0x%lx, ptr = 0x%lx, size = 0x%lx\n", m_gpuMemoryBlock.data, m_gpuMemoryBlock.size, this->ptr, this->size);
    #endif
  }
  ~CudaArray() {
    #ifdef MEMORY_POOL_DEBUG
    printf("~CudaArray() : m_gpuMemoryBlock.data = 0x%lx, m_gpuMemoryBlock.size = 0x%lx, ptr = 0x%lx, size = 0x%lx\n", m_gpuMemoryBlock.data, m_gpuMemoryBlock.size, this->ptr, this->size);
    #endif

    needle::singleton::Singleton<needle::cuda::utilities::GpuMemoryPool>::instance().freeBlock(m_gpuMemoryBlock);
    ptr = nullptr;
  }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
  needle::cuda::utilities::GpuMemoryBlock m_gpuMemoryBlock;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides




__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  
  size_t gid_ = gid;
  if (gid < size){
    int currentInputIdx = 0;
    for (int axis = shape.size - 1; axis >= 0; --axis){
      currentInputIdx += (gid % shape.data[axis]) * strides.data[axis];
      gid = gid / shape.data[axis];
    }
    out[gid_] = a[currentInputIdx + offset];    
  }
  
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EWiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset){

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    size_t gid_ = gid;
    int currentInputIdx = 0;
    for (int axis = shape.size - 1; axis >= 0; --axis){
      currentInputIdx += (gid % shape.data[axis]) * strides.data[axis];
      gid = gid / shape.data[axis];
    }
    out[currentInputIdx + offset] = a[gid_];    
  }

}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  
  CudaDims dim = CudaOneDim(out->size);
  size_t size = a.size;
  EWiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, size, VecToCuda(shape),
                                        VecToCuda(strides), offset);
  
}



__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    // size_t gid_ = gid;
    int currentInputIdx = 0;
    for (int axis = shape.size - 1; axis >= 0; --axis){
      currentInputIdx += (gid % shape.data[axis]) * strides.data[axis];
      gid = gid / shape.data[axis];
    }
    out[currentInputIdx + offset] = val;    
  }
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                        VecToCuda(strides), offset);
  
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}
void EwiseMul_(const CudaArray& a, const CudaArray& b, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}
void EwiseDiv_(const CudaArray& a, const CudaArray& b, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}
void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = std::pow(a[gid], val);
}
void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = a[gid] > b[gid] ? a[gid] : b[gid];
  }
}
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = a[gid] > val ? a[gid] : val;
  }
}
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = a[gid] == b[gid] ? (scalar_t)1.0 : (scalar_t)0.0;
  }
}
void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = a[gid] == val ? (scalar_t)1.0 : (scalar_t)0.0;
  }
}
void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = a[gid] >= b[gid] ? (scalar_t)1.0 : (scalar_t)0.0;
  }
}
void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = a[gid] >= val ? (scalar_t)1.0 : (scalar_t)0.0;
  }
}
void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = std::log(a[gid]);
  }
}
void EwiseLog(const CudaArray& a, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = std::exp(a[gid]);
  }
}
void EwiseExp(const CudaArray& a, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = std::tanh(a[gid]);
  }
}
void EwiseTanh(const CudaArray& a, CudaArray* out)
{
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// __global__ void MatmulKernel(const scalar_t* A, const scalar_t* B, scalar_t* C, 
//                              const size_t S, const size_t L, 
//                              size_t M, size_t N, size_t P){
__global__ void MatmulKernel(const scalar_t* A, const scalar_t* B, scalar_t* C, 
                             size_t M, size_t N, size_t P){
  __shared__ scalar_t sA[L][S];
  __shared__ scalar_t sB[S][L];

  scalar_t c[TILE][TILE] = {0};

  size_t yBlock = blockIdx.y;
  size_t xBlock = blockIdx.x;
  size_t yBase = blockIdx.y * blockDim.y + threadIdx.y;
  size_t xBase = blockIdx.x * blockDim.x + threadIdx.x;

    for (int ko = 0; ko < N; ko += S){
      __syncthreads();
      // Cooperative Fetching. Global Mem -> Share Mem 
      size_t nThreads = blockDim.x * blockDim.y;
      size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
      for (int j = 0; j < L * S / nThreads; ++j){
        size_t y_sB = (j*nThreads + tid) / L;
        size_t x_sB = (j*nThreads + tid) % L;

        size_t x_sA = (j*nThreads + tid) / L;
        size_t y_sA = (j*nThreads + tid) % L; 
        if ((yBlock * L + y_sA) < M && \
            (ko + x_sA) < N ){
            sA[y_sA][x_sA] = A[(yBlock * L + y_sA) * N + ko + x_sA];
          }

        if ((ko + y_sB) < N && \
            (xBlock * L + x_sB) < P){
            sB[y_sB][x_sB] = B[(ko + y_sB) * P + xBlock * L + x_sB];            
          }
      }
      __syncthreads();
      for (int ki = 0; ki < S && (ko + ki) < N; ++ki){  
        // Share Mem -> Register
        scalar_t a[TILE] = {0};
        scalar_t b[TILE] = {0};
        for (int i = 0; i < TILE; ++i){
          if ( (yBase * TILE + i) < M){
            a[i] = sA[threadIdx.y * TILE + i][ki];
          }
          if ( (xBase * TILE + i) < P){
            b[i] = sB[ki][threadIdx.x * TILE + i];
          }
        }

        for (int y = 0; y < TILE; ++y){
          for (int x = 0; x < TILE; ++x){
            c[y][x] += a[y] * b[x];
          }
        }

      }

    }

    // Register -> Global Mem. 
    for (int x = 0; x < TILE; ++x){
      for (int y = 0; y < TILE; ++y){
        if ((yBase * TILE + y < M) && (xBase * TILE + x < P)){
          C[(yBase * TILE + y) * P + xBase * TILE + x] = c[y][x];
        }
      }
    }

}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  
  dim3 blocks {(P - 1 + L)/ L, (M - 1 + L)/ L, 1};
  dim3 threads {L / TILE, L / TILE, 1};
  MatmulKernel<<<blocks, threads>>>(a.ptr, b.ptr, out->ptr, 
                                    M, N, P);
  
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    // find max
    scalar_t max = a[gid * reduce_size];
    for (int i = 0; i < reduce_size; ++i){
      if (max < a[i + gid * reduce_size]){
        max = a[i + gid * reduce_size];
      }
    }
    out[gid] = max;
  }
}


void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    scalar_t sum = 0.0;
    for (int i = 0; i < reduce_size; ++i){
      sum += a[gid * reduce_size + i];
    }
    out[gid] = sum;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(std::string("to_numpy") + cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    #ifdef MEMORY_POOL_DEBUG
    printf("from_numpy() : out->m_gpuMemoryBlock.data = 0x%lx, out->ptr = 0x%lx, size = 0x%lx\n", out->m_gpuMemoryBlock.data, out->ptr, out->size);
    #endif

    cudaError_t err = cudaMemcpy(out->ptr, a.request().ptr, (out->size) * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(std::string("from_numpy") + cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul_);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv_);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}