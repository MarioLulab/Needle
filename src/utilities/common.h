#ifndef _COMMON_H
#define _COMMON_H

#include <vector>
#include <cassert>

namespace needle {
namespace cuda {
namespace utilities{

using size_t = std::size_t;

#define ROUNDUP(total, grain, shift) (((total + grain - 1) >> shift) << shift)


const size_t PITCH_GRANULARITY_LARGE = (0x1u << 20); // 1 MB aligned
const size_t PITCH_SHIFT_LARGE = 20;
const size_t PITCH_GRANULARITY_MINOR = (0x1u << 4); // 16 B aligned
const size_t PITCH_SHIFT_MINOR = 4;


const size_t MAX_BYTES = ((0x1 << 26) | (0x1 << 25) | (0x1 << 22)); // 100MB. max allocated memory block that _mallocSmallBlock can malloc
enum BLOCK_SIZE_INDEX{
	KB_512 = 0,
	MB_1,	
    MB_2,
    MB_4,
    MB_8,
    MB_16,
    MB_32,
    MB_64,
    MB_80,
    MB_100,
    MB_LARGE,
};

namespace CONST_VALUE{
    using namespace needle::cuda::utilities;

    const size_t BYTES_512_KB = (0x1 << 19);
    const size_t BYTES_1_MB = (0x1 << 20);
    const size_t BYTES_2_MB = (0x1 << 21);
    const size_t BYTES_4_MB = (0x1 << 22);
    const size_t BYTES_8_MB = (0x1 << 23);
    const size_t BYTES_16_MB = (0x1 << 24);
    const size_t BYTES_32_MB = (0x1 << 25);
    const size_t BYTES_64_MB = (0x1 << 26);
    const size_t BYTES_80_MB = ((0x1 << 26) + (0x1 << 24));
    const size_t BYTES_100_MB = ((0x1 << 26) + (0x1 << 25) + (0x1 << 22));
    const size_t BYTES_VEC[BLOCK_SIZE_INDEX::MB_LARGE] = {
		BYTES_512_KB, 
		BYTES_1_MB, BYTES_2_MB, BYTES_4_MB, BYTES_8_MB, BYTES_16_MB,
        BYTES_32_MB, BYTES_64_MB, BYTES_80_MB,
        BYTES_100_MB
    };
} // CONST_VALUE

inline static BLOCK_SIZE_INDEX _getBlockCookies(size_t size){
    BLOCK_SIZE_INDEX ret;
    if (size <= CONST_VALUE::BYTES_512_KB){
		ret = BLOCK_SIZE_INDEX::KB_512;
	}
	else if (size <= CONST_VALUE::BYTES_1_MB){
		ret = BLOCK_SIZE_INDEX::MB_1;
	}
	else if (size <= CONST_VALUE::BYTES_2_MB){
        ret = BLOCK_SIZE_INDEX::MB_2;
    }
    else if (size <= CONST_VALUE::BYTES_4_MB){
        ret = BLOCK_SIZE_INDEX::MB_4;
    }
    else if (size <= CONST_VALUE::BYTES_8_MB){
        ret = BLOCK_SIZE_INDEX::MB_8;
    }
    else if (size <= CONST_VALUE::BYTES_16_MB){
        ret = BLOCK_SIZE_INDEX::MB_16;
    }
    else if (size <= CONST_VALUE::BYTES_32_MB){
        ret = BLOCK_SIZE_INDEX::MB_32;
    }
    else if (size <= CONST_VALUE::BYTES_64_MB){
        ret = BLOCK_SIZE_INDEX::MB_64;
    }
    else if (size <= CONST_VALUE::BYTES_80_MB){
        ret = BLOCK_SIZE_INDEX::MB_80;
    }
    else if (size <= CONST_VALUE::BYTES_100_MB){
        ret = BLOCK_SIZE_INDEX::MB_100;
    }
    else{
        ret = BLOCK_SIZE_INDEX::MB_LARGE;
    }

   return ret; 
}
}   // utilities
}   // cuda
}   // needle

#endif