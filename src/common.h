#ifndef KMC_COMMON_H
#define KMC_COMMON_H

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace kmc {

using bigint = unsigned long long;


inline constexpr bool isPowerOf2(int n) {
  return n > 0 && !(n & (n - 1));
}


struct Site {
  int s;   // state at (x, y, z)
  int sx0; // state at (x-1,y,z)
  int sx1; // state at (x+1,y,z)
  int sy0; // state at (x,y-1,z)
  int sy1; // state at (x,y+1,z)
  int sz0; // state at (x,y,z-1)
  int sz1; // state at (x,y,z+1)
};
  
} // namespace kmc

#endif // KMC_COMMON_H
