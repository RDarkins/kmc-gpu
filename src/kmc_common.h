#ifndef KMC_COMMON_H
#define KMC_COMMON_H

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace kmc {

using bigint = unsigned long long;

enum {X0 = 0, X1, Y0, Y1, Z0, Z1};

struct Idx {
  int x;
  int y;
  int z;
};

struct Config {
  int s;
  int sx0;
  int sx1;
  int sy0;
  int sy1;
  int sz0;
  int sz1;
  int dim;
};

inline bool isPowerOf2(int x) {
  return x > 0 && !(x & (x - 1));
}

} // namespace kmc

#endif // KMC_COMMON_H
