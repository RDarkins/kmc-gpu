#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include "kmc_common.h"
#include <curand_kernel.h>
#include <vector>
#include <cstdio>

namespace kmc {

#define gpu_assert(ans) { gpu_assertImpl((ans), __FILE__, __LINE__); }
inline void gpu_assertImpl(cudaError_t code, const char *file, int line, bool abort=true) {
  if(code != cudaSuccess) {
    fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

inline __device__ double atomicMinDouble(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
  } while(assumed != old);
  return __longlong_as_double(old);
}

class DevicePRNG {
public:
  __device__
  DevicePRNG() {}

  __device__
  void init(bigint seed, bigint seq) {
    curand_init(seed, seq, 0, &state);
  }

  __device__
  double uniform() {
    return curand_uniform_double(&state);
  }

private:
  curandState state;
};

int const MAX_PARAMS = 32;
extern __constant__ double d_modelParams[];
void loadModelParams(std::vector<double> const& params);

struct Event {
  int idx;
  int i;
  int j;
  int k;
  double t;
};

struct GlobalMem {
  int* stateIn;
  int* stateOut;
  double* tevtIn;
  double* tevtOut;
  DevicePRNG* prngIn;
  DevicePRNG* prngOut;
  double* checksum;
};

struct TileIdx {
  int idx;
  int i;
  int j;
  int k;
};

struct GridParams {
  int nx;
  int ny;
  int nz;
  int NX;
  int NY;
  int NZ;
  int tileVol;
  int nTiles;
  int dim;
};

} // namespace kmc

#endif // CUDA_COMMON_H
