#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include "../common.h"
#include <cassert>
#include <curand_kernel.h>
#include <vector>
#include <cstdio>

namespace kmc {

#define gpuAssert(ans) { gpuAssertImpl((ans), __FILE__, __LINE__); }

inline void gpuAssertImpl(cudaError_t code, char const* file, int line, bool abort = true) {
  if(code != cudaSuccess) {
    fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}


int const MAX_PARAMS = 32;
inline __constant__ double d_modelParams[MAX_PARAMS];


struct TileIdx {
  int idx;
  int i;
  int j;
  int k;
};


struct GlobalMem {
  int* stateFront;
  int* stateBack;
  double* tevtFront;
  double* tevtBack;
  class DevicePRNG* prngFront;
  class DevicePRNG* prngBack;
  double* checksum;
};


struct GridParams {
  int nx; // number of cells along x
  int ny; // number of cells along y
  int nz; // number of cells along z
  int NX; // number of tiles along x
  int NY; // number of tiles along y
  int NZ; // number of tiles along z
  int tileVol; // volume of each tile
  int nTiles; // number of tiles (NX*NY*NZ)
  int dim; // dimensions of space
};

} // namespace kmc

#endif // CUDA_COMMON_H
