#include "cuda_common.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <cfloat>

namespace kmc {
  
__constant__ double d_modelParams[MAX_PARAMS];

void loadModelParams(std::vector<double> const& params) {
  assert(params.size() <= MAX_PARAMS);
  gpu_assert(cudaMemcpyToSymbol(d_modelParams, params.data(), params.size() * sizeof(double)));
}

} // namespace kmc
