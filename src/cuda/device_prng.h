#ifndef DEVICE_RAND_H
#define DEVICE_RAND_H

#include <cuda_runtime_api.h>

namespace kmc {

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

} // namespace kmc

#endif // DEVICE_RAND_H
