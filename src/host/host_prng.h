#ifndef KMC_RAND_H
#define KMC_RAND_H

#include "../common.h"
#include <random>

namespace kmc {

class HostPRNG {
public:
  HostPRNG(bigint seed) {
    prng.seed(seed);
    prng.discard(700000);
  }
  
  double uniform() {
    return dist(prng);
  }
  
 private:
  std::mt19937 prng;
  std::uniform_real_distribution<double> dist;
};

} // namespace kmc

#endif // KMC_RAND_H
