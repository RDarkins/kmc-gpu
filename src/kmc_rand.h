#ifndef KMC_RAND_H
#define KMC_RAND_H

#include "kmc_common.h"
#include <random>

namespace kmc {

class HostPRNG {
public:
  HostPRNG(bigint seed);
  double uniform();
  
 private:
  std::mt19937 prng;
  std::uniform_real_distribution<double> dist;
};

} // namespace kmc

#endif // KMC_RAND_H
