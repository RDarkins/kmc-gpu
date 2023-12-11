#include "kmc_rand.h"

namespace kmc {

HostPRNG::HostPRNG(bigint seed) {
  prng.seed(seed);
  prng.discard(700000);
}

double HostPRNG::uniform() {
  return dist(prng);
}

} // namespace kmc
