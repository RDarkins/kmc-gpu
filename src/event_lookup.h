#ifndef EVENT_LOOKUP_H
#define EVENT_LOOKUP_H

#include <tuple>
#include "kmc_rand.h"

namespace kmc {

class EventLookup {
public:
  virtual ~EventLookup() = default;

  /**
   * Set propensity of cell i to p.
   */
  virtual void update(int i, double p) = 0;

  /**
   * Randomly pick event with probability proportional to propensity,
   * return {cell idx, time step dt}.
   */
  virtual std::tuple<int, double> event(HostPRNG&) const = 0;
};

} // namespace kmc

#endif // EVENT_LOOKUP_H
