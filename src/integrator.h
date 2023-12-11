#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "kmc_common.h"
#include "lattice.h"
#include "model.h"
#include <ctime>
#include <vector>

namespace kmc {

struct Callback {
  int nCalls;
  void (*pfunc)(Lattice&);
};

struct Statistics {
  double walltime = 0; // seconds
  bigint nEvents = 0;
};

template<class M>
class Integrator {

public:

  Integrator() {
    // ensure class M derives from Model<M>
    static_assert(std::is_base_of<Model<M>, M>::value);
  }

  virtual ~Integrator() = default;

  template<typename... A>
  Statistics integrate(Lattice& lattice,
		       double duration,
		       Callback callback,
		       A&&... args) {
    // instantiate template parameter M
    M model(std::forward<A>(args)...);
    assert((model.dimMask() & (1<<lattice.dim)) && "Model incompatible with lattice dimension");
    auto modelParams = model.parameters();

    // init the integrator
    init(lattice, modelParams);

    // integrate the lattice, and invoke the callback function periodically if requested
    Statistics stats {};
    if(callback.nCalls < 1) callback.pfunc = nullptr;
    int nCalls = std::max(callback.nCalls, 1);
    double stepsPerCallback = duration / nCalls;
    std::clock_t clockStart = std::clock();
    for(int i = 0; i < nCalls; i++) {
      stats.nEvents += integrateImpl(lattice, modelParams, stepsPerCallback, model.requestRandom);
      if(callback.pfunc) callback.pfunc(lattice);
    }
    std::clock_t clockEnd = std::clock();
    stats.walltime = (clockEnd - clockStart) / static_cast<double>(CLOCKS_PER_SEC);
    
    cleanup();
    return stats;
  }

protected:

  virtual void init(Lattice&, std::vector<double> const&) {}
  virtual bigint integrateImpl(Lattice&, std::vector<double> const&, double, bool) = 0;
  virtual void cleanup() {}
};
  
} // namespace kmc

#endif // INTEGRATOR_H
