#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "common.h"
#include "lattice.h"
#include "model.h"
#include <ctime>
#include <vector>

namespace kmc {

struct Callback {
  int nCalls;
  void (*func)(Lattice&);
};


struct Statistics {
  double walltime = 0; // seconds
  bigint nEvents = 0;
};


template<template<int> typename M, int dim>
class Integrator {

  static_assert(std::is_base_of<Model<M, dim>, M<dim>>::value);
  static_assert(dim==1 || dim==2 || dim==3);
  
public:
  
  virtual ~Integrator() = default;

  template<typename... A>
  Statistics integrate(Lattice& lattice,
		       double duration,
		       Callback callback,
		       A&&... modelArgs) {
    M<dim> model(std::forward<A>(modelArgs)...);
    auto modelParams = model.params();
    init(lattice, modelParams);
    
    Statistics stats {};
    if(callback.nCalls < 1) {
      callback.nCalls = 1;
      callback.func = nullptr;
    }
    double durationPerCallback = duration / callback.nCalls;

    std::clock_t clockStart = std::clock();
    for(int i = 0; i < callback.nCalls; i++) {
      stats.nEvents += integrateImpl(lattice, modelParams, durationPerCallback, model.requestRandom);
      if(callback.func) callback.func(lattice);
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
