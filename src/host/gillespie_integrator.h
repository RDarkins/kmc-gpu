#ifndef GILLESPIE_INTEGRATOR_H
#define GILLESPIE_INTEGRATOR_H

#include "../integrator.h"
#include "../common.h"
#include "event_lookup.h"
#include "event_lookup_tree.h"
#include "host_prng.h"
#include <memory>
#include <iostream>

namespace kmc {

enum class Scaling { OLogN };


template<template<int> class M, int dim>
class GillespieIntegrator : public Integrator<M, dim> {

public:

  GillespieIntegrator(bigint seed, Scaling scaling = Scaling::OLogN)
    : m_prng(seed), m_scaling(scaling) {}

protected:

  void init(Lattice& lattice,
	    std::vector<double> const& modelParams) override {
    // choose the event lookup method
    switch(m_scaling) {
      break; case Scaling::OLogN: default:
      m_lookup.reset(new EventLookupTree(lattice.volume));
    }

    // compute the propensity of each cell
    for(int idx = 0; idx < lattice.volume; idx++) {
      updateCell<0,0,0>(lattice, modelParams.data(), lattice.fold(idx));
    }
  }

  bigint integrateImpl(Lattice& lattice,
		       std::vector<double> const& modelParams,
		       double duration,
		       bool requestRandom) override {
    double tuntil = lattice.time + duration;
    double const* ptrModelParams = modelParams.data();
    bigint nEvents = 0;

    for(;;) {
      // pick the next event
      auto nextEvt = m_lookup->event(m_prng);
      int idx = std::get<0>(nextEvt);
      double dt = std::get<1>(nextEvt);
      if(idx < 0) break;
      if(lattice.time + dt > tuntil) break;
      
      // perform event
      ++nEvents;
      lattice.time += dt;
      auto site = lattice.site(idx);
      double rand01 = requestRandom ? m_prng.uniform() : 0.;
      lattice.setState(idx, M<dim>::pickNextState(site, ptrModelParams, rand01));
      
      // update local cells
      Idx fidx = lattice.fold(idx);
      updateCell<0,0,0>(lattice, ptrModelParams, fidx);
      updateCell<-1,0,0>(lattice, ptrModelParams, fidx);
      updateCell<1,0,0>(lattice, ptrModelParams, fidx);
      if(dim != 1) {
	updateCell<0,-1,0>(lattice, ptrModelParams, fidx);
	updateCell<0,1,0>(lattice, ptrModelParams, fidx);
	if(dim != 2) {
	  updateCell<0,0,-1>(lattice, ptrModelParams, fidx);
	  updateCell<0,0,1>(lattice, ptrModelParams, fidx);
	}
      }
    }

    return nEvents;
  }

private:

  std::unique_ptr<EventLookup> m_lookup;
  Scaling m_scaling;
  HostPRNG m_prng;

  template<int dx, int dy, int dz>
  void updateCell(Lattice& lattice,
		  double const* ptrModelParams,
		  Idx const& idx0) {
    Idx idx = {idx0.x + dx, idx0.y + dy, idx0.z + dz};
    lattice.wrap(idx);
    auto site = lattice.site(idx);
    double p = M<dim>::propensity(site, ptrModelParams);
    m_lookup->update(lattice.unfold(idx), p);
  }

};
  
} // namespace kmc

#endif // GILLESPIE_INTEGRATOR_H
