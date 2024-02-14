#ifndef MODEL_H
#define MODEL_H

#include "common.h"
#include <vector>

namespace kmc {

/**
 * Template arg T<dim> must derive from Model<T, dim>. This use of the CRTP
 * allows for safe and efficient polymorphism on the GPU.
 * 
 * See ising_model.h and sos_model.h for examples of deriving classes.
 */
template<template<int> typename T, int dim>
class Model {
public:

  const bool requestRandom; // see pickNextState()

  Model(bool requestRandom = true)
    : requestRandom(requestRandom) {
    static_assert(std::is_base_of<Model<T, dim>, T<dim>>::value);
  }

  virtual ~Model() = default;
  
  /**
   * Model parameters or other useful precomputed values to be passed
   * to propensity() and pickNextState().
   */
  virtual std::vector<double> params() const { return std::vector<double>(); }

  /**
   * The average rate (times any constant) at which the site changes state.
   */
  CUDA_CALLABLE
  static double propensity(Site const& site, double const* params) {
    return T<dim>::propensity(site, params);
  }

  /**
   * An event has occurred at this site, pick its new state. rand01 will be a
   * random number in [0,1) if requestRandom is true, otherwise it will be 0.
   */
  CUDA_CALLABLE
  static int pickNextState(Site const& site, double const* params, double rand01) {
    return T<dim>::pickNextState(site, params, rand01);
  }
};

} // namespace kmc

#endif // MODEL_H
