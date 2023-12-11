#ifndef ISING_MODEL_H
#define ISING_MODEL_H

#include <vector>
#include "model.h"

namespace kmc {

/**
  Ising model with interaction energy J in kT units.
 */
class IsingModel : public Model<IsingModel> {

public:

  IsingModel(double J)
    : Model<IsingModel>(false)
    , m_J(J) {}

  int dimMask() const override {
    return DIM1 | DIM2 | DIM3; // 1, 2 and 3 dimensions supported
  }

  std::vector<double> parameters() const override {
    return {exp(m_J), exp(-m_J)};
  }

  CUDA_CALLABLE
  static double propensity(Config const& cfg, double const*const params) {
    const double expJ = params[0];
    const double exp_J = params[1];
    double p = 1;
    if(cfg.s == cfg.sx0) p *= exp_J; else p *= expJ;
    if(cfg.s == cfg.sx1) p *= exp_J; else p *= expJ;
    if(cfg.dim != 1) {
      if(cfg.s == cfg.sy0) p *= exp_J; else p *= expJ;
      if(cfg.s == cfg.sy1) p *= exp_J; else p *= expJ;
      if(cfg.dim != 2) {
	if(cfg.s == cfg.sz0) p *= exp_J; else p *= expJ;
	if(cfg.s == cfg.sz1) p *= exp_J; else p *= expJ;
      }
    }
    return p;
  }

  CUDA_CALLABLE
  static int pickState(Config const& cfg, double const*const /*params*/, double /*rand01*/) {
    return -cfg.s;
  }

private:

  double m_J;

};

} // namespace kmc

#endif // ISING_MODEL_H
