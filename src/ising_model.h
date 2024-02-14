#ifndef ISING_MODEL_H
#define ISING_MODEL_H

#include "model.h"

namespace kmc {

/**
  Ising model with interaction energy J in kT units.
 */
template<int dim>
class IsingModel : public Model<IsingModel, dim> {
  
  static_assert(dim==1 || dim==2 || dim==3);
  
public:

  IsingModel(double J)
    : Model<IsingModel, dim>(false)
    , m_J(J) {}
  
  std::vector<double> params() const override {
    return {exp(m_J), exp(-m_J)};
  }

  CUDA_CALLABLE
  static double propensity(Site const& site, double const* params) {
    const double expJ = params[0];
    const double exp_J = params[1];
    double p = 1;
    if(site.s == site.sx0) p *= exp_J; else p *= expJ;
    if(site.s == site.sx1) p *= exp_J; else p *= expJ;
    if(dim != 1) {
      if(site.s == site.sy0) p *= exp_J; else p *= expJ;
      if(site.s == site.sy1) p *= exp_J; else p *= expJ;
      if(dim != 2) {
	if(site.s == site.sz0) p *= exp_J; else p *= expJ;
	if(site.s == site.sz1) p *= exp_J; else p *= expJ;
      }
    }
    return p;
  }

  CUDA_CALLABLE
  static int pickNextState(Site const& site, double const* /*params*/, double /*rand01*/) {
    return -site.s;
  }

private:

  double m_J;

};

} // namespace kmc

#endif // ISING_MODEL_H
