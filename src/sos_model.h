#ifndef SOS_MODEL_H
#define SOS_MODEL_H

#include "model.h"

namespace kmc {

/**
  2D solid-on-solid model of crystal growth. Irreversible adsorption of atoms
  with step free energy phi in kT units.
 */
template<int dim>
class SOSModel : public Model<SOSModel, dim> {

  static_assert(dim == 2, "SOS Model supports 2 dimensions only.");

public:

  SOSModel(double phi)
    : Model<SOSModel>(false)
    , m_phi(phi) {}

  std::vector<double> params() const override {
    return {exp(-4.*m_phi), exp(2.*m_phi)};
  }

  CUDA_CALLABLE
  static double propensity(Site const& site, double const* params) {
    const double exp_4phi = params[0];
    const double exp2phi = params[1];
    double p = exp_4phi;
    if(site.sx0 > site.s) p *= exp2phi;
    if(site.sx1 > site.s) p *= exp2phi;
    if(site.sy0 > site.s) p *= exp2phi;
    if(site.sy1 > site.s) p *= exp2phi;
    return p;
  }

  CUDA_CALLABLE
  static int pickNextState(Site const& site, double const* /*params*/, double /*rand01*/) {
    return site.s + 1;
  }

private:

  double m_phi;

};

} // namespace kmc

#endif // SOS_MODEL_H
