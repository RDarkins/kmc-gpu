#ifndef SOS_MODEL_H
#define SOS_MODEL_H

#include <vector>
#include "model.h"

namespace kmc {

/**
  2D solid-on-solid model of crystal growth.
  Irreversible adsorption of atoms with step free energy phi in kT units.
 */
class SOSModel : public Model<SOSModel> {

public:

  SOSModel(double phi)
    : Model<SOSModel>(false)
    , m_phi(phi) {}

  int dimMask() const override {
    return DIM2; // only 2D is supported
  }

  std::vector<double> parameters() const override {
    return {exp(-4.*m_phi), exp(2.*m_phi)};
  }

  CUDA_CALLABLE
  static double propensity(Config const& cfg, double const*const params) {
    const double exp_4phi = params[0];
    const double exp2phi = params[1];
    double p = exp_4phi;
    if(cfg.sx0 > cfg.s) p *= exp2phi;
    if(cfg.sx1 > cfg.s) p *= exp2phi;
    if(cfg.sy0 > cfg.s) p *= exp2phi;
    if(cfg.sy1 > cfg.s) p *= exp2phi;
    return p;
  }

  CUDA_CALLABLE
  static int pickState(Config const& cfg, double const*const /*params*/, double /*rand01*/) {
    return cfg.s + 1;
  }

private:

  double m_phi;

};

} // namespace kmc

#endif // SOS_MODEL_H
