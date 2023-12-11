#ifndef MODEL_H
#define MODEL_H

#include "kmc_common.h"
#include <vector>

namespace kmc {
  
enum {DIM1=1<<1,
      DIM2=1<<2,
      DIM3=1<<3};

/**
 * Template arg T must derive from Model<T> (curiously recurring template pattern).
 *  
 * T must define two static functions:
 *   static double propensity(const Config&, double const*const);
 *   static int pickState(const Config&, double const*const, double);
 *  
 * This approach allows for safe and efficient polymorphism on the GPU.
 *
 * See ising_model.h or sos_model.h for examples of deriving classes.
 */
template<typename T>
class Model {
public:

  const bool requestRandom;

  Model(bool requestRandom = true)
    : requestRandom(requestRandom) {
    static_assert(std::is_base_of<Model<T>, T>::value);
    [[maybe_unused]]
    auto dummy1 = static_cast<double (*)(const Config&, double const*const)>(T::propensity);
    [[maybe_unused]]
    auto dummy2 = static_cast<int (*)(const Config&, double const*const, double)>(T::pickState);
  }

  virtual ~Model() = default;
  virtual int dimMask() const = 0;
  virtual std::vector<double> parameters() const { return std::vector<double>(); }
};

} // namespace kmc

#endif // MODEL_H
