#include "kmc_common.h"
#include "cuda_integrator.h"
#include "gillespie_integrator.h"
#include "integrator.h"
#include "ising_model.h"
#include "lattice.h"
#include <chrono>

using namespace kmc;

int main(int /*narg*/, char** /*varg*/) {
  int n = 1 << 9; // grid size
  double J = 1.; // Ising model parameter
  double duration = 100; // how long to integrate for (model time)
  bigint seed = 829102930L;
  
  //Lattice latt(n);         // 1D (n)
  Lattice latt({n, n});      // 2D (n x n)
  //Lattice latt({n, n, n}); // 3D (n x n x n)
  
  // starting state
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      latt.setState({i, j}, (i + j) % 2 ? 1 : -1); // checkerboard
    }
  }
  
  // CPU integrator
  // auto intgr = std::make_unique<GillespieIntegrator<IsingModel>>(seed);
  
  // GPU integrator
  auto intgr = std::make_unique<CudaIntegrator<IsingModel, 2, 8>>(seed); // Ising model, 2D lattice, tiles of 8x8 cells

  // integrate (note that arguments to the model constructor are forwarded via Integrator::integrate)
  Statistics stats = intgr->integrate(latt, duration, {}, J);
  
  std::cout<<"Performed "<<stats.nEvents<<" events in "<<stats.walltime<<" seconds"<<std::endl;
  
  return 0;
}
