#include "common.h"
#include "cuda/cuda_integrator.h"
#include "host/gillespie_integrator.h"
#include "integrator.h"
#include "ising_model.h"
#include "lattice.h"
#include <chrono>
#include <fstream>

using namespace kmc;

int main(int /*narg*/, char** /*varg*/)
{
  int n = 1 << 9; // grid length
  double isingParamJ = 1.;
  double duration = 100; // model time
  bigint prngSeed = 829102930L;
  
  //Lattice lattice(n);         // 1D (n)
  Lattice lattice({n, n});      // 2D (n x n)
  //Lattice lattice({n, n, n}); // 3D (n x n x n)
  
  // checkerboard starting state
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      lattice.setState({i, j}, (i + j) % 2 ? 1 : -1);
    }
  }
  
  // GPU integrator (template args: model, lattice dimension, tile width)
  auto intgr = std::make_unique<CudaIntegrator<IsingModel, 2, 8>>(prngSeed);

  // integrate (arguments to the model class are forwarded via Integrator::integrate)
  Statistics stats = intgr->integrate(lattice, duration, {}, isingParamJ);
  std::cout << "Performed " << stats.nEvents << " events in " << stats.walltime << " seconds" << std::endl;

  // save lattice state to file
  std::ofstream outputFile("lattice.final", std::ios::out | std::ios::app);
  if(outputFile.is_open()) {
    outputFile << lattice;
    outputFile.close();
  }
  
  return 0;
}
