#ifndef CUDA_INTEGRATOR_H
#define CUDA_INTEGRATOR_H

#include "cuda_common.h"
#include "cuda_bridge.h"
#include "cuda_tiling.h"
#include "integrator.h"
#include "kmc_common.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <iostream>
#include <limits>

namespace kmc {

template<class M, int dim, int tw>
class CudaIntegrator : public Integrator<M> {

public:

  CudaIntegrator(bigint seed)
    : m_seed(seed) {}

protected:

  void init(Lattice& lattice,
	    std::vector<double> const& modelParams) override {
    assert(dim == lattice.dim);

    // grid params
    gp.dim = lattice.dim;
    gp.nx = lattice.grid.x;
    gp.ny = lattice.grid.y;
    gp.nz = lattice.grid.z;
    gp.NX = gp.nx / tw;
    gp.NY = gp.ny / tw;
    gp.NZ = gp.nz / tw;
    gp.tileVol = dim == 1 ? tw : (dim == 2 ? (tw * tw) : (tw * tw * tw));
    gp.nTiles = dim == 1 ? gp.NX : (dim == 2 ? (gp.NX * gp.NY) : (gp.NX * gp.NY * gp.NZ));

    // grid and block sizes
    int nSMs;
    cudaDeviceGetAttribute(&nSMs, cudaDevAttrMultiProcessorCount, 0);
    m_gridSize = (unsigned)(nSMs);
    m_blockSize = 64; 

    // allocate
    int v = lattice.volume;
    gpu_assert(cudaMalloc((void**)&glob.stateIn, v * sizeof(int)));
    gpu_assert(cudaMalloc((void**)&glob.stateOut, v * sizeof(int)));
    gpu_assert(cudaMalloc((void**)&glob.tevtIn, v * sizeof(double)));
    gpu_assert(cudaMalloc((void**)&glob.tevtOut, v * sizeof(double)));
    gpu_assert(cudaMalloc((void**)&glob.prngIn, v * sizeof(DevicePRNG)));
    gpu_assert(cudaMalloc((void**)&glob.prngOut, v * sizeof(DevicePRNG)));
    gpu_assert(cudaMalloc((void**)&glob.checksum, gp.nTiles * 2 * lattice.dim * sizeof(double)));
    gpu_assert(cudaMalloc((void**)&nEvents, sizeof(bigint)));
    double* d_tmin;
    gpu_assert(cudaMalloc((void**)&d_tmin, sizeof(double)));
    double maxDouble = std::numeric_limits<double>::max();
    cudaMemcpy(d_tmin, &maxDouble, sizeof(double), cudaMemcpyHostToDevice);

    // copy to device mem
    loadModelParams(modelParams);
    tiles::bridge::loadLattice<tw>(glob.stateIn, lattice, gp);

    // init tevt and prng
    tiles::initDeviceState<M, tw, dim><<<m_gridSize, m_blockSize>>>(lattice.time, m_seed, glob, gp, d_tmin);
    cudaDeviceSynchronize();
    gpu_assert(cudaPeekAtLastError());

    // initial dt
    double tmin;
    cudaMemcpy(&tmin, d_tmin, sizeof(double), cudaMemcpyDeviceToHost);
    m_dt = (tmin - lattice.time);
    cudaFree(d_tmin);
  }

  bigint integrateImpl(Lattice& lattice,
		       std::vector<double> const& modelParams,
		       double duration,
		       bool requestRandom) override {
    assert(duration > 0);
    double tuntil = lattice.time + duration;
    bool overshot = false;
    bigint nSuccessfulEvents = 0;
    while(lattice.time < tuntil) {
      m_dt = std::min(m_dt, tuntil - lattice.time);

      // integrate by one time step
      gpu_assert(cudaMemset(nEvents, 0, sizeof(bigint)));
      tiles::integrate<M, tw, dim><<<m_gridSize, m_blockSize>>>(glob, lattice.time + m_dt, requestRandom, gp, nEvents);
      cudaDeviceSynchronize();
      gpu_assert(cudaPeekAtLastError());
      bigint h_nEvents;
      gpu_assert(cudaMemcpy(&h_nEvents, nEvents, sizeof(bigint), cudaMemcpyDeviceToHost));

      // check for errors
      if(checksumPass()) {
	lattice.time += m_dt;
	std::swap(glob.stateIn, glob.stateOut);
	std::swap(glob.tevtIn, glob.tevtOut);
	std::swap(glob.prngIn, glob.prngOut);
	nSuccessfulEvents += h_nEvents;
	
	if(!overshot)
	  m_dt *= 10.0;
	else
	  m_dt *= 1.03;
      } else {
	m_dt *= 0.5;
	overshot = true;
      }
    }

    // copy lattice from gpu back to host
    tiles::bridge::unloadLattice<tw>(lattice, glob.stateIn, gp);

    return nSuccessfulEvents;
  }

  void cleanup() override {
    cudaFree(glob.stateIn);
    cudaFree(glob.stateOut);
    cudaFree(glob.tevtIn);
    cudaFree(glob.tevtOut);
    cudaFree(glob.prngIn);
    cudaFree(glob.prngOut);
    cudaFree(glob.checksum);
    cudaFree(nEvents);
  }

private:

  bigint m_seed;
  double m_dt;
  dim3 m_gridSize;
  dim3 m_blockSize;
  GlobalMem glob;
  GridParams gp;
  bigint* nEvents;

  bool checksumPass() {
    int* d_fail;
    gpu_assert(cudaMalloc((void**)&d_fail, sizeof(int)));
    gpu_assert(cudaMemset(d_fail, 0, sizeof(int)));

    tiles::checksumValidation<tw, dim><<<m_gridSize, m_blockSize>>>(glob.checksum, d_fail, gp);
    cudaDeviceSynchronize();

    int fail;
    gpu_assert(cudaMemcpy(&fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_fail);
    gpu_assert(cudaPeekAtLastError());

    return !fail;
  }

};
  
} // namespace kmc

#endif // CUDA_INTEGRATOR_H
