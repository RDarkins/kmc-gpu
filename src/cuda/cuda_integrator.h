#ifndef CUDA_INTEGRATOR_H
#define CUDA_INTEGRATOR_H

#ifdef __CUDACC__

#include "bridge.h"
#include "common.h"
#include "tiling.h"
#include "../integrator.h"
#include "../common.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>

namespace kmc {

template<template<int> class M, int dim, int tw>
class CudaIntegrator : public Integrator<M, dim> {
  
  static_assert(isPowerOf2(tw), "Tile width must be a power of 2.");

public:

  CudaIntegrator(bigint seed)
    : m_seed(seed) {}

protected:

  void init(Lattice& lattice, std::vector<double> const& modelParams) override {
    assert(dim == lattice.dim);
    buildGridParams(lattice);
    determineThreadDistribution();
    allocateGlobalMemory(lattice);
    transferToGlobalMemory(lattice, modelParams);
    initDeviceState(lattice.time);
  }

  bigint integrateImpl(Lattice& lattice,
		       std::vector<double> const& modelParams,
		       double duration,
		       bool requestRandom) override {
    assert(duration > 0);
    double tuntil = lattice.time + duration;
    bool hasNeverOvershot = true;
    bigint nSuccessfulEvents = 0;
    
    while(lattice.time < tuntil) {
      m_dt = std::min(m_dt, tuntil - lattice.time);

      bigint nEvents;
      bool errorFree = advanceToTime(lattice.time + m_dt, requestRandom, nEvents);

      if(errorFree) {
	commitStep();
	
	lattice.time += m_dt;
	nSuccessfulEvents += nEvents;
	
	if(hasNeverOvershot)
	  m_dt *= 10.0;
	else
	  m_dt *= 1.03;
      } else {
	rejectStep();
	
	m_dt *= 0.5;
	hasNeverOvershot = false;
      }
    }

    transferLatticeBackToHost(lattice);
    
    return nSuccessfulEvents;
  }

  void cleanup() override {
    cudaFree(m_glob.stateFront);
    cudaFree(m_glob.stateBack);
    cudaFree(m_glob.tevtFront);
    cudaFree(m_glob.tevtBack);
    cudaFree(m_glob.prngFront);
    cudaFree(m_glob.prngBack);
    cudaFree(m_glob.checksum);
    cudaFree(m_nEvents);
  }

private:

  bigint m_seed;
  double m_dt;
  dim3 m_gridSize;
  dim3 m_blockSize;
  GlobalMem m_glob;
  GridParams m_gp;
  bigint* m_nEvents;

  void buildGridParams(Lattice const& lattice) {
    m_gp.dim = lattice.dim;
    m_gp.nx = lattice.gridSize.x;
    m_gp.ny = lattice.gridSize.y;
    m_gp.nz = lattice.gridSize.z;
    m_gp.NX = m_gp.nx / tw;
    m_gp.NY = m_gp.ny / tw;
    m_gp.NZ = m_gp.nz / tw;
    m_gp.tileVol = dim == 1 ? tw : (dim == 2 ? (tw * tw) : (tw * tw * tw));
    m_gp.nTiles = dim == 1 ? m_gp.NX : (dim == 2 ? (m_gp.NX * m_gp.NY) : (m_gp.NX * m_gp.NY * m_gp.NZ));
  }
  
  void determineThreadDistribution() {
    // 64 threads per block, one block per SM
    int nSMs;
    cudaDeviceGetAttribute(&nSMs, cudaDevAttrMultiProcessorCount, 0);
    m_gridSize = (unsigned)(nSMs);
    m_blockSize = 64;
  }
  
  void allocateGlobalMemory(Lattice const& lattice) {
    int v = lattice.volume;
    gpuAssert(cudaMalloc((void**)&m_glob.stateFront, v * sizeof(int)));
    gpuAssert(cudaMalloc((void**)&m_glob.stateBack, v * sizeof(int)));
    gpuAssert(cudaMalloc((void**)&m_glob.tevtFront, v * sizeof(double)));
    gpuAssert(cudaMalloc((void**)&m_glob.tevtBack, v * sizeof(double)));
    gpuAssert(cudaMalloc((void**)&m_glob.prngFront, v * sizeof(DevicePRNG)));
    gpuAssert(cudaMalloc((void**)&m_glob.prngBack, v * sizeof(DevicePRNG)));
    gpuAssert(cudaMalloc((void**)&m_glob.checksum, m_gp.nTiles * 2 * lattice.dim * sizeof(double)));
    gpuAssert(cudaMalloc((void**)&m_nEvents, sizeof(bigint)));
  }

  void transferToGlobalMemory(Lattice const& lattice, std::vector<double> const& modelParams) {
    // transfer model params
    assert(modelParams.size() <= MAX_PARAMS);
    gpuAssert(cudaMemcpyToSymbol(d_modelParams, modelParams.data(), modelParams.size() * sizeof(double)));

    // transfer lattice
    tiles::bridge::loadLattice<tw>(m_glob.stateFront, lattice, m_gp);
  }
  
  void initDeviceState(double initial_time) {
    double* d_tmin;
    gpuAssert(cudaMalloc((void**)&d_tmin, sizeof(double)));
    double maxDouble = std::numeric_limits<double>::max();
    cudaMemcpy(d_tmin, &maxDouble, sizeof(double), cudaMemcpyHostToDevice);
    
    // init tevt and prng and compute the time of the first event (d_tmin)
    tiles::initDeviceState<M<dim>, tw, dim><<<m_gridSize, m_blockSize>>>(initial_time, m_seed, m_glob, m_gp, d_tmin);
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    // initial time step dt
    double tmin;
    cudaMemcpy(&tmin, d_tmin, sizeof(double), cudaMemcpyDeviceToHost);
    m_dt = (tmin - initial_time);
    cudaFree(d_tmin);
  }

  bool advanceToTime(double tuntil, bool requestRandom, bigint& nEvents) {
    gpuAssert(cudaMemset(m_nEvents, 0, sizeof(bigint)));

    tiles::integrate<M<dim>, tw, dim><<<m_gridSize, m_blockSize>>>(m_glob, tuntil, requestRandom, m_gp, m_nEvents);

    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());
    gpuAssert(cudaMemcpy(&nEvents, m_nEvents, sizeof(bigint), cudaMemcpyDeviceToHost));

    bool errorFree = checksumPass();
    return errorFree;
  }

  bool checksumPass() {
    int* d_fail;
    gpuAssert(cudaMalloc((void**)&d_fail, sizeof(int)));
    gpuAssert(cudaMemset(d_fail, 0, sizeof(int)));

    tiles::checksumValidation<tw, dim><<<m_gridSize, m_blockSize>>>(m_glob.checksum, d_fail, m_gp);

    cudaDeviceSynchronize();
    int fail;
    gpuAssert(cudaMemcpy(&fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_fail);
    gpuAssert(cudaPeekAtLastError());

    return !fail;
  }

  inline void commitStep() {
    // swap front and back buffers
    std::swap(m_glob.stateFront, m_glob.stateBack);
    std::swap(m_glob.tevtFront, m_glob.tevtBack);
    std::swap(m_glob.prngFront, m_glob.prngBack);
  }

  inline void rejectStep() {}

  void transferLatticeBackToHost(Lattice& lattice) {
    tiles::bridge::unloadLattice<tw>(lattice, m_glob.stateFront, m_gp);
  }

};
  
} // namespace kmc

#else

#error "This class requires the CUDA compiler."

#endif // __CUDA_CC__

#endif // CUDA_INTEGRATOR_H
