/**
 * This file contains the GPU implementation of the tiling parts
 * of the KMC algorithm. Specific functionality included:
 *  - initialise the lattice state
 *  - integrate by one time step
 *  - build and validate the checksums
 */

#ifndef CUDA_TILING_H
#define CUDA_TILING_H

#include "cuda_common.h"
#include "cuda_bridge.h"
#include "kmc_common.h"
#include "lattice.h"
#include <cuda_runtime_api.h>
#include <cfloat>
#include <vector>

namespace kmc {
namespace tiles {

/**
 * The config of 3, 3x3 or 3x3x3 tiles to be stored in SM registers
 */
template<typename M, int tw, int dim>
struct LocalTiles {
  constexpr static int volume= dim == 1 ? (3 * tw) : (dim == 2 ? (9 * tw * tw) : (27 * tw * tw * tw));
  constexpr static int dx = 1; // (1,0,0) in unfolded index space
  constexpr static int dy = 3 * tw; // (0,1,0) in "
  constexpr static int dz = 9 * tw * tw; // (0,0,1) in "
  constexpr static int lb = tw - 1; // lower bound of central tile in each dimension
  constexpr static int ub = 2 * tw - 1; // upper bound "
  int state[volume];
  double tevt[volume];
  DevicePRNG prng[volume];
  double checksum[2 * dim];
  
  __device__ __forceinline__ bool edgeX0(Event const& evt) {
    if(dim == 1) {
      return evt.i == lb;
    } else if(dim == 2) {
      return evt.i == lb && evt.j >= lb && evt.j <= ub;
    } else {
      return evt.i == lb && evt.j >= lb && evt.j <= ub && evt.k >= lb && evt.k <= ub;
    }
  }
  __device__ __forceinline__ bool edgeX1(Event const& evt) {
    if(dim == 1) {
      return evt.i == ub;
    } else if(dim == 2) {
      return evt.i == ub && evt.j >= lb && evt.j <= ub;
    } else {
      return evt.i == ub && evt.j >= lb && evt.j <= ub && evt.k >= lb && evt.k <= ub;
    }
  }
  __device__ __forceinline__ bool edgeY0(Event const& evt) {
    if(dim == 2) {
      return evt.j == lb && evt.i >= lb && evt.i <= ub;
    } else {
      return evt.j == lb && evt.i >= lb && evt.i <= ub && evt.k >= lb && evt.k <= ub;
    }
  }
  
  __device__ __forceinline__ bool edgeY1(Event const& evt) {
    if(dim == 2) {
      return evt.j == ub && evt.i >= lb && evt.i <= ub;
    } else {
      return evt.j == ub && evt.i >= lb && evt.i <= ub && evt.k >= lb && evt.k <= ub;
    }
  }

  __device__ __forceinline__ bool edgeZ0(Event const& evt) {
    return evt.k == lb && evt.i >= lb && evt.i <= ub && evt.j >= lb && evt.j <= ub;
  }
  
  __device__ __forceinline__ bool edgeZ1(Event const& evt) {
    return evt.k == ub && evt.i >= lb && evt.i <= ub && evt.j >= lb && evt.j <= ub;
  }
  
  __device__ __forceinline__
  void addToChecksums(Event const& evt, double tau) {
    if(edgeX0(evt)) checksum[X0] += tau;
    if(edgeX1(evt)) checksum[X1] += tau;
    if(dim != 1) {
      if(edgeY0(evt)) checksum[Y0] += tau;
      if(edgeY1(evt)) checksum[Y1] += tau;
      if(dim != 2) {
	if(edgeZ0(evt)) checksum[Z0] += tau;
	if(edgeZ1(evt)) checksum[Z1] += tau;
      }
    }
  }

  __device__
  LocalTiles() {}

  __device__
  void integrate(double const tuntil, bool const requestRandom, bigint& nEvents) {
    checksum[X0] = 0;
    checksum[X1] = 0;
    if(dim != 1) {
      checksum[Y0] = 0;
      checksum[Y1] = 0;
      if(dim != 2) {
	checksum[Z0] = 0;
	checksum[Z1] = 0;
      }
    }
    
    Event nextEvt;
    while(peekNextEvent(tuntil, nextEvt)) {
      Config cfg = config(nextEvt.idx);
      double rand01 = requestRandom ? uniform(nextEvt.idx) : 0.;
      state[nextEvt.idx] = M::pickState(cfg, d_modelParams, rand01);
      updateNbhd(nextEvt);
      ++nEvents;
    }
  }
  
  __device__
  bool peekNextEvent(double const tuntil, Event& nextEvt) {
    nextEvt.t = tuntil;

    constexpr int start = 1;
    constexpr int end = 3 * tw - 2;
    
    if(dim == 1) {
#pragma unroll
      for(int i = start; i <= end; i++) {
	peek(i, i, 0, 0, nextEvt);
      }
    } else if(dim == 2) {
#pragma unroll
      for(int i = start; i <= end; i++) {
#pragma unroll
	for(int j = start; j <= end; j++) {
	  int idx = i + j * 3 * tw;
	  peek(idx, i, j, 0, nextEvt);
	}
      }
    } else {
#pragma unroll
      for(int i = start; i <= end; i++) {
#pragma unroll
	for(int j = start; j <= end; j++) {
#pragma unroll
	  for(int k = start; k <= end; k++) {
	    int idx = i + j * 3 * tw + k * 9 * tw * tw;
	    peek(idx, i, j, k, nextEvt);
	  }
	}
      }
    }

    return nextEvt.t < tuntil;
  }

  __device__ __forceinline__
  double uniform(int idx) {
    return prng[idx].uniform();
  }

  __device__ __forceinline__
  void updateNbhd(Event const& evt) {
    constexpr int start = 1;
    constexpr int end = 3 * tw - 2;

    update(evt.idx, evt.t);
    addToChecksums(evt, tevt[evt.idx]);

    if(evt.i > start) update(evt.idx - dx, evt.t);
    if(evt.i < end) update(evt.idx + dx, evt.t);
    if(dim != 1) {
      if(evt.j > start) update(evt.idx - dy, evt.t);
      if(evt.j < end) update(evt.idx + dy, evt.t);
      if(dim != 2) {
	if(evt.k > start) update(evt.idx - dz, evt.t);
	if(evt.k < end) update(evt.idx + dz, evt.t);
      }
    }
  }

  __device__ __forceinline__
  void update(int idx, double t0) {
    Config cfg = config(idx);
    double p = M::propensity(cfg, d_modelParams);
    tevt[idx] = t0 - log(uniform(idx)) / p;
  }

private:

  __device__ __forceinline__
  Config config(int idx) {
    Config cfg;
    cfg.dim = dim;
    cfg.s = state[idx];
    cfg.sx0 = state[idx - dx];
    cfg.sx1 = state[idx + dx];
    if(dim != 1) {
      cfg.sy0 = state[idx - dy];
      cfg.sy1 = state[idx + dy];
      if(dim != 2) {
	cfg.sz0 = state[idx - dz];
	cfg.sz1 = state[idx + dz];
      }
    }
    return cfg;
  }
  
  __device__ __forceinline__
  void peek(int idx, int i, int j, int k, Event& nextEvt) {
    if(tevt[idx] < nextEvt.t) {
      nextEvt = {idx, i, j, k, tevt[idx]};
    }
  }
};


template<typename M, int tw, int dim>
__global__
void initDeviceState(double t0, unsigned long seed, GlobalMem glob, GridParams gp, double* tmin) {
  LocalTiles<M, tw, dim> tiles;

  // pick a central tile
  TileIdx central;
  for(central.idx = blockIdx.x * blockDim.x + threadIdx.x;
      central.idx < gp.nTiles;
      central.idx += blockDim.x * gridDim.x) {

    central.i = central.idx & (gp.NX - 1);
    if(dim != 1) {
      central.j = (central.idx / gp.NX) & (gp.NY - 1);
      if(dim != 2) {
	central.k = central.idx / (gp.NX * gp.NY);
      }
    }
    
    // load central tile and its neighbours (3, 3x3 or 3x3x3)
    bridge::importTiles<tw, dim>(tiles.state, glob.stateIn, central, gp);

    // init prng and tevt
    if(dim == 1) {
      for(int di = 0; di < tw; di++) {
	int i = central.i * tw + di;
	int gidx = i;
	int lidx = gp.tileVol + di;
	tiles.prng[lidx].init(seed, gidx);
	tiles.update(lidx, t0);
	atomicMinDouble(tmin, tiles.tevt[lidx]);
      }
    } else if(dim == 2) {
      for(int di = 0; di < tw; di++) {
	for(int dj = 0; dj < tw; dj++) {
	  int i = central.i * tw + di;
	  int j = central.j * tw + dj;
	  int gidx = i + j * gp.nx;
	  int lidx = (tw + di) + 3 * tw * (tw + dj);
	  tiles.prng[lidx].init(seed, gidx);
	  tiles.update(lidx, t0);
	  atomicMinDouble(tmin, tiles.tevt[lidx]);
	}
      }
    } else {
      for(int di = 0; di < tw; di++) {
	for(int dj = 0; dj < tw; dj++) {
	  for(int dk = 0; dk < tw; dk++) {
	    int i = central.i * tw + di;
	    int j = central.j * tw + dj;
	    int k = central.k * tw + dk;
	    int gidx = i + j * gp.nx + k * gp.nx * gp.ny;
	    int lidx = (tw + di) + 3 * tw * (tw + dj) + 9 * tw * tw * (tw + dk);
	    tiles.prng[lidx].init(seed, gidx);
	    tiles.update(lidx, t0);
	    atomicMinDouble(tmin, tiles.tevt[lidx]);
	  }
	}
      }
    }

    // export the central tile and discard the ghost tiles
    bridge::exportTiles<tw, dim>(glob.tevtIn, tiles.tevt, central, gp);
    bridge::exportTiles<tw, dim>(glob.prngIn, tiles.prng, central, gp);
  }
}

template<typename M, int tw, int dim>
__global__
void integrate(GlobalMem glob, const double tuntil, const bool requestRandom, GridParams gp, bigint* nEventsTotal) {
  LocalTiles<M, tw, dim> tiles;
  bigint nEvents = 0;

  // pick a central tile
  TileIdx central;
  for(central.idx = blockIdx.x * blockDim.x + threadIdx.x;
      central.idx < gp.nTiles;
      central.idx += blockDim.x * gridDim.x) {
    central.i = central.idx & (gp.NX - 1);
    if(dim != 1) {
      central.j = (central.idx / gp.NX) & (gp.NY - 1);
      if(dim != 2) {
	central.k = central.idx / (gp.NX * gp.NY);
      }
    }
    
    // load central tile and its neighbours (3, 3x3 or 3x3x3)
    bridge::importTiles<tw, dim>(tiles.state, glob.stateIn, central, gp);
    bridge::importTiles<tw, dim>(tiles.tevt, glob.tevtIn, central, gp);
    bridge::importTiles<tw, dim>(tiles.prng, glob.prngIn, central, gp);
    
    // advance the tile neighbourhood by a single time step
    tiles.integrate(tuntil, requestRandom, nEvents);
    
    // export central tile and its checksum
    bridge::exportTiles<tw, dim>(glob.stateOut, tiles.state, central, gp);
    bridge::exportTiles<tw, dim>(glob.tevtOut, tiles.tevt, central, gp);
    bridge::exportTiles<tw, dim>(glob.prngOut, tiles.prng, central, gp);
    bridge::exportChecksum<dim>(glob.checksum, tiles.checksum, central, gp);
  }
  
  atomicAdd(nEventsTotal, nEvents);
}

/**
 * Tile index of base+(di, dj, dk)
 */   
template<int dim, int di, int dj, int dk>
__device__
int neighTileIdx(TileIdx const& base, GridParams const& gp) {
  if(dim == 1) {
    int i = di == 0 ? base.i : ((base.i + di) & (gp.NX - 1));
    return i;
  } else if(dim == 2) {
    int i = di == 0 ? base.i : ((base.i + di) & (gp.NX - 1));
    int j = dj == 0 ? base.j : ((base.j + dj) & (gp.NY - 1));
    return i + j * gp.NX;
  } else {
    int i = di == 0 ? base.i : ((base.i + di) & (gp.NX - 1));
    int j = dj == 0 ? base.j : ((base.j + dj) & (gp.NY - 1));
    int k = dk == 0 ? base.k : ((base.k + dk) & (gp.NZ - 1));
    return i + j * gp.NX + k * gp.NX * gp.NY;
  }
}

template<int tw, int dim>
__global__
void checksumValidation(double* checksum, int* fail, GridParams gp) {
  // pick a central tile
  TileIdx central;
  for(central.idx = blockIdx.x * blockDim.x + threadIdx.x;
      central.idx < gp.nTiles;
      central.idx += blockDim.x * gridDim.x) {

    central.i = central.idx & (gp.NX - 1);
    if(dim != 1) {
      central.j = (central.idx / gp.NX) & (gp.NY - 1);
      if(dim != 2) {
	central.k = central.idx / (gp.NX * gp.NY);
      }
    }
    
    int m = 2 * dim;
    int nidx;
    int my_fail = 0;
    
    // check for inconsistency with neighbouring tiles
    nidx = neighTileIdx<dim, 1, 0, 0>(central, gp);
    if(checksum[nidx * m + X0] != checksum[central.idx * m + X1]) my_fail = 1;
    if(dim != 1) {
      nidx = neighTileIdx<dim, 0, 1, 0>(central, gp);
      if(checksum[nidx * m + Y0] != checksum[central.idx * m + Y1]) my_fail = 1;
      if(dim != 2) {
	nidx = neighTileIdx<dim, 0, 0, 1>(central, gp);
	if(checksum[nidx * m + Z0] != checksum[central.idx * m + Z1]) my_fail = 1;
      }
    }
    
    if(my_fail) atomicOr(fail, 1);
  }
}

} // namespace tiles
} // namespace kmc

#endif // CUDA_TILING_H

