/**
 * This file contains the GPU implementation of the tiling parts
 * of the KMC algorithm. Specific functionality included:
 *  - initialise the lattice state
 *  - integrate by one time step
 *  - build and validate the checksums
 */

#ifndef CUDA_TILING_H
#define CUDA_TILING_H

#include "bridge.h"
#include "common.h"
#include "device_prng.h"
#include "../common.h"
#include "../lattice.h"
#include <cfloat>
#include <cuda_runtime_api.h>
#include <vector>

namespace kmc {
namespace tiles {

enum {X0=0, // (-1,0, 0)
      X1,   // (1, 0, 0)
      Y0,   // (0,-1, 0)
      Y1,   // (0, 1, 0)
      Z0,   // (0, 0,-1)
      Z1    // (0, 0, 1)
};


struct Event {
  int idx;
  int i;
  int j;
  int k;
  double t;
};


__device__
inline double atomicMinDouble(double* address, double val) {
  unsigned long long int* addressAsUll = (unsigned long long int*) address;
  unsigned long long int old = *addressAsUll, assumed;
  do {
    assumed = old;
    old = atomicCAS(addressAsUll, assumed,
		    __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
  } while(assumed != old);
  return __longlong_as_double(old);
}


/**
 * The site of 3, 3x3 or 3x3x3 tiles to be stored in SM registers
 */
template<typename M, int tw, int dim>
class LocalTiles {

  constexpr static int volume= dim == 1 ? (3 * tw) : (dim == 2 ? (9 * tw * tw) : (27 * tw * tw * tw));
  constexpr static int dx = 1; // (1,0,0) in unfolded index space
  constexpr static int dy = 3 * tw; // (0,1,0) in unfolded index space
  constexpr static int dz = 9 * tw * tw; // (0,0,1) in unfolded index space
  constexpr static int lb = tw - 1; // lower bound of central tile in each dimension
  constexpr static int ub = 2 * tw - 1; // upper bound of central tile in each dimension
  
public:

  int state[volume];
  double tevt[volume];
  DevicePRNG prng[volume];
  double checksum[2 * dim]; // X0, X1, [Y0, Y1, [Z0, Z1]]
  
  __device__
  LocalTiles() {}
  
  __device__
  void integrate(double tuntil, bool requestRandom, bigint& nEvents) {
    // reset checksums
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
    
    // perform all events in the local tiles until time tuntil is reached
    Event nextEvt;
    while(peekNextEvent(tuntil, nextEvt)) {
      Site site = getSite(nextEvt.idx);
      double rand01 = requestRandom ? uniform(nextEvt.idx) : 0.;
      state[nextEvt.idx] = M::pickNextState(site, d_modelParams, rand01);
      updateNbhd(nextEvt);
      ++nEvents;
    }
  }

  __device__ __forceinline__
  void updateTimeOfNextEvent(int idx, double currentTime) {
    Site site = getSite(idx);
    double p = M::propensity(site, d_modelParams);
    tevt[idx] = currentTime - log(uniform(idx)) / p;
  }

private:
  
  __device__
  bool peekNextEvent(double tuntil, Event& nextEvt) {
    nextEvt.t = tuntil;

    constexpr int start = 1;
    constexpr int end = 3 * tw - 2;
    
    // Search all cells (minus the boundary) for the next event.
    // A linear search avoids thread branching.
    
    if(dim == 1) {
#pragma unroll
      for(int i = start; i <= end; i++) {
	updateNextEventIfEarlier(i, i, 0, 0, nextEvt);
      }
    } else if(dim == 2) {
#pragma unroll
      for(int i = start; i <= end; i++) {
#pragma unroll
	for(int j = start; j <= end; j++) {
	  int idx = i + j * 3 * tw;
	  updateNextEventIfEarlier(idx, i, j, 0, nextEvt);
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
	    updateNextEventIfEarlier(idx, i, j, k, nextEvt);
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

    // update the cell associated with event evt
    updateTimeOfNextEvent(evt.idx, evt.t);
    addToChecksums(evt, tevt[evt.idx]);

    // update the neighbouring cells (excluding the fixed boundary)
    if(evt.i > start) updateTimeOfNextEvent(evt.idx - dx, evt.t);
    if(evt.i < end) updateTimeOfNextEvent(evt.idx + dx, evt.t);
    if(dim != 1) {
      if(evt.j > start) updateTimeOfNextEvent(evt.idx - dy, evt.t);
      if(evt.j < end) updateTimeOfNextEvent(evt.idx + dy, evt.t);
      if(dim != 2) {
	if(evt.k > start) updateTimeOfNextEvent(evt.idx - dz, evt.t);
	if(evt.k < end) updateTimeOfNextEvent(evt.idx + dz, evt.t);
      }
    }
  }

  __device__ __forceinline__
  bool edgeX0(Event const& evt) {
    if(dim == 1) {
      return evt.i == lb;
    } else if(dim == 2) {
      return evt.i == lb && evt.j >= lb && evt.j <= ub;
    } else {
      return evt.i == lb && evt.j >= lb && evt.j <= ub && evt.k >= lb && evt.k <= ub;
    }
  }

  __device__ __forceinline__
  bool edgeX1(Event const& evt) {
    if(dim == 1) {
      return evt.i == ub;
    } else if(dim == 2) {
      return evt.i == ub && evt.j >= lb && evt.j <= ub;
    } else {
      return evt.i == ub && evt.j >= lb && evt.j <= ub && evt.k >= lb && evt.k <= ub;
    }
  }

  __device__ __forceinline__
  bool edgeY0(Event const& evt) {
    if(dim == 2) {
      return evt.j == lb && evt.i >= lb && evt.i <= ub;
    } else {
      return evt.j == lb && evt.i >= lb && evt.i <= ub && evt.k >= lb && evt.k <= ub;
    }
  }
  
  __device__ __forceinline__
  bool edgeY1(Event const& evt) {
    if(dim == 2) {
      return evt.j == ub && evt.i >= lb && evt.i <= ub;
    } else {
      return evt.j == ub && evt.i >= lb && evt.i <= ub && evt.k >= lb && evt.k <= ub;
    }
  }

  __device__ __forceinline__
  bool edgeZ0(Event const& evt) {
    return evt.k == lb && evt.i >= lb && evt.i <= ub && evt.j >= lb && evt.j <= ub;
  }
  
  __device__ __forceinline__
  bool edgeZ1(Event const& evt) {
    return evt.k == ub && evt.i >= lb && evt.i <= ub && evt.j >= lb && evt.j <= ub;
  }
  
  __device__ __forceinline__
  void addToChecksums(Event const& evt, double tau) {
    if(edgeX0(evt)) checksum[X0] += tau;
    else if(edgeX1(evt)) checksum[X1] += tau;
    if(dim != 1) {
      if(edgeY0(evt)) checksum[Y0] += tau;
      else if(edgeY1(evt)) checksum[Y1] += tau;
      if(dim != 2) {
	if(edgeZ0(evt)) checksum[Z0] += tau;
	else if(edgeZ1(evt)) checksum[Z1] += tau;
      }
    }
  }

  __device__ __forceinline__
  Site getSite(int idx) {
    Site site;
    site.s = state[idx];
    site.sx0 = state[idx - dx];
    site.sx1 = state[idx + dx];
    if(dim != 1) {
      site.sy0 = state[idx - dy];
      site.sy1 = state[idx + dy];
      if(dim != 2) {
	site.sz0 = state[idx - dz];
	site.sz1 = state[idx + dz];
      }
    }
    return site;
  }
  
  __device__ __forceinline__
  void updateNextEventIfEarlier(int idx, int i, int j, int k, Event& nextEvt) {
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
    bridge::importTiles<tw, dim>(tiles.state, glob.stateFront, central, gp);

    // init prng and tevt
    if(dim == 1) {
      for(int di = 0; di < tw; di++) {
	int i = central.i * tw + di;
	int gidx = i;
	int lidx = gp.tileVol + di;
	tiles.prng[lidx].init(seed, gidx);
	tiles.updateTimeOfNextEvent(lidx, t0);
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
	  tiles.updateTimeOfNextEvent(lidx, t0);
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
	    tiles.updateTimeOfNextEvent(lidx, t0);
	    atomicMinDouble(tmin, tiles.tevt[lidx]);
	  }
	}
      }
    }

    // export the central tile and discard the ghost tiles
    bridge::exportTiles<tw, dim>(glob.tevtFront, tiles.tevt, central, gp);
    bridge::exportTiles<tw, dim>(glob.prngFront, tiles.prng, central, gp);
  }
}

template<typename M, int tw, int dim>
__global__
void integrate(GlobalMem glob, double tuntil, bool requestRandom, GridParams gp, bigint* nEventsTotal) {
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
    
    // load central tile and its neighbours (3, 3x3 or 3x3x3) from front buffer
    bridge::importTiles<tw, dim>(tiles.state, glob.stateFront, central, gp);
    bridge::importTiles<tw, dim>(tiles.tevt, glob.tevtFront, central, gp);
    bridge::importTiles<tw, dim>(tiles.prng, glob.prngFront, central, gp);
    
    // advance the tile neighbourhood by a single time step
    tiles.integrate(tuntil, requestRandom, nEvents);
    
    // export central tile and its checksum to back buffer
    bridge::exportTiles<tw, dim>(glob.stateBack, tiles.state, central, gp);
    bridge::exportTiles<tw, dim>(glob.tevtBack, tiles.tevt, central, gp);
    bridge::exportTiles<tw, dim>(glob.prngBack, tiles.prng, central, gp);
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
    
    int twoDim = 2 * dim;
    int nidx;
    int myFail = 0;
    
    // checksum consistency between central and neighbouring tiles
    nidx = neighTileIdx<dim, 1, 0, 0>(central, gp);
    if(checksum[nidx * twoDim + X0] != checksum[central.idx * twoDim + X1]) myFail = 1;
    if(dim != 1) {
      nidx = neighTileIdx<dim, 0, 1, 0>(central, gp);
      if(checksum[nidx * twoDim + Y0] != checksum[central.idx * twoDim + Y1]) myFail = 1;
      if(dim != 2) {
	nidx = neighTileIdx<dim, 0, 0, 1>(central, gp);
	if(checksum[nidx * twoDim + Z0] != checksum[central.idx * twoDim + Z1]) myFail = 1;
      }
    }
    
    if(myFail) atomicOr(fail, 1);
  }
}

} // namespace tiles
} // namespace kmc

#endif // CUDA_TILING_H

