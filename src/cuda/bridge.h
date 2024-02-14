/**
 * This header provides a 'bridge' between different memory types.
 * It contains functions for transferring lattice data between the
 * host memory, the GPU global memory, and the GPU registers.
 */

#ifndef CUDA_BRIDGE_H
#define CUDA_BRIDGE_H

#include "common.h"
#include "../lattice.h"

namespace kmc {
namespace tiles {
namespace bridge {

/**
 * Import the 3, 3x3 or 3x3x3 tiles from glob (GPU glob mem) into loc (GPU registers).
 */
template<int tw, int dim, typename T>
__device__
void importTiles(T* __restrict__ loc,
		 T const * __restrict__ glob,
		 TileIdx const& central,
		 GridParams const& gp) {

  constexpr int dy = 3 * tw;
  constexpr int dz = 9 * tw * tw;

  if(dim == 1) {
    for(int di = -1; di <= 1; di++) {
      int ti = (central.i + di) & (gp.NX - 1);
      int tidx = ti;
      int li = (1 + di) * tw;
      int gi = tidx;
      for(int i = li; i < li + tw; i++) {
	loc[i] = glob[gi];
	gi += gp.nTiles;
      }
    }
  } else if(dim == 2) {
    for(int di = -1; di <= 1; di++) {
      for(int dj = -1; dj <= 1; dj++) {
	int ti = (central.i + di) & (gp.NX - 1);
	int tj = (central.j + dj) & (gp.NY - 1);
	int tidx = ti + tj * gp.NX;
	int li = (1 + di) * tw;
	int lj = (1 + dj) * tw;
	int gi = tidx;
	for(int j = lj; j < lj + tw; j++) {
	  for(int i = li; i < li + tw; i++) {
	    loc[i + j * dy] = glob[gi];
	    gi += gp.nTiles;
	  }
	}
      }
    }
  } else {
    for(int di = -1; di <= 1; di++) {
      for(int dj = -1; dj <= 1; dj++) {
	for(int dk = -1; dk <= 1; dk++) {
	  int ti = (central.i + di) & (gp.NX - 1);
	  int tj = (central.j + dj) & (gp.NY - 1);
	  int tk = (central.k + dk) & (gp.NZ - 1);
	  int tidx = ti + tj * gp.NX + tk * gp.NX * gp.NY;
	  int li = (1 + di) * tw;
	  int lj = (1 + dj) * tw;
	  int lk = (1 + dk) * tw;
	  int gi = tidx;
	  for(int k = lk; k < lk + tw; k++) {
	    for(int j = lj; j < lj + tw; j++) {
	      for(int i = li; i < li + tw; i++) {
		loc[i + j * dy + k * dz] = glob[gi];
		gi += gp.nTiles;
	      }
	    }
	  }
	}
      }
    }
  }
}

/**
 * Export the 3, 3x3 or 3x3x3 tiles from loc (registers) into glob (GPU glob mem).
 */
template<int tw, int dim, typename T>
__device__
void exportTiles(T* __restrict__ glob,
		 T const * __restrict__ loc,
		 TileIdx const& central,
		 GridParams const& gp) {
  constexpr int dy = 3 * tw;
  constexpr int dz = 9 * tw * tw;

  if(dim == 1) {
    int li = tw;
    int gi = central.idx;
    for(int i = li; i < li + tw; i++) {
      glob[gi] = loc[i];
      gi += gp.nTiles;
    }
  } else if(dim == 2) {
    int li = tw;
    int lj = tw;
    int gi = central.idx;
    for(int j = lj; j < lj + tw; j++) {
      for(int i = li; i < li + tw; i++) {
	glob[gi] = loc[i + j * dy];
	gi += gp.nTiles;
      }
    }
  } else {
    int li = tw;
    int lj = tw;
    int lk = tw;
    int gi = central.idx;
    for(int k = lk; k < lk + tw; k++) {
      for(int j = lj; j < lj + tw; j++) {
	for(int i = li; i < li + tw; i++) {
	  glob[gi] = loc[i + j * dy + k * dz];
	  gi += gp.nTiles;
	}
      }
    }
  }
}

/**
 * Export tile checksums from loc (GPU registers) to glob (GPU glob mem).
 */
template<int dim>
__device__
void exportChecksum(double* __restrict__ glob,
		    double const* __restrict__ loc,
		    TileIdx const& central,
		    GridParams const& gp) {
  int m = 2 * dim;
#pragma unroll
  for(int i = 0; i < m; i++) {
    glob[central.idx * m + i] = loc[i];
  }
}

/**
 * Transfer lattice (host) to d_state (GPU glob mem).
 * Organise data to allow the threads coallesced access.
 */
template<int tw>
void loadLattice(int* d_state, Lattice const& lattice, GridParams const& gp) {
  std::vector<int> h_state(lattice.volume);
  if(gp.dim == 1) {
    for(int i = 0; i < lattice.gridSize.x; i++) {
      int ti = i / tw;
      int tidx = ti;
      int ci = i & (tw - 1);
      int cidx = ci;
      h_state[tidx + cidx * gp.nTiles] = lattice.state(i);
    }
  } else if(gp.dim == 2) {
    for(int i = 0; i < lattice.gridSize.x; i++) {
      for(int j = 0; j < lattice.gridSize.y; j++) {
	int ti = i / tw;
	int tj = j / tw;
	int tidx = ti + tj * gp.NX;
	int ci = i & (tw - 1);
	int cj = j & (tw - 1);
	int cidx = ci + cj * tw;
	h_state[tidx + cidx * gp.nTiles] = lattice.state({i,j});
      }
    }
  } else {
    for(int i = 0; i < lattice.gridSize.x; i++) {
      for(int j = 0; j < lattice.gridSize.y; j++) {
	for(int k = 0; k < lattice.gridSize.z; k++) {
	  int ti = i / tw;
	  int tj = j / tw;
	  int tk = k / tw;
	  int tidx = ti + tj * gp.NX + tk * gp.NX * gp.NY;
	  int ci = i & (tw - 1);
	  int cj = j & (tw - 1);
	  int ck = k & (tw - 1);
	  int cidx = ci + cj * tw + ck * tw * tw;
	  h_state[tidx + cidx * gp.nTiles] = lattice.state({i,j,k});
	}
      }
    }
  }
  gpuAssert(cudaMemcpy(d_state, h_state.data(), h_state.size() * sizeof(int), cudaMemcpyHostToDevice));
}

/**
 * Transfer from d_state (GPU glob mem) back to lattice (host).
 */
template<int tw>
void unloadLattice(Lattice& lattice, int* d_state, GridParams const& gp) {
  std::vector<int> h_state(lattice.volume);
  gpuAssert(cudaMemcpy(h_state.data(), d_state, h_state.size() * sizeof(int), cudaMemcpyDeviceToHost));
  
  if(gp.dim == 1) {
    for(int i = 0; i < lattice.gridSize.x; i++) {
      int ti = i / tw;
      int tidx = ti;
      int ci = i & (tw - 1);
      int cidx = ci;
      lattice.setState(i, h_state[tidx + cidx * gp.nTiles]);
    }
  } else if(gp.dim == 2) {
    for(int i = 0; i < lattice.gridSize.x; i++) {
      for(int j = 0; j < lattice.gridSize.y; j++) {
	int ti = i / tw;
	int tj = j / tw;
	int tidx = ti + tj * gp.NX;
	int ci = i & (tw - 1);
	int cj = j & (tw - 1);
	int cidx = ci + cj * tw;
	lattice.setState({i, j}, h_state[tidx + cidx * gp.nTiles]);
      }
    }
  } else {
    for(int i = 0; i < lattice.gridSize.x; i++) {
      for(int j = 0; j < lattice.gridSize.y; j++) {
	for(int k = 0; k < lattice.gridSize.z; k++) {
	  int ti = i / tw;
	  int tj = j / tw;
	  int tk = k / tw;
	  int tidx = ti + tj * gp.NX + tk * gp.NX * gp.NY;
	  int ci = i & (tw - 1);
	  int cj = j & (tw - 1);
	  int ck = k & (tw - 1);
	  int cidx = ci + cj * tw + ck * tw * tw;
	  lattice.setState({i, j, k}, h_state[tidx + cidx * gp.nTiles]);
	}
      }
    }
  }
}

} // namespace bridge
} // namespace tiles
} // namespace kmc

#endif // CUDA_BRIDGE_H
