#ifndef KMC_LATTICE_H
#define KMC_LATTICE_H

#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "common.h"

namespace kmc {

struct Idx {
  int x;
  int y;
  int z;
};


class Lattice {

public:

  const Idx gridSize;
  const int dim;
  const int volume;
  double time;

  Lattice(Idx const& gridSize, int defaultState = 0)
    : gridSize(gridSize)
    , dim((!!gridSize.x) + (!!gridSize.y) + (!!gridSize.z))
    , volume(gridSize.x * (gridSize.y ? gridSize.y : 1) * (gridSize.z ? gridSize.z : 1))
    , time(0.)
    , m_state(volume, defaultState) {
    assert(isPowerOf2(gridSize.x) &&
	   (gridSize.y == 0 ||
	    (isPowerOf2(gridSize.y) &&
	     (gridSize.z == 0 || isPowerOf2(gridSize.z)))));
    auto log2Lambda = [](int x){int l = 0; while(x >>= 1) ++l; return l;};
    m_l2GridSize = transform(gridSize, log2Lambda);
  }

  inline int state(int idx) const {
    return m_state[idx];
  }

  inline int state(Idx const& idx) const {
    return m_state[unfold(idx)];
  }

  inline int stateSafe(Idx idx) const {
    wrap(idx);
    return m_state[unfold(idx)];
  }

  inline void setState(int idx, int newState) {
    m_state[idx] = newState;
  }

  inline void setState(Idx const& idx, int newState) {
    m_state[unfold(idx)] = newState;
  }

  inline void wrap(Idx& idx) const {
    idx.x &= gridSize.x - 1;
    idx.y &= gridSize.y - 1;
    idx.z &= gridSize.z - 1;
  }

  inline Idx fold(int idx) const {
    return {idx & (gridSize.x - 1),
	    (idx >> m_l2GridSize.x) & (gridSize.y - 1),
	    idx >> (m_l2GridSize.x + m_l2GridSize.y)};
  }

  inline int unfold(Idx const& idx) const {
    return idx.x + idx.y * gridSize.x + idx.z * gridSize.x * gridSize.y;
  }

  std::vector<int> toVector() const {
    return m_state;
  }

  void reset(std::vector<int> const& v) {
    m_state = v;
  }

  void reset(std::vector<int>&& v) {
    m_state = std::move(v);
  }
  
  inline Site site(int idx) const {
    return site(fold(idx));
  }
  
  Site site(Idx const& idx) const {
    Site site;
    site.s = state(idx);
    site.sx0 = stateSafe({idx.x - 1, idx.y, idx.z});
    site.sx1 = stateSafe({idx.x + 1, idx.y, idx.z});
    if(dim != 1) {
      site.sy0 = stateSafe({idx.x, idx.y - 1, idx.z});
      site.sy1 = stateSafe({idx.x, idx.y + 1, idx.z});
      if(dim != 2) {
	site.sz0 = stateSafe({idx.x, idx.y, idx.z - 1});
	site.sz1 = stateSafe({idx.x, idx.y, idx.z + 1});
      }
    }
    return site;
  }

  friend std::ostream& operator<<(std::ostream& os, Lattice const& lattice) {
    os << lattice.gridSize.x << " " << lattice.gridSize.y << " " << lattice.gridSize.z << std::endl;
    if(lattice.dim == 1) {
      for(int i = 0; i < lattice.gridSize.x; i++) {
	os << lattice.state({i}) << std::endl;
      }
    } else if(lattice.dim == 2) {
      for(int i = 0; i < lattice.gridSize.x; i++) {
	for(int j = 0; j < lattice.gridSize.y; j++) {
	  os << lattice.state({i, j}) << std::endl;
	}
      }
    } else {
      for(int i = 0; i < lattice.gridSize.x; i++) {
	for(int j = 0; j < lattice.gridSize.y; j++) {
	  for(int k = 0; k < lattice.gridSize.z; k++) {
	    os << lattice.state({i, j, k}) << std::endl;
	  }
	}
      }
    }
    return os;
  }
  
private:

  std::vector<int> m_state;
  Idx m_l2GridSize;

  Idx transform(Idx const& idx, int op(int)) const {
    return {op(idx.x),
	    dim > 1 ? op(idx.y) : 0,
	    dim > 2 ? op(idx.z) : 0};
  }

};

} // namespace kmc

#endif // LATTICE_H
