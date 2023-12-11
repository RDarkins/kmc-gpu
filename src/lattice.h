#ifndef KMC_LATTICE_H
#define KMC_LATTICE_H

#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "kmc_common.h"

namespace kmc {

class Lattice {

public:

  const Idx grid;
  const int dim;
  const int volume;
  double time;

  Lattice(Idx const& grid, int defaultState = 0)
    : grid(grid)
    , dim((!!grid.x) + (!!grid.y) + (!!grid.z))
    , volume(vol(grid))
    , time(0.)
    , m_state(volume, defaultState) {
    assert(isPowerOf2(grid.x) &&
	   (grid.y == 0 || isPowerOf2(grid.y)) &&
	   (grid.z == 0 || (isPowerOf2(grid.z) && grid.y > 0)));
    m_l2Grid = transform(grid, [](int x){int l = 0; while(x >>= 1) ++l; return l;}); // log2
  }

  Config config(int idx) const {
    return config(fold(idx));
  }

  Config config(Idx const& idx) const {
    Config cfg;
    cfg.dim = dim;

    cfg.s = state(idx);
    cfg.sx0 = stateSafe({idx.x - 1, idx.y, idx.z});
    cfg.sx1 = stateSafe({idx.x + 1, idx.y, idx.z});
    if(dim != 1) {
      cfg.sy0 = stateSafe({idx.x, idx.y - 1, idx.z});
      cfg.sy1 = stateSafe({idx.x, idx.y + 1, idx.z});
      if(dim != 2) {
	cfg.sz0 = stateSafe({idx.x, idx.y, idx.z - 1});
	cfg.sz1 = stateSafe({idx.x, idx.y, idx.z + 1});
      }
    }

    return cfg;
  }

  int state(int idx) const {
    return m_state[idx];
  }

  int state(Idx const& idx) const {
    return m_state[unfold(idx)];
  }

  int stateSafe(Idx idx) const {
    wrap(idx);
    return m_state[unfold(idx)];
  }

  void setState(int idx, int newState) {
    m_state[idx] = newState;
  }

  void setState(Idx const& idx, int newState) {
    m_state[unfold(idx)] = newState;
  }

  void wrap(Idx& idx) const {
    idx.x &= grid.x - 1;
    idx.y &= grid.y - 1;
    idx.z &= grid.z - 1;
  }

  Idx fold(int idx) const {
    return {idx & (grid.x - 1), (idx >> m_l2Grid.x) & (grid.y - 1), idx >> (m_l2Grid.x + m_l2Grid.y)};
  }

  int unfold(Idx const& idx) const {
    return idx.x + idx.y * grid.x + idx.z * grid.x * grid.y;
  }

  std::vector<int> toVector() const {
    return m_state;
  }

  void reset(std::vector<int> v) {
    m_state.swap(v);
  }

private:

  std::vector<int> m_state;
  Idx m_l2Grid;

  static Idx transform(Idx const& idx, int op(int)) {
    return {op(idx.x), op(idx.y), op(idx.z)};
  }

  static int vol(Idx const& grid) {
    return grid.x * (grid.y ? grid.y : 1) * (grid.z ? grid.z : 1);
  }

};

} // namespace kmc

#endif // LATTICE_H
