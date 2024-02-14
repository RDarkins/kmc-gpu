#ifndef EVENT_LOOKUP_TREE_H
#define EVENT_LOOKUP_TREE_H

#include "event_lookup.h"
#include "host_prng.h"
#include <iostream>
#include <cmath>
#include <vector>

namespace kmc {

/**
 * Use a binary tree for event lookups.
 */
class EventLookupTree : public EventLookup {

public:

  explicit EventLookupTree(int n) {
    // create tree with at least n leaves
    m_n = 1;
    while(m_n < n) m_n *= 2;
    m_tree = std::vector<double>(2 * m_n - 1, 0.);
    m_offset = m_n - 1;
    m_sum = 0.;
  }

  void update(int i, double p) override {
    i += m_offset;
    m_tree[i] = p;

    int parent, sibling;
    while (i > 0) {
      sibling = (i % 2) ? (i + 1) : (i - 1);
      parent = (i - 1) / 2;
      m_tree[parent] = m_tree[i] + m_tree[sibling];
      i = parent;
    }
    m_sum = m_tree[0];
  }

  std::tuple<int, double> event(HostPRNG& prng) const override {
    if(m_sum == 0.) return {-1, 0.};

    double r1 = prng.uniform();
    double r2 = prng.uniform();
    double rs = r1 * m_sum;

    int i, leftchild;
    i = 0;
    while(i < m_offset) {
      leftchild = 2 * i + 1;
      if(rs <= m_tree[leftchild]) {
	i = leftchild;
      } else {
	rs -= m_tree[leftchild];
	i = leftchild + 1;
      }
    }
    int id = i - m_offset;
    double dt = -log(r2) / m_sum;
    return {id, dt};
  }

private:

  int m_n;
  int m_offset;
  std::vector<double> m_tree;
  double m_sum;

};

} // namespace kmc

#endif // EVENT_LOOKUP_TREE_H
