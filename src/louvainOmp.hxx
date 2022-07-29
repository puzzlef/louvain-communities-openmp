#pragma once
#include <utility>
#include <vector>
#include <iostream>
#include "_main.hxx"
#include "modularity.hxx"
#include "accumulate.hxx"
#include "louvain.hxx"

using std::vector;
using std::make_pair;
using std::move;
using std::cout;




/**
 * Find the total edge weight of each vertex.
 * @param vtot total edge weight of each vertex (updated)
 * @param x original graph
 */
template <class G, class V>
void louvainVertexWeights(vector<V>& vtot, const G& x) {
  // vtot should be filled with 0
  x.forEachVertexKey([&](auto u) {
    x.forEachEdge(u, [&](auto v, auto w) {
      vtot[u] += w;
    });
  });
}


/**
 * Find the total edge weight of each community.
 * @param ctot total edge weight of each community (updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class V>
void louvainCommunityWeights(vector<V>& ctot, const G& x, const vector<K>& vcom, const vector<V>& vtot) {
  // ctot should be filled with 0
  x.forEachVertexKey([&](auto u) {
    K c = vcom[u];
    ctot[c] += vtot[u];
  });
}


/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to
 * @param ctot total edge weight of each community
 * @param x original graph
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class V>
void louvainInitialize(vector<K>& vcom, vector<V>& ctot, const G& x, const vector<V>& vtot) {
  // vcom, ctot should be filled with 0
  x.forEachVertexKey([&](auto u) {
    vcom[u] = u;
    ctot[u] = vtot[u];
  });
}


/**
 * Scan communities connected to a vertex.
 * @param vcs communities vertex u is linked to
 * @param vcout total edge weight from vertex u to community C
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 */
template <class G, class K, class V>
void louvainScanCommunities(vector<K>& vcs, vector<V>& vcout, const G& x, K u, const vector<K>& vcom) {
  x.forEachEdge(u, [&](auto v, auto w) {
    if (u==v) return;
    K c = vcom[v];
    if (!vcout[c]) vcs.push_back(c);
    vcout[c] += w;
  });
}


/**
 * Clear communities scan data.
 * @param vcs total edge weight from vertex u to community C
 * @param vcout communities vertex u is linked to
 */
template <class K, class V>
void louvainClearScan(vector<K>& vcs, vector<V>& vcout) {
  for (K c : vcs)
    vcout[c] = V();
  vcs.clear();
}


/**
 * Choose connected community with best delta modularity.
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param ctot total edge weight of each community
 * @param vcs communities vertex u is linked to
 * @param vcout total edge weight from vertex u to community C
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @returns [best community, delta modularity]
 */
template <class G, class K, class V>
auto louvainChooseCommunity(const G& x, K u, const vector<K>& vcom, const vector<V>& vtot, const vector<V>& ctot, const vector<K>& vcs, const vector<V>& vcout, V M, V R) {
  K cmax = K(), d = vcom[u];
  V emax = V();
  for (K c : vcs) {
    if (c==d) continue;
    V e = deltaModularity(vcout[c], vcout[d], vtot[u], ctot[c], ctot[d], M, R);
    if (e>emax) { emax = e; cmax = c; }
  }
  return make_pair(cmax, emax);
}


/**
 * Move vertex to another community C.
 * @param vcom community each vertex belongs to
 * @param ctot total edge weight of each community
 * @param x original graph
 * @param u given vertex
 * @param c community to move to
 * @param vdom community each vertex belonged to
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class V>
void louvainChangeCommunity(vector<K>& vcom, vector<V>& ctot, const G& x, K u, K c, const vector<K>& vdom, const vector<V>& vtot) {
  K d = vdom[u];
  ctot[d] -= vtot[u];
  ctot[c] += vtot[u];
  vcom[u] = c;
}


/**
 * Louvain algorithm's local moving phase.
 * @param vdom community each vertex belonged to (zeros)
 * @param vcom community each vertex belongs to (initial)
 * @param dtot previous total edge weight of each community (zeros)
 * @param ctot total edge weight of each community (precalculated)
 * @param vcs communities vertex u is linked to (temporary buffer)
 * @param vcout total edge weight from vertex u to community C (temporary buffer)
 * @param x original graph
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param E tolerance (0)
 * @param L max iterations (500)
 * @returns iterations
 */
template <bool O, class G, class K, class V>
int louvainMove(vector<K>& vdom, vector<K>& vcom, vector<V>& dtot, vector<V>& ctot, vector<K>& vcs, vector<V>& vcout, const G& x, const vector<V>& vtot, V M, V R, V E, int L) {
  K S = x.span(), l = 0; V Q = V();
  for (; l<L;) {
    V el = V();
    if (!O) copyValues(vcom, vdom);
    if (!O) copyValues(ctot, dtot);
    x.forEachVertexKey([&](auto u) {
      louvainClearScan(vcs, vcout);
      louvainScanCommunities(vcs, vcout, x, u, vdom);
      auto [c, e] = louvainChooseCommunity(x, u, vdom, vtot, dtot, vcs, vcout, M, R);
      if (O) { if (c)              louvainChangeCommunity(vcom, ctot, x, u, c, vdom, vtot); }
      else   { if (c && c<vdom[u]) louvainChangeCommunity(vcom, ctot, x, u, c, vdom, vtot); }
      el += e;   // l1-norm
    }); ++l;
    if (el<=E) break;
  }
  return l;
}


/**
 * Louvain algorithm's community aggregation phase.
 * @param a output graph
 * @param x original graph
 * @param vcom community each vertex belongs to
 */
template <class G, class K>
void louvainAggregate(G& a, const G& x, const vector<K>& vcom) {
  using V = typename G::edge_value_type;
  OrderedOutDiGraph<K, NONE, V> b;
  x.forEachVertexKey([&](auto u) {
    K c = vcom[u];
    b.addVertex(c);
  });
  x.forEachVertexKey([&](auto u) {
    K c = vcom[u];
    x.forEachEdge(u, [&](auto v, auto w) {
      K d = vcom[v];
      if (!b.hasEdge(c, d)) b.addEdge(c, d, w);
      else b.setEdgeValue(c, d, w + b.edgeValue(c, d));
    });
  });
  duplicateW(a, b);
}
template <class G, class K>
auto louvainAggregate(const G& x, const vector<K>& vcom) {
  G a; louvainAggregate(a, x, vcom);
  return a;
}


/**
 * Update community membership in a tree-like fashion (to handle aggregation).
 * @param a output ommunity each vertex belongs to
 * @param vcom community each vertex belongs to (at this aggregation level)
 */
template <class K>
void louvainLookupCommunities(vector<K>& a, const vector<K>& vcom) {
  for (auto& v : a)
    v = vcom[v];
}


template <bool O, class G, class V=float>
auto louvainSeq(const G& x, LouvainOptions<V> o={}) {
  using K = typename G::key_type;
  V   R = o.resolution;
  V   E = o.tolerance;
  V   D = o.passTolerance;
  int L = o.maxIterations,      l = 0;
  int P = o.maxPasses, p = 0;
  V   M = edgeWeight(x)/2;
  size_t S = x.span();
  vector<K> vdom(S), vcom(S), vcs, a(S);
  vector<V> vtot(S), dtot(S), ctot(S), vcout(S);
  float t = measureDurationMarked([&](auto mark) {
    V Q0 = modularity(x, M, R);
    G y  = duplicate(x);
    fillValueU(vcom, K());
    fillValueU(vtot, V());
    fillValueU(ctot, V());
    mark([&]() {
      louvainVertexWeights(vtot, y);
      louvainInitialize(vcom, ctot, y, vtot);
      copyValues(vcom, a);
      for (l=0, p=0; p<P;) {
        if (O) l += louvainMove<O>(vcom, vcom, ctot, ctot, vcs, vcout, y, vtot, M, R, E, L);
        else   l += louvainMove<O>(vdom, vcom, dtot, ctot, vcs, vcout, y, vtot, M, R, E, L);
        y = louvainAggregate(y, vcom); ++p;
        louvainLookupCommunities(a, vcom);
        V Q = modularity(y, M, R);
        V M = edgeWeight(y)/2;
        if (Q-Q0<=D) break;
        fillValueU(vcom, K());
        fillValueU(vtot, V());
        fillValueU(ctot, V());
        louvainVertexWeights(vtot, y);
        louvainInitialize(vcom, ctot, y, vtot);
        Q0 = Q;
      }
    });
  }, o.repeat);
  return LouvainResult<K>(a, l, p, t);
}
