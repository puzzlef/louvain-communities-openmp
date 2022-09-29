#pragma once
#include <utility>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "_main.hxx"
#include "properties.hxx"
#include "duplicate.hxx"
#include "modularity.hxx"
#include "louvain.hxx"

using std::tuple;
using std::vector;
using std::min;




// LOUVAIN-CHANGE-COMMUNITY
// ------------------------

/**
 * Move vertex to another community C.
 * @param vcom community each vertex belongs to (updated)
 * @param ctot total edge weight of each community (updated)
 * @param x original graph
 * @param u given vertex
 * @param c community to move to
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class V>
void louvainChangeCommunityOmp(vector<K>& vcom, vector<V>& ctot, const G& x, K u, K c, const vector<V>& vtot) {
  K d = vcom[u];
  #pragma omp atomic
  ctot[d] -= vtot[u];
  #pragma omp atomic
  ctot[c] += vtot[u];
  vcom[u] = c;
}




// LOUVAIN-MOVE
// ------------

/**
 * Louvain algorithm's local moving phase.
 * @param vcom community each vertex belongs to (initial, updated)
 * @param ctot total edge weight of each community (precalculated, updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param E tolerance
 * @param L max iterations
 * @param fa is a vertex affected?
 * @param fp process vertices whose communities have changed
 * @returns iterations performed
 */
template <class G, class K, class V, class FA, class FP>
int louvainMoveOmp(vector<K>& vcom, vector<V>& ctot, vector2d<K>& vcs, vector2d<V>& vcout, const G& x, const vector<V>& vtot, V M, V R, V E, int L, FA fa, FP fp) {
  K S = x.span();
  int l = 0; V Q = V();
  for (; l<L;) {
    V el = V();
    #pragma omp parallel for schedule(auto) reduction(+:el)
    for (K u=K(); u<S; ++u) {
      int t = omp_get_thread_num();
      if (!x.hasVertex(u)) continue;
      if (!fa(u)) continue;
      louvainClearScan(vcs[t], vcout[t]);
      louvainScanCommunities(vcs[t], vcout[t], x, u, vcom);
      auto [c, e] = louvainChooseCommunity(x, u, vcom, vtot, ctot, vcs[t], vcout[t], M, R);
      if (c)      { louvainChangeCommunityOmp(vcom, ctot, x, u, c, vtot); fp(u); }
      el += e;  // l1-norm
    } ++l;
    if (el<=E) break;
  }
  return l;
}
template <class G, class K, class V, class FA>
inline int louvainMoveOmp(vector<K>& vcom, vector<V>& ctot, vector2d<K>& vcs, vector2d<V>& vcout, const G& x, const vector<V>& vtot, V M, V R, V E, int L, FA fa) {
  auto fp = [](auto u) {};
  return louvainMoveOmp(vcom, ctot, vcs, vcout, x, vtot, M, R, E, L, fa, fp);
}
template <class G, class K, class V>
inline int louvainMoveOmp(vector<K>& vcom, vector<V>& ctot, vector2d<K>& vcs, vector2d<V>& vcout, const G& x, const vector<V>& vtot, V M, V R, V E, int L) {
  auto fa = [](auto u) { return true; };
  return louvainMoveOmp(vcom, ctot, vcs, vcout, x, vtot, M, R, E, L, fa);
}




// LOUVAIN-AGGREGATE
// -----------------

/**
 * Louvain algorithm's community aggregation phase.
 * @param a output graph
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 */
template <class G, class K, class V>
void louvainAggregateOmp(G& a, vector2d<K>& vcs, vector2d<V>& vcout, const G& x, const vector<K>& vcom) {
  K S = x.span();
  auto comv = louvainCommunityVertices(x, vcom);
  for (K c=0; c<comv.size(); ++c) {
    if (comv[c].empty()) continue;
    a.addVertex(c);
  }
  #pragma omp parallel for schedule(auto)
  for (K c=0; c<comv.size(); ++c) {
    int t = omp_get_thread_num();
    if (comv[c].empty()) continue;
    louvainClearScan(vcs[t], vcout[t]);
    for (K u : comv[c])
      louvainScanCommunities<true>(vcs[t], vcout[t], x, u, vcom);
    for (auto d : vcs[t])
      a.addEdge(c, d, vcout[t][d]);
  }
  a.correct();
}
template <class G, class K, class V>
inline auto louvainAggregateOmp(vector2d<K>& vcs, vector2d<V>& vcout, const G& x, const vector<K>& vcom) {
  G a; louvainAggregateOmp(a, vcs, vcout, x, vcom);
  return a;
}




// LOUVAIN-OMP
// -----------

template <class G, class K, class V, class FA, class FP>
auto louvainOmp(const G& x, const vector<K>* q, const LouvainOptions<V>& o, FA fa, FP fp) {
  V   R = o.resolution;
  V   D = o.passTolerance;
  int L = o.maxIterations, l = 0;
  int P = o.maxPasses, p = 0;
  K   S = x.span();
  V   M = edgeWeight(x)/2;
  int T = omp_get_max_threads();
  vector<K> vcom(S), a(S);
  vector<V> vtot(S), ctot(S);
  vector2d<K> vcs(T);
  vector2d<V> vcout(T, vector<V>(S));
  float t = measureDurationMarked([&](auto mark) {
    V E  = o.tolerance;
    V Q0 = modularity(x, M, R);
    G y  = duplicate(x);
    fillValueU(vcom, K());
    fillValueU(vtot, V());
    fillValueU(ctot, V());
    mark([&]() {
      louvainVertexWeights(vtot, y);
      louvainInitialize(vcom, ctot, y, vtot);
      if (q) copyValues(*q, vcom, 0, min((*q).size(), vcom.size()));
      if (q) louvainCommunityWeights(ctot, y, vcom, vtot);
      copyValues(vcom, a);
      for (l=0, p=0; p<P;) {
        if (p==0) l += louvainMoveOmp(vcom, ctot, vcs, vcout, y, vtot, M, R, E, L, fa, fp);
        else      l += louvainMoveOmp(vcom, ctot, vcs, vcout, y, vtot, M, R, E, L);
        y  = louvainAggregateOmp(vcs, vcout, y, vcom); ++p;
        louvainLookupCommunities(a, vcom);
        V Q = modularity(y, M, R);
        if (Q-Q0<=D) break;
        fillValueU(vcom, K());
        fillValueU(vtot, V());
        fillValueU(ctot, V());
        louvainVertexWeights(vtot, y);
        louvainInitialize(vcom, ctot, y, vtot);
        E /= o.tolerenceDeclineFactor;
        Q0 = Q;
      }
    });
  }, o.repeat);
  return LouvainResult<K>(a, l, p, t);
}
template <class G, class K, class V, class FA>
inline auto louvainOmp(const G& x, const vector<K>* q, const LouvainOptions<V>& o, FA fa) {
  auto fp = [](auto u) {};
  return louvainOmp(x, q, o, fa, fp);
}
template <class G, class K, class V>
inline auto louvainOmp(const G& x, const vector<K>* q, const LouvainOptions<V>& o) {
  auto fa = [](auto u) { return true; };
  return louvainOmp(x, q, o, fa);
}




// LOUVAIN-OMP-STATIC
// ------------------

template <class G, class K, class V=float>
inline auto louvainOmpStatic(const G& x, const vector<K>* q=nullptr, const LouvainOptions<V>& o={}) {
  return louvainOmp(x, q, o);
}




// LOUVAIN-OMP-DYNAMIC-DELTA-SCREENING
// -----------------------------------

template <class G, class K, class V>
inline auto louvainOmpDynamicDeltaScreening(const G& x, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>* q, const LouvainOptions<V>& o={}) {
  K S = x.span();
  V R = o.resolution;
  V M = edgeWeight(x)/2;
  const vector<K>& vcom = *q;
  vector<V> vtot(S), ctot(S);
  louvainVertexWeights(vtot, x);
  louvainCommunityWeights(ctot, x, vcom, vtot);
  auto vaff = louvainAffectedVerticesDeltaScreening(x, deletions, insertions, vcom, vtot, ctot, M, R);
  auto fa   = [&](auto u) { return vaff[u]==true; };
  return louvainOmp(x, q, o, fa);
}




// LOUVAIN-SEQ-DYNAMIC-FRONTIER
// ----------------------------

template <class G, class K, class V>
inline auto louvainOmpDynamicFrontier(const G& x, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>* q, const LouvainOptions<V>& o={}) {
  K S = x.span();
  const vector<K>& vcom = *q;
  auto vaff = louvainAffectedVerticesFrontier(x, deletions, insertions, vcom);
  auto fa = [&](auto u) { return vaff[u]==true; };
  auto fp = [&](auto u) { x.forEachEdgeKey(u, [&](auto v) { vaff[v] = true; }); };
  return louvainOmp(x, q, o, fa, fp);
}
