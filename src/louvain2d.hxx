#pragma once
#include <utility>
#include <algorithm>
#include <vector>
#include "_main.hxx"
#include "Graph.hxx"
#include "duplicate.hxx"
#include "properties.hxx"
#include "modularity.hxx"
#include "louvain.hxx"

#ifdef OPENMP
#include <omp.h>
#endif

using std::pair;
using std::tuple;
using std::vector;
using std::make_pair;
using std::move;
using std::get;
using std::min;




// LOUVAIN HASHTABLES
// ------------------

/**
 * Allocate a number of hashtables.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param S size of each hashtable
 */
template <class K, class W>
inline void louvain2dAllocateHashtablesW(vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, size_t S) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    vcs[i]   = new vector<K>();
    vcout[i] = new vector<W>(S);
  }
}


/**
 * Free a number of hashtables.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 */
template <class K, class W>
inline void louvain2dFreeHashtablesW(vector<vector<K>*>& vcs, vector<vector<W>*>& vcout) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    delete vcs[i];
    delete vcout[i];
  }
}




// LOUVAIN INITIALIZE
// ------------------

/**
 * Find the total edge weight of each vertex.
 * @param vtot total edge weight of each vertex (updated, should be initialized to 0)
 * @param x original graph
 */
template <class G, class W>
inline void louvain2dVertexWeightsW(vector<W>& vtot, const G& x) {
  x.forEachVertexKey([&](auto u) {
    x.forEachEdge(u, [&](auto v, auto w) {
      vtot[u] += w;
    });
  });
}

#ifdef OPENMP
template <class G, class W>
inline void louvain2dVertexWeightsOmpW(vector<W>& vtot, const G& x) {
  using  K = typename G::key_type;
  size_t S = x.span();
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    x.forEachEdge(u, [&](auto v, auto w) { vtot[u] += w; });
  }
}
#endif


/**
 * Find the total edge weight of each community.
 * @param ctot total edge weight of each community (updated, should be initialized to 0)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class W>
inline void louvain2dCommunityWeightsW(vector<W>& ctot, const G& x, const vector<K>& vcom, const vector<W>& vtot) {
  x.forEachVertexKey([&](auto u) {
    K c = vcom[u];
    ctot[c] += vtot[u];
  });
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvain2dCommunityWeightsOmpW(vector<W>& ctot, const G& x, const vector<K>& vcom, const vector<W>& vtot) {
  size_t S = x.span();
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u];
    #pragma omp atomic
    ctot[c] += vtot[u];
  }
}
#endif


/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to (updated, should be initialized to 0)
 * @param ctot total edge weight of each community (updated, should be initilized to 0)
 * @param x original graph
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class W>
inline void louvain2dInitializeW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot) {
  x.forEachVertexKey([&](auto u) {
    vcom[u] = u;
    ctot[u] = vtot[u];
  });
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvain2dInitializeOmpW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot) {
  size_t S = x.span();
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    vcom[u] = u;
    ctot[u] = vtot[u];
  }
}
#endif


/**
 * Initialize communities from given initial communities.
 * @param vcom community each vertex belongs to (updated, should be initialized to 0)
 * @param ctot total edge weight of each community (updated, should be initilized to 0)
 * @param x original graph
 * @param vtot total edge weight of each vertex
 * @param q initial community each vertex belongs to
 */
template <class G, class K, class W>
inline void louvain2dInitializeFromW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot, const vector<K>& q) {
  copyValuesW(vcom.data(), q.data(), min(q.size(), vcom.size()));
  louvain2dCommunityWeightsW(ctot, x, vcom, vtot);
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvain2dInitializeFromOmpW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot, const vector<K>& q) {
  copyValuesOmpW(vcom.data(), q.data(), min(q.size(), vcom.size()));
  louvain2dCommunityWeightsOmpW(ctot, x, vcom, vtot);
}
#endif




// LOUVAIN COMMUNITY VERTICES
// --------------------------

/**
 * Find the number of vertices in each community.
 * @param a number of vertices belonging to each community (updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @returns number of communities
 */
template <class G, class K>
inline size_t louvain2dCountCommunityVerticesW(K *a, const G& x, const K *vcom) {
  size_t S = x.span();
  size_t n = 0;
  fillValueU(a, S, K());
  x.forEachVertexKey([&](auto u) {
    K c = vcom[u];
    if (a[c]==0) ++n;
    ++a[c];
  });
  return n;
}
template <class G, class K>
inline size_t louvain2dCountCommunityVerticesW(vector<K>& a, const G& x, const vector<K>& vcom) {
  return louvain2dCountCommunityVerticesW(a.data(), x, vcom.data());
}


#ifdef OPENMP
/**
 * Find the number of vertices in each community.
 * @param a number of vertices belonging to each community (updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @returns number of communities
 */
template <class G, class K>
inline size_t louvain2dCountCommunityVerticesOmpW(K *a, const G& x, const K *vcom) {
  size_t S = x.span();
  size_t n = 0;
  fillValueOmpU(a, S, K());
  #pragma omp parallel for schedule(auto) reduction(+:n)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u], m = 0;
    #pragma omp atomic capture
    { m = a[c]; ++a[c]; }
    if (m==0) ++n;
  }
  return n;
}
template <class G, class K>
inline size_t louvain2dCountCommunityVerticesOmpW(vector<K>& a, const G& x, const vector<K>& vcom) {
  return louvain2dCountCommunityVerticesOmpW(a.data(), x, vcom.data());
}
#endif




/**
 * Find the vertices in each community.
 * @param a vertices belonging to each community (updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 */
template <class G, class K>
inline void louvain2dCommunityVerticesW(vector2d<K>& a, const G& x, const vector<K>& vcom) {
  x.forEachVertexKey([&](auto u) { a[vcom[u]].push_back(u); });
}
template <class G, class K>
inline auto louvain2dCommunityVertices(const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  vector2d<K> a(S);
  louvain2dCommunityVerticesW(a, x, vcom);
  return a;
}

#ifdef OPENMP
template <class G, class K>
inline void louvain2dCommunityVerticesOmpW(vector2d<K>& a, const G& x, const vector<K>& vcom) {
  #pragma omp parallel
  {
    x.forEachVertexKey([&](auto u) {
      if (belongsOmp(vcom[u])) a[vcom[u]].push_back(u);
    });
  }
}
template <class G, class K>
inline auto louvain2dCommunityVerticesOmp(const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  vector2d<K> a(S);
  louvain2dCommunityVerticesOmpW(a, x, vcom);
  return a;
}
#endif




// LOUVAIN LOOKUP COMMUNITIES
// --------------------------

/**
 * Update community membership in a tree-like fashion (to handle aggregation).
 * @param a output community each vertex belongs to (updated)
 * @param vcom community each vertex belongs to (at this aggregation level)
 */
template <class K>
inline void louvain2dLookupCommunitiesU(vector<K>& a, const vector<K>& vcom) {
  for (auto& v : a)
    v = vcom[v];
}

#ifdef OPENMP
template <class K>
inline void louvain2dLookupCommunitiesOmpU(vector<K>& a, const vector<K>& vcom) {
  size_t S = a.size();
  #pragma omp parallel for schedule(auto)
  for (size_t u=0; u<S; ++u)
    a[u] = vcom[a[u]];
}
#endif




// LOUVAIN CHANGE COMMUNITY
// ------------------------

/**
 * Scan an edge community connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param u given vertex
 * @param v outgoing edge vertex
 * @param w outgoing edge weight
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class K, class V, class W>
inline void louvain2dScanCommunityW(vector<K>& vcs, vector<W>& vcout, K u, K v, V w, const vector<K>& vcom) {
  if (!SELF && u==v) return;
  K c = vcom[v];
  if (!vcout[c]) vcs.push_back(c);
  vcout[c] += w;
}


/**
 * Scan communities connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class G, class K, class W>
inline void louvain2dScanCommunitiesW(vector<K>& vcs, vector<W>& vcout, const G& x, K u, const vector<K>& vcom) {
  x.forEachEdge(u, [&](auto v, auto w) { louvain2dScanCommunityW<SELF>(vcs, vcout, u, v, w, vcom); });
}


/**
 * Clear communities scan data.
 * @param vcs total edge weight from vertex u to community C (updated)
 * @param vcout communities vertex u is linked to (updated)
 */
template <class K, class W>
inline void louvain2dClearScanW(vector<K>& vcs, vector<W>& vcout) {
  for (K c : vcs)
    vcout[c] = W();
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
template <bool SELF=false, class G, class K, class W>
inline auto louvain2dChooseCommunity(const G& x, K u, const vector<K>& vcom, const vector<W>& vtot, const vector<W>& ctot, const vector<K>& vcs, const vector<W>& vcout, double M, double R) {
  K cmax = K(), d = vcom[u];
  W emax = W();
  for (K c : vcs) {
    if (!SELF && c==d) continue;
    W e = deltaModularity(vcout[c], vcout[d], vtot[u], ctot[c], ctot[d], M, R);
    if (e>emax) { emax = e; cmax = c; }
  }
  return make_pair(cmax, emax);
}


/**
 * Move vertex to another community C.
 * @param vcom community each vertex belongs to (updated)
 * @param ctot total edge weight of each community (updated)
 * @param x original graph
 * @param u given vertex
 * @param c community to move to
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class W>
inline void louvain2dChangeCommunityW(vector<K>& vcom, vector<W>& ctot, const G& x, K u, K c, const vector<W>& vtot) {
  K d = vcom[u];
  ctot[d] -= vtot[u];
  ctot[c] += vtot[u];
  vcom[u] = c;
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvain2dChangeCommunityOmpW(vector<K>& vcom, vector<W>& ctot, const G& x, K u, K c, const vector<W>& vtot) {
  K d = vcom[u];
  #pragma omp atomic
  ctot[d] -= vtot[u];
  #pragma omp atomic
  ctot[c] += vtot[u];
  vcom[u] = c;
}
#endif




// LOUVAIN MOVE
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
 * @param L max iterations
 * @param fc has local moving phase converged?
 * @param fa is a vertex affected?
 * @param fp process vertices whose communities have changed
 * @returns iterations performed
 */
template <class G, class K, class W, class FC, class FA, class FP>
inline int louvain2dMoveW(vector<K>& vcom, vector<W>& ctot, vector<K>& vcs, vector<W>& vcout, const G& x, const vector<W>& vtot, double M, double R, int L, FC fc, FA fa, FP fp) {
  int l = 0;
  for (; l<L;) {
    W el = W();
    x.forEachVertexKey([&](auto u) {
      if (!fa(u)) return;
      louvain2dClearScanW(vcs, vcout);
      louvain2dScanCommunitiesW(vcs, vcout, x, u, vcom);
      auto [c, e] = louvain2dChooseCommunity(x, u, vcom, vtot, ctot, vcs, vcout, M, R);
      if (c)      { louvain2dChangeCommunityW(vcom, ctot, x, u, c, vtot); fp(u); }
      el += e;  // l1-norm
    });
    if (fc(el, l++)) break;
  }
  return l;
}

#ifdef OPENMP
template <class G, class K, class W, class FC, class FA, class FP>
inline int louvain2dMoveOmpW(vector<K>& vcom, vector<W>& ctot, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<W>& vtot, double M, double R, int L, FC fc, FA fa, FP fp) {
  size_t S = x.span();
  int l = 0;
  for (; l<L;) {
    W el = W();
    #pragma omp parallel for schedule(auto) reduction(+:el)
    for (K u=0; u<S; ++u) {
      int t = omp_get_thread_num();
      if (!x.hasVertex(u)) continue;
      if (!fa(u)) continue;
      louvain2dClearScanW(*vcs[t], *vcout[t]);
      louvain2dScanCommunitiesW(*vcs[t], *vcout[t], x, u, vcom);
      auto [c, e] = louvain2dChooseCommunity(x, u, vcom, vtot, ctot, *vcs[t], *vcout[t], M, R);
      if (c)      { louvain2dChangeCommunityOmpW(vcom, ctot, x, u, c, vtot); fp(u); }
      el += e;  // l1-norm
    }
    if (fc(el, l++)) break;
  }
  return l;
}
#endif


template <class G, class K, class W, class FC>
inline int louvain2dMoveW(vector<K>& vcom, vector<W>& ctot, vector<K>& vcs, vector<W>& vcout, const G& x, const vector<W>& vtot, double M, double R, int L, FC fc) {
  auto fa = [](auto u) { return true; };
  auto fp = [](auto u) {};
  return louvain2dMoveW(vcom, ctot, vcs, vcout, x, vtot, M, R, L, fc, fa, fp);
}

#ifdef OPENMP
template <class G, class K, class W, class FC>
inline int louvain2dMoveOmpW(vector<K>& vcom, vector<W>& ctot, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<W>& vtot, double M, double R, int L, FC fc) {
  auto fa = [](auto u) { return true; };
  auto fp = [](auto u) {};
  return louvain2dMoveOmpW(vcom, ctot, vcs, vcout, x, vtot, M, R, L, fc, fa, fp);
}
#endif




// LOUVAIN AGGREGATE
// -----------------

/**
 * Louvain algorithm's community aggregation phase.
 * @param a output graph
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @param comv vertices belonging to each community
 */
template <class G, class K, class W>
inline void louvain2dAggregateW(G& a, vector<K>& vcs, vector<W>& vcout, const G& x, const vector<K>& vcom, const vector2d<K>& comv) {
  for (K c=0; c<comv.size(); ++c) {
    if (comv[c].empty()) continue;
    louvain2dClearScanW(vcs, vcout);
    for (K u : comv[c])
      louvain2dScanCommunitiesW<true>(vcs, vcout, x, u, vcom);
    a.addVertex(c);
    for (auto d : vcs)
      a.addEdge(c, d, vcout[d]);
  }
  a.update();
}
template <class G, class K, class W>
inline void louvain2dAggregateW(G& a, vector<K>& vcs, vector<W>& vcout, const G& x, const vector<K>& vcom) {
  auto comv = louvain2dCommunityVertices(x, vcom);
  louvain2dAggregateW(a, vcs, vcout, x, vcom, comv);
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvain2dAggregateOmpW(G& a, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcom, const vector2d<K>& comv) {
  size_t  S = x.span();
  for (K c=0; c<comv.size(); ++c) {
    if (comv[c].empty()) continue;
    a.addVertex(c);
  }
  #pragma omp parallel for schedule(auto)
  for (K c=0; c<comv.size(); ++c) {
    int t = omp_get_thread_num();
    if (comv[c].empty()) continue;
    louvain2dClearScanW(*vcs[t], *vcout[t]);
    for (K u : comv[c])
      louvain2dScanCommunitiesW<true>(*vcs[t], *vcout[t], x, u, vcom);
    for (auto d : *vcs[t])
      a.addEdge(c, d, (*vcout[t])[d]);
  }
  updateOmpU(a);
}
template <class G, class K, class W>
inline void louvain2dAggregateOmpW(G& a, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcom) {
  auto comv = louvain2dCommunityVerticesOmp(x, vcom);
  louvain2dAggregateOmpW(a, vcs, vcout, x, vcom, comv);
}
#endif


template <class G, class K, class W>
inline auto louvain2dAggregate(vector<K>& vcs, vector<W>& vcout, const G& x, const vector<K>& vcom, const vector2d<K>& comv) {
  G a; louvain2dAggregateW(a, vcs, vcout, x, vcom, comv);
  return a;
}
template <class G, class K, class W>
inline auto louvain2dAggregate(vector<K>& vcs, vector<W>& vcout, const G& x, const vector<K>& vcom) {
  G a; louvain2dAggregateW(a, vcs, vcout, x, vcom);
  return a;
}

#ifdef OPENMP
template <class G, class K, class W>
inline auto louvain2dAggregateOmp(vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcom, const vector2d<K>& comv) {
  G a; louvain2dAggregateOmpW(a, vcs, vcout, x, vcom, comv);
  return a;
}
template <class G, class K, class W>
inline auto louvain2dAggregateOmp(vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcom) {
  G a; louvain2dAggregateOmpW(a, vcs, vcout, x, vcom);
  return a;
}
#endif




// LOUVAIN AFFECTED VERTICES DELTA-SCREENING
// -----------------------------------------
// Using delta-screening approach.
// - All edge batches are undirected, and sorted by source vertex-id.
// - For edge additions across communities with source vertex `i` and highest modularity changing edge vertex `j*`,
//   `i`'s neighbors and `j*`'s community is marked as affected.
// - For edge deletions within the same community `i` and `j`,
//   `i`'s neighbors and `j`'s community is marked as affected.

/**
 * Find the vertices which should be processed upon a batch of edge insertions and deletions.
 * @param x original graph
 * @param deletions edge deletions for this batch update (undirected, sorted by source vertex id)
 * @param insertions edge insertions for this batch update (undirected, sorted by source vertex id)
 * @param vcom community each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param ctot total edge weight of each community
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @returns flags for each vertex marking whether it is affected
 */
template <class B, class G, class K, class V, class W>
inline auto louvain2dAffectedVerticesDeltaScreeningW(vector<K>& vcs, vector<W>& vcout, vector<B>& vertices, vector<B>& neighbors, vector<B>& communities, const G& x, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom, const vector<W>& vtot, const vector<W>& ctot, double M, double R=1) {
  fillValueU(vertices,    B());
  fillValueU(neighbors,   B());
  fillValueU(communities, B());
  for (const auto& [u, v] : deletions) {
    if (vcom[u] != vcom[v]) continue;
    vertices[u]  = 1;
    neighbors[u] = 1;
    communities[vcom[v]] = 1;
  }
  for (size_t i=0; i<insertions.size();) {
    K u = get<0>(insertions[i]);
    louvain2dClearScanW(vcs, vcout);
    for (; i<insertions.size() && get<0>(insertions[i])==u; ++i) {
      K v = get<1>(insertions[i]);
      V w = get<2>(insertions[i]);
      if (vcom[u] == vcom[v]) continue;
      louvain2dScanCommunityW(vcs, vcout, u, v, w, vcom);
    }
    auto [c, e] = louvain2dChooseCommunity(x, u, vcom, vtot, ctot, vcs, vcout, M, R);
    if (e<=0) continue;
    vertices[u]  = 1;
    neighbors[u] = 1;
    communities[c] = 1;
  }
  x.forEachVertexKey([&](auto u) {
    if (neighbors[u]) x.forEachEdgeKey(u, [&](auto v) { vertices[v] = 1; });
    if (communities[vcom[u]]) vertices[u] = 1;
  });
}


#ifdef OPENMP
template <class B, class G, class K, class V, class W>
inline auto louvain2dAffectedVerticesDeltaScreeningOmpW(vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, vector<B>& vertices, vector<B>& neighbors, vector<B>& communities, const G& x, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom, const vector<W>& vtot, const vector<W>& ctot, double M, double R=1) {
  size_t S = x.span();
  size_t D = deletions.size();
  size_t I = insertions.size();
  fillValueOmpU(vertices,    B());
  fillValueOmpU(neighbors,   B());
  fillValueOmpU(communities, B());
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<D; ++i) {
    K u = get<0>(deletions[i]);
    K v = get<1>(deletions[i]);
    if (vcom[u] != vcom[v]) continue;
    vertices[u]  = 1;
    neighbors[u] = 1;
    communities[vcom[v]] = 1;
  }
  #pragma omp parallel
  {
    int T = omp_get_num_threads();
    int t = omp_get_thread_num();
    K  u0 = I>0? get<0>(insertions[0]) : 0;
    for (size_t i=0, n=0; i<I;) {
      K u = get<0>(insertions[i]);
      if (u!=u0) { ++n; u0 = u; }
      if (n % T != t) { ++i; continue; }
      louvain2dClearScanW(*vcs[t], *vcout[t]);
      for (; i<I && get<0>(insertions[i])==u; ++i) {
        K v = get<1>(insertions[i]);
        V w = get<2>(insertions[i]);
        if (vcom[u] == vcom[v]) continue;
        louvain2dScanCommunityW(*vcs[t], *vcout[t], u, v, w, vcom);
      }
      auto [c, e] = louvain2dChooseCommunity(x, u, vcom, vtot, ctot, *vcs[t], *vcout[t], M, R);
      if (e<=0) continue;
      vertices[u]  = 1;
      neighbors[u] = 1;
      communities[c] = 1;
    }
  }
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    if (neighbors[u]) x.forEachEdgeKey(u, [&](auto v) { vertices[v] = 1; });
    if (communities[vcom[u]]) vertices[u] = 1;
  }
}
#endif




// LOUVAIN AFFECTED VERTICES FRONTIER
// ----------------------------------
// Using frontier based approach.
// - All source and destination vertices are marked as affected for insertions and deletions.
// - For edge additions across communities with source vertex `i` and destination vertex `j`,
//   `i` is marked as affected.
// - For edge deletions within the same community `i` and `j`,
//   `i` is marked as affected.
// - Vertices whose communities change in local-moving phase have their neighbors marked as affected.

/**
 * Find the vertices which should be processed upon a batch of edge insertions and deletions.
 * @param x original graph
 * @param deletions edge deletions for this batch update (undirected, sorted by source vertex id)
 * @param insertions edge insertions for this batch update (undirected, sorted by source vertex id)
 * @param vcom community each vertex belongs to
 * @returns flags for each vertex marking whether it is affected
 */
template <class B, class G, class K, class V>
inline void louvain2dAffectedVerticesFrontierW(vector<B>& vertices, const G& x, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom) {
  fillValueU(vertices, B());
  for (const auto& [u, v] : deletions) {
    if (vcom[u] != vcom[v]) continue;
    vertices[u]  = 1;
  }
  for (const auto& [u, v, w] : insertions) {
    if (vcom[u] == vcom[v]) continue;
    vertices[u]  = 1;
  }
}


#ifdef OPENMP
template <class B, class G, class K, class V>
inline void louvain2dAffectedVerticesFrontierOmpW(vector<B>& vertices, const G& x, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom) {
  fillValueOmpU(vertices, B());
  size_t D = deletions.size();
  size_t I = insertions.size();
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<D; ++i) {
    K u = get<0>(deletions[i]);
    K v = get<1>(deletions[i]);
    if (vcom[u] != vcom[v]) continue;
    vertices[u]  = 1;
  }
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<I; ++i) {
    K u = get<0>(insertions[i]);
    K v = get<1>(insertions[i]);
    if (vcom[u] == vcom[v]) continue;
    vertices[u]  = 1;
  }
}
#endif




// LOUVAIN
// -------

template <class G, class K, class FM, class FA, class FP>
auto louvain2dSeq(const G& x, const vector<K>* q, const LouvainOptions& o, FM fm, FA fa, FP fp) {
  using  W = LOUVAIN_WEIGHT_TYPE;
  double R = o.resolution;
  int    L = o.maxIterations, l = 0;
  int    P = o.maxPasses, p = 0;
  size_t S = x.span();
  double M = edgeWeight(x)/2;
  vector<K> vcom(S), vcs, a(S);
  vector<W> vtot(S), ctot(S), vcout(S);
  vector<K> comn(S);
  float tm = 0;
  float t  = measureDurationMarked([&](auto mark) {
    double E  = o.tolerance;
    auto   fc = [&](double el, int l) { return el < E; };
    G y; y.respan(S);
    fillValueU(vcom, K());
    fillValueU(vtot, W());
    fillValueU(ctot, W());
    mark([&]() {
      tm = measureDuration(fm);
      louvain2dVertexWeightsW(vtot, x);
      if (q) louvain2dInitializeFromW(vcom, ctot, x, vtot, *q);
      else   louvain2dInitializeW(vcom, ctot, x, vtot);
      for (l=0, p=0; M>0 && p<P;) {
        int m = 0;
        if (p==0) m = louvain2dMoveW(vcom, ctot, vcs, vcout, x, vtot, M, R, L, fc, fa, fp);
        else      m = louvain2dMoveW(vcom, ctot, vcs, vcout, y, vtot, M, R, L, fc);
        if (p==0) copyValuesW(a, vcom);
        else      louvain2dLookupCommunitiesU(a, vcom);
        l += m; ++p;
        if (m<=1 || p>=P) break;
        const G& g = p<=1? x : y;
        size_t gn = g.order();
        size_t yn = louvain2dCountCommunityVerticesW(comn, g, vcom);
        if (double(yn)/gn >= o.aggregationTolerance) break;
        y = louvain2dAggregate(vcs, vcout, g, vcom);
        fillValueU(vcom, K());
        fillValueU(vtot, W());
        fillValueU(ctot, W());
        louvain2dVertexWeightsW(vtot, y);
        louvain2dInitializeW(vcom, ctot, y, vtot);
        E /= o.toleranceDecline;
      }
    });
  }, o.repeat);
  return LouvainResult<K>(a, l, p, t, tm);
}

#ifdef OPENMP
template <class G, class K, class FM, class FA, class FP>
auto louvain2dOmp(const G& x, const vector<K>* q, const LouvainOptions& o, FM fm, FA fa, FP fp) {
  using  W = LOUVAIN_WEIGHT_TYPE;
  double R = o.resolution;
  int    L = o.maxIterations, l = 0;
  int    P = o.maxPasses, p = 0;
  size_t S = x.span();
  double M = edgeWeightOmp(x)/2;
  int    T = omp_get_max_threads();
  vector<K> vcom(S), a(S);
  vector<W> vtot(S), ctot(S);
  vector<K> comn(S);
  vector<vector<K>*> vcs(T);
  vector<vector<W>*> vcout(T);
  louvain2dAllocateHashtablesW(vcs, vcout, S);
  float tm = 0;
  float t  = measureDurationMarked([&](auto mark) {
    double E  = o.tolerance;
    auto   fc = [&](double el, int l) { return el < E; };
    G y; y.respan(S);
    fillValueOmpU(vcom, K());
    fillValueOmpU(vtot, W());
    fillValueOmpU(ctot, W());
    mark([&]() {
      tm = measureDuration(fm);
      louvain2dVertexWeightsOmpW(vtot, x);
      if (q) louvain2dInitializeFromOmpW(vcom, ctot, x, vtot, *q);
      else   louvain2dInitializeOmpW(vcom, ctot, x, vtot);
      for (l=0, p=0; M>0 && p<P;) {
        int m = 0;
        if (p==0) m = louvain2dMoveOmpW(vcom, ctot, vcs, vcout, x, vtot, M, R, L, fc, fa, fp);
        else      m = louvain2dMoveOmpW(vcom, ctot, vcs, vcout, y, vtot, M, R, L, fc);
        if (p==0) copyValuesW(a, vcom);
        else      louvain2dLookupCommunitiesOmpU(a, vcom);
        l += m; ++p;
        if (m<=1 || p>=P) break;
        const G& g = p<=1? x : y;
        size_t gn = g.order();
        size_t yn = louvain2dCountCommunityVerticesOmpW(comn, g, vcom);
        if (double(yn)/gn >= o.aggregationTolerance) break;
        y = louvain2dAggregateOmp(vcs, vcout, g, vcom);
        fillValueOmpU(vcom, K());
        fillValueOmpU(vtot, W());
        fillValueOmpU(ctot, W());
        louvain2dVertexWeightsOmpW(vtot, y);
        louvain2dInitializeOmpW(vcom, ctot, y, vtot);
        E /= o.toleranceDecline;
      }
    });
  }, o.repeat);
  louvain2dFreeHashtablesW(vcs, vcout);
  return LouvainResult<K>(a, l, p, t, tm);
}
#endif




// LOUVAIN-STATIC
// --------------

template <class G, class K>
inline auto louvain2dStaticSeq(const G& x, const vector<K>* q=nullptr, const LouvainOptions& o={}) {
  auto fm = []() {};
  auto fa = [](auto u) { return true; };
  auto fp = [](auto u) {};
  return louvain2dSeq(x, q, o, fm, fa, fp);
}

#ifdef OPENMP
template <class G, class K>
inline auto louvain2dStaticOmp(const G& x, const vector<K>* q=nullptr, const LouvainOptions& o={}) {
  auto fm = []() {};
  auto fa = [](auto u) { return true; };
  auto fp = [](auto u) {};
  return louvain2dOmp(x, q, o, fm, fa, fp);
}
#endif
