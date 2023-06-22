#pragma once
#include <utility>
#include <algorithm>
#include <vector>
#include <map>
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
using std::vector;
using std::map;
using std::make_pair;
using std::move;
using std::min;
using std::max;




// LOUVAIN HASHTABLES
// ------------------

/**
 * Allocate a number of hashtables.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param S size of each hashtable
 */
template <class K, class W>
inline void louvainMapAllocateHashtablesW(vector<vector<K>>& vcs, vector<map<K, W>>& vcout, size_t S) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    vcs[i]   = vector<K>();
    vcout[i] = map<K, W>();
  }
}


/**
 * Free a number of hashtables.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 */
template <class K, class W>
inline void louvainMapFreeHashtablesW(vector<vector<K>>& vcs, vector<map<K, W>>& vcout) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    vcs[i].clear();
    vcout[i].clear();
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
inline void louvainMapVertexWeightsW(vector<W>& vtot, const G& x) {
  x.forEachVertexKey([&](auto u) {
    x.forEachEdge(u, [&](auto v, auto w) {
      vtot[u] += w;
    });
  });
}

#ifdef OPENMP
template <class G, class W>
inline void louvainMapVertexWeightsOmpW(vector<W>& vtot, const G& x) {
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
inline void louvainMapCommunityWeightsW(vector<W>& ctot, const G& x, const vector<K>& vcom, const vector<W>& vtot) {
  x.forEachVertexKey([&](auto u) {
    K c = vcom[u];
    ctot[c] += vtot[u];
  });
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvainMapCommunityWeightsOmpW(vector<W>& ctot, const G& x, const vector<K>& vcom, const vector<W>& vtot) {
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
inline void louvainMapInitializeW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot) {
  x.forEachVertexKey([&](auto u) {
    vcom[u] = u;
    ctot[u] = vtot[u];
  });
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvainMapInitializeOmpW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot) {
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
inline void louvainMapInitializeFromW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot, const vector<K>& q) {
  x.forEachVertexKey([&](auto u) {
    K c = q[u];
    vcom[u]  = c;
    ctot[c] += vtot[u];
  });
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvainMapInitializeFromOmpW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot, const vector<K>& q) {
  size_t S = x.span();
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = q[u];
    vcom[u]  = c;
    #pragma omp atomic
    ctot[c] += vtot[u];
  }
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
inline size_t louvainMapCountCommunityVerticesW(K *a, const G& x, const K *vcom) {
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
inline size_t louvainMapCountCommunityVerticesW(vector<K>& a, const G& x, const vector<K>& vcom) {
  return louvainMapCountCommunityVerticesW(a.data(), x, vcom.data());
}


#ifdef OPENMP
template <class G, class K>
inline size_t louvainMapCountCommunityVerticesOmpW(K *a, const G& x, const K *vcom) {
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
inline size_t louvainMapCountCommunityVerticesOmpW(vector<K>& a, const G& x, const vector<K>& vcom) {
  return louvainMapCountCommunityVerticesOmpW(a.data(), x, vcom.data());
}
#endif




/**
 * Find the vertices in each community.
 * @param co csr offsets for vertices belonging to each community (updated)
 * @param ce csr data vertices belonging to each community (updated)
 * @param cn number of vertices in each community (updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 */
template <class G, class K>
inline void louvainMapCommunityVerticesW(vector<K>& co, vector<K>& ce, vector<K>& cn, const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  co[S] = exclusiveScanW(co, cn);
  fillValueU(cn, K());
  x.forEachVertexKey([&](auto u) {
    K c = vcom[u];
    K i = cn[c]++;
    ce[co[c] + i] = u;
  });
}


#ifdef OPENMP
template <class G, class K>
inline void louvainMapCommunityVerticesOmpW(vector<K>& co, vector<K>& ce, vector<K>& cn, vector<K>& bufk, const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  co[S] = exclusiveScanOmpW(co, bufk, cn);
  fillValueOmpU(cn, K());
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u], i = 0;
    #pragma omp atomic capture
    { i = cn[c]; ++cn[c]; }
    ce[co[c] + i] = u;
  }
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
inline void louvainMapLookupCommunitiesU(vector<K>& a, const vector<K>& vcom) {
  for (auto& v : a)
    v = vcom[v];
}

#ifdef OPENMP
template <class K>
inline void louvainMapLookupCommunitiesOmpU(vector<K>& a, const vector<K>& vcom) {
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
inline void louvainMapScanCommunityW(vector<K>& vcs, map<K, W>& vcout, K u, K v, V w, const vector<K>& vcom) {
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
inline void louvainMapScanCommunitiesW(vector<K>& vcs, map<K, W>& vcout, const G& x, K u, const vector<K>& vcom) {
  x.forEachEdge(u, [&](auto v, auto w) { louvainMapScanCommunityW<SELF>(vcs, vcout, u, v, w, vcom); });
}


/**
 * Clear communities scan data.
 * @param vcs total edge weight from vertex u to community C (updated)
 * @param vcout communities vertex u is linked to (updated)
 */
template <class K, class W>
inline void louvainMapClearScanW(vector<K>& vcs, map<K, W>& vcout) {
  vcout.clear();
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
inline auto louvainMapChooseCommunity(const G& x, K u, const vector<K>& vcom, const vector<W>& vtot, const vector<W>& ctot, const vector<K>& vcs, map<K, W>& vcout, double M, double R) {
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
inline void louvainMapChangeCommunityW(vector<K>& vcom, vector<W>& ctot, const G& x, K u, K c, const vector<W>& vtot) {
  K d = vcom[u];
  ctot[d] -= vtot[u];
  ctot[c] += vtot[u];
  vcom[u] = c;
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvainMapChangeCommunityOmpW(vector<K>& vcom, vector<W>& ctot, const G& x, K u, K c, const vector<W>& vtot) {
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
 * @param vaff is vertex affected flag (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param L max iterations
 * @param fc has local moving phase converged?
 * @returns iterations performed (0 if converged already)
 */
template <class G, class K, class W, class B, class FC>
inline int louvainMapMoveW(vector<K>& vcom, vector<W>& ctot, vector<B>& vaff, vector<K>& vcs, map<K, W>& vcout, const G& x, const vector<W>& vtot, double M, double R, int L, FC fc) {
  int l = 0;
  W  el = W();
  for (; l<L;) {
    el = W();
    x.forEachVertexKey([&](auto u) {
      if (!vaff[u]) return;
      louvainMapClearScanW(vcs, vcout);
      louvainMapScanCommunitiesW(vcs, vcout, x, u, vcom);
      auto [c, e] = louvainMapChooseCommunity(x, u, vcom, vtot, ctot, vcs, vcout, M, R);
      if (c)      { louvainMapChangeCommunityW(vcom, ctot, x, u, c, vtot); x.forEachEdgeKey(u, [&](auto v) { vaff[v] = B(1); }); }
      vaff[u] = B();
      el += e;  // l1-norm
    });
    if (fc(el, l++)) break;
  }
  return l>1 || el? l : 0;
}

#ifdef OPENMP
template <class G, class K, class W, class B, class FC>
inline int louvainMapMoveOmpW(vector<K>& vcom, vector<W>& ctot, vector<B>& vaff, vector<vector<K>>& vcs, vector<map<K, W>>& vcout, const G& x, const vector<W>& vtot, double M, double R, int L, FC fc) {
  size_t S = x.span();
  int l = 0;
  W  el = W();
  for (; l<L;) {
    el = W();
    #pragma omp parallel for schedule(auto) reduction(+:el)
    for (K u=0; u<S; ++u) {
      int t = omp_get_thread_num();
      if (!x.hasVertex(u)) continue;
      if (!vaff[u]) continue;
      louvainMapClearScanW(vcs[t], vcout[t]);
      louvainMapScanCommunitiesW(vcs[t], vcout[t], x, u, vcom);
      auto [c, e] = louvainMapChooseCommunity(x, u, vcom, vtot, ctot, vcs[t], vcout[t], M, R);
      if (c)      { louvainMapChangeCommunityOmpW(vcom, ctot, x, u, c, vtot); x.forEachEdgeKey(u, [&](auto v) { vaff[v] = B(1); }); }
      vaff[u] = B();
      el += e;  // l1-norm
    }
    if (fc(el, l++)) break;
  }
  return l>1 || el? l : 0;
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
 * @param co csr offsets for vertices belonging to each community
 * @param ce csr data vertices belonging to each community
 */
template <class G, class K, class W>
inline void louvainMapAggregateW(G& a, vector<K>& vcs, map<K, W>& vcout, const G& x, const vector<K>& vcom, const vector<K>& co, const vector<K>& ce) {
  size_t S = x.span();
  a.respan(S);
  for (K c=0; c<S; ++c) {
    K oc = co[c];
    K nc = co[c+1] - co[c];
    if (nc==0) continue;
    louvainMapClearScanW(vcs, vcout);
    for (K i=0; i<nc; ++i) {
      K u = ce[oc+i];
      louvainMapScanCommunitiesW<true>(vcs, vcout, x, u, vcom);
    }
    // a.reserveEdges(c, vcs.size());
    a.addVertex(c);
    for (auto d : vcs)
      a.addEdge(c, d, vcout[d]);
  }
  // Aggregated graph has unique edges, so an update may not be necessary.
  a.update();
}

#ifdef OPENMP
template <class G, class K, class W>
inline void louvainMapAggregateOmpW(G& a, vector<vector<K>>& vcs, vector<map<K, W>>& vcout, const G& x, const vector<K>& vcom, const vector<K>& co, const vector<K>& ce) {
  size_t S = x.span();
  a.respan(S);
  #pragma omp parallel for schedule(auto)
  for (K c=0; c<S; ++c) {
    int t = omp_get_thread_num();
    K oc = co[c];
    K nc = co[c+1] - co[c];
    if (nc==0) continue;
    louvainMapClearScanW(vcs[t], vcout[t]);
    for (K i=0; i<nc; ++i) {
      K u = ce[oc+i];
      louvainMapScanCommunitiesW<true>(vcs[t], vcout[t], x, u, vcom);
    }
    // a.reserveEdges(c, (vcs[t]).size());
    for (auto d : vcs[t])
      a.addEdge(c, d, (vcout[t])[d]);
  }
  // Aggregated graph has unique edges, so an update may not be necessary.
  updateOmpU(a);
}
#endif


template <class G, class K, class W>
inline auto louvainMapAggregate(vector<K>& vcs, map<K, W>& vcout, const G& x, const vector<K>& vcom, const vector<K>& co, const vector<K>& ce) {
  G a; louvainMapAggregateW(a, vcs, vcout, x, vcom, co, ce);
  return a;
}

#ifdef OPENMP
template <class G, class K, class W>
inline auto louvainMapAggregateOmp(vector<vector<K>>& vcs, vector<map<K, W>>& vcout, const G& x, const vector<K>& vcom, const vector<K>& co, const vector<K>& ce) {
  G a; louvainMapAggregateOmpW(a, vcs, vcout, x, vcom, co, ce);
  return a;
}
#endif




// LOUVAIN
// -------

/**
 * Find the community each vertex belongs to.
 * @param x original graph
 * @param q initial community each vertex belongs to
 * @param o louvainMap options
 * @param fm marking affected vertices / preprocessing to be performed (vaff)
 * @returns community each vertex belongs to
 */
template <class FLAG=char, class G, class K, class FM>
auto louvainMapSeq(const G& x, const vector<K> *q, const LouvainOptions& o, FM fm) {
  using  W = LOUVAIN_WEIGHT_TYPE;
  using  B = FLAG;
  double R = o.resolution;
  int    L = o.maxIterations, l = 0;
  int    P = o.maxPasses, p = 0;
  size_t S = x.span(), naff = 0;
  double M = edgeWeight(x)/2;
  vector<K> vcom(S), vcs, a(S);
  vector<W> vtot(S), ctot(S); map<K, W> vcout;
  vector<K> co(S+1), ce(S), cn(S);
  vector<B> vaff(S);
  float tm = 0, tp = 0, tl = 0, ta = 0;
  float t  = measureDurationMarked([&](auto mark) {
    double E  = o.tolerance;
    auto   fc = [&](double el, int l) { return el<=E; };
    G y; y.respan(S);
    fillValueU(vcom, K());
    fillValueU(vtot, W());
    fillValueU(ctot, W());
    fillValueU(a, K());
    mark([&]() {
      tm += measureDuration([&]() { fm(vaff); });
      naff = sumValues(vaff, size_t());
      auto t0 = timeNow(), t1 = t0;
      louvainMapVertexWeightsW(vtot, x);
      if (q) louvainMapInitializeFromW(vcom, ctot, x, vtot, *q);
      else   louvainMapInitializeW(vcom, ctot, x, vtot);
      for (l=0, p=0; M>0 && p<P;) {
        if (p==1) t1 = timeNow();
        const G& g = p==0? x : y;
        int m = 0;
        tl += measureDuration([&]() { m = louvainMapMoveW(vcom, ctot, vaff, vcs, vcout, g, vtot, M, R, L, fc); });
        if (p==0) copyValuesW(a, vcom);
        else      louvainMapLookupCommunitiesU(a, vcom);
        l += max(m, 1); ++p;
        if (m<=1 || p>=P) break;
        size_t gn = g.order();
        size_t yn = louvainMapCountCommunityVerticesW(cn, g, vcom);
        if (double(yn)/gn >= o.aggregationTolerance) break;
        louvainMapCommunityVerticesW(co, ce, cn, g, vcom);
        ta += measureDuration([&]() { y = louvainMapAggregate(vcs, vcout, g, vcom, co, ce); });
        fillValueU(vcom, K());
        fillValueU(vtot, W());
        fillValueU(ctot, W());
        fillValueU(vaff, B(1));
        louvainMapVertexWeightsW(vtot, y);
        louvainMapInitializeW(vcom, ctot, y, vtot);
        E /= o.toleranceDecline;
      }
      if (p<=1) t1 = timeNow();
      tp += duration(t0, t1);
    });
  }, o.repeat);
  return LouvainResult<K>(a, l, p, t, tm/o.repeat, tp/o.repeat, tl/o.repeat, ta/o.repeat, naff);
}

#ifdef OPENMP
template <class FLAG=char, class G, class K, class FM>
auto louvainMapOmp(const G& x, const vector<K> *q, const LouvainOptions& o, FM fm) {
  using  W = LOUVAIN_WEIGHT_TYPE;
  using  B = FLAG;
  double R = o.resolution;
  int    L = o.maxIterations, l = 0;
  int    P = o.maxPasses, p = 0;
  size_t S = x.span(), naff = 0;
  double M = edgeWeightOmp(x)/2;
  int    T = omp_get_max_threads();
  vector<K> bufk(T);
  vector<K> vcom(S), a(S);
  vector<W> vtot(S), ctot(S);
  vector<K> co(S+1), ce(S), cn(S);
  vector<B> vaff(S);
  vector<vector<K>> vcs(T);
  vector<map<K, W>> vcout(T);
  louvainMapAllocateHashtablesW(vcs, vcout, S);
  float tm = 0, tp = 0, tl = 0, ta = 0;
  float t  = measureDurationMarked([&](auto mark) {
    double E  = o.tolerance;
    auto   fc = [&](double el, int l) { return el<=E; };
    G y; y.respan(S);
    fillValueOmpU(vcom, K());
    fillValueOmpU(vtot, W());
    fillValueOmpU(ctot, W());
    fillValueOmpU(a, K());
    mark([&]() {
      tm += measureDuration([&]() { fm(vaff); });
      naff = sumValuesOmp(vaff, size_t());
      auto t0 = timeNow(), t1 = t0;
      louvainMapVertexWeightsOmpW(vtot, x);
      if (q) louvainMapInitializeFromOmpW(vcom, ctot, x, vtot, *q);
      else   louvainMapInitializeOmpW(vcom, ctot, x, vtot);
      for (l=0, p=0; M>0 && p<P;) {
        if (p==1) t1 = timeNow();
        const G& g = p==0? x : y;
        int m = 0;
        tl += measureDuration([&]() { m = louvainMapMoveOmpW(vcom, ctot, vaff, vcs, vcout, g, vtot, M, R, L, fc); });
        if (p==0) copyValuesW(a, vcom);
        else      louvainMapLookupCommunitiesOmpU(a, vcom);
        l += max(m, 1); ++p;
        if (m<=1 || p>=P) break;
        size_t gn = g.order();
        size_t yn = louvainMapCountCommunityVerticesOmpW(cn, g, vcom);
        if (double(yn)/gn >= o.aggregationTolerance) break;
        louvainMapCommunityVerticesOmpW(co, ce, cn, bufk, g, vcom);
        ta += measureDuration([&]() { y = louvainMapAggregateOmp(vcs, vcout, g, vcom, co, ce); });
        fillValueOmpU(vcom, K());
        fillValueOmpU(vtot, W());
        fillValueOmpU(ctot, W());
        fillValueOmpU(vaff, B(1));
        louvainMapVertexWeightsOmpW(vtot, y);
        louvainMapInitializeOmpW(vcom, ctot, y, vtot);
        E /= o.toleranceDecline;
      }
      if (p<=1) t1 = timeNow();
      tp += duration(t0, t1);
    });
  }, o.repeat);
  louvainMapFreeHashtablesW(vcs, vcout);
  return LouvainResult<K>(a, l, p, t, tm/o.repeat, tp/o.repeat, tl/o.repeat, ta/o.repeat, naff);
}
#endif




// LOUVAIN-STATIC
// --------------

template <class FLAG=char, class G, class K>
inline auto louvainMapStaticSeq(const G& x, const vector<K>* q=nullptr, const LouvainOptions& o={}) {
  auto fm = [](auto& vertices) { fillValueU(vertices, FLAG(1)); };
  return louvainMapSeq<FLAG>(x, q, o, fm);
}

#ifdef OPENMP
template <class FLAG=char, class G, class K>
inline auto louvainMapStaticOmp(const G& x, const vector<K>* q=nullptr, const LouvainOptions& o={}) {
  auto fm = [](auto& vertices) { fillValueOmpU(vertices, FLAG(1)); };
  return louvainMapOmp<FLAG>(x, q, o, fm);
}
#endif