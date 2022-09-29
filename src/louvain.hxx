#pragma once
#include <utility>
#include <vector>
#include "_main.hxx"
#include "Graph.hxx"
#include "duplicate.hxx"
#include "modularity.hxx"

using std::pair;
using std::tuple;
using std::vector;
using std::get;
using std::move;
using std::make_pair;




// LOUVAIN-OPTIONS
// ---------------

template <class T>
struct LouvainOptions {
  int repeat;
  T   resolution;
  T   tolerance;
  T   passTolerance;
  T   tolerenceDeclineFactor;
  int maxIterations;
  int maxPasses;

  LouvainOptions(int repeat=1, T resolution=1, T tolerance=1e-2, T passTolerance=0, T tolerenceDeclineFactor=10, int maxIterations=500, int maxPasses=500) :
  repeat(repeat), resolution(resolution), tolerance(tolerance), passTolerance(passTolerance), tolerenceDeclineFactor(tolerenceDeclineFactor), maxIterations(maxIterations), maxPasses(maxPasses) {}
};




// LOUVAIN-RESULT
// --------------

template <class K>
struct LouvainResult {
  vector<K> membership;
  int   iterations;
  int   passes;
  float time;

  LouvainResult(vector<K>&& membership, int iterations=0, int passes=0, float time=0) :
  membership(membership), iterations(iterations), passes(passes), time(time) {}

  LouvainResult(vector<K>& membership, int iterations=0, int passes=0, float time=0) :
  membership(move(membership)), iterations(iterations), passes(passes), time(time) {}
};




// LOUVAIN-INITIALIZE
// ------------------

/**
 * Find the total edge weight of each vertex.
 * @param vtot total edge weight of each vertex (updated, should be initialized to 0)
 * @param x original graph
 */
template <class G, class V>
void louvainVertexWeights(vector<V>& vtot, const G& x) {
  x.forEachVertexKey([&](auto u) {
    x.forEachEdge(u, [&](auto v, auto w) {
      vtot[u] += w;
    });
  });
}


/**
 * Find the total edge weight of each community.
 * @param ctot total edge weight of each community (updated, should be initialized to 0)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class V>
void louvainCommunityWeights(vector<V>& ctot, const G& x, const vector<K>& vcom, const vector<V>& vtot) {
  x.forEachVertexKey([&](auto u) {
    K c = vcom[u];
    ctot[c] += vtot[u];
  });
}


/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to (updated, should be initialized to 0)
 * @param ctot total edge weight of each community (updated, should be initilized to 0)
 * @param x original graph
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class V>
void louvainInitialize(vector<K>& vcom, vector<V>& ctot, const G& x, const vector<V>& vtot) {
  x.forEachVertexKey([&](auto u) {
    vcom[u] = u;
    ctot[u] = vtot[u];
  });
}




// LOUVAIN-CHANGE-COMMUNITY
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
template <bool SELF=false, class K, class V>
void louvainScanCommunity(vector<K>& vcs, vector<V>& vcout, K u, K v, V w, const vector<K>& vcom) {
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
template <bool SELF=false, class G, class K, class V>
void louvainScanCommunities(vector<K>& vcs, vector<V>& vcout, const G& x, K u, const vector<K>& vcom) {
  x.forEachEdge(u, [&](auto v, auto w) { louvainScanCommunity<SELF>(vcs, vcout, u, v, w, vcom); });
}


/**
 * Clear communities scan data.
 * @param vcs total edge weight from vertex u to community C (updated)
 * @param vcout communities vertex u is linked to (updated)
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
template <bool SELF=false, class G, class K, class V>
auto louvainChooseCommunity(const G& x, K u, const vector<K>& vcom, const vector<V>& vtot, const vector<V>& ctot, const vector<K>& vcs, const vector<V>& vcout, V M, V R) {
  K cmax = K(), d = vcom[u];
  V emax = V();
  for (K c : vcs) {
    if (!SELF && c==d) continue;
    V e = deltaModularity(vcout[c], vcout[d], vtot[u], ctot[c], ctot[d], M, R);
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
template <class G, class K, class V>
void louvainChangeCommunity(vector<K>& vcom, vector<V>& ctot, const G& x, K u, K c, const vector<V>& vtot) {
  K d = vcom[u];
  ctot[d] -= vtot[u];
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
int louvainMove(vector<K>& vcom, vector<V>& ctot, vector<K>& vcs, vector<V>& vcout, const G& x, const vector<V>& vtot, V M, V R, V E, int L, FA fa, FP fp) {
  K S = x.span();
  int l = 0; V Q = V();
  for (; l<L;) {
    V el = V();
    x.forEachVertexKey([&](auto u) {
      if (!fa(u)) return;
      louvainClearScan(vcs, vcout);
      louvainScanCommunities(vcs, vcout, x, u, vcom);
      auto [c, e] = louvainChooseCommunity(x, u, vcom, vtot, ctot, vcs, vcout, M, R);
      if (c)      { louvainChangeCommunity(vcom, ctot, x, u, c, vtot); fp(u); }
      el += e;  // l1-norm
    }); ++l;
    if (el<=E) break;
  }
  return l;
}
template <class G, class K, class V, class FA>
inline int louvainMove(vector<K>& vcom, vector<V>& ctot, vector<K>& vcs, vector<V>& vcout, const G& x, const vector<V>& vtot, V M, V R, V E, int L, FA fa) {
  auto fp = [](auto u) {};
  return louvainMove(vcom, ctot, vcs, vcout, x, vtot, M, R, E, L, fa, fp);
}
template <class G, class K, class V>
inline int louvainMove(vector<K>& vcom, vector<V>& ctot, vector<K>& vcs, vector<V>& vcout, const G& x, const vector<V>& vtot, V M, V R, V E, int L) {
  auto fa = [](auto u) { return true; };
  return louvainMove(vcom, ctot, vcs, vcout, x, vtot, M, R, E, L, fa);
}




// LOUVAIN-AGGREGATE
// -----------------

template <class G, class K>
auto louvainCommunityVertices(const G& x, const vector<K>& vcom) {
  K S = x.span();
  vector2d<K> a(S);
  x.forEachVertexKey([&](auto u) { a[vcom[u]].push_back(u); });
  return a;
}


/**
 * Louvain algorithm's community aggregation phase.
 * @param a output graph
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 */
template <class G, class K, class V>
void louvainAggregate(G& a, vector<K>& vcs, vector<V>& vcout, const G& x, const vector<K>& vcom) {
  K S = x.span();
  auto comv = louvainCommunityVertices(x, vcom);
  for (K c=0; c<comv.size(); ++c) {
    if (comv[c].empty()) continue;
    louvainClearScan(vcs, vcout);
    for (K u : comv[c])
      louvainScanCommunities<true>(vcs, vcout, x, u, vcom);
    a.addVertex(c);
    for (auto d : vcs)
      a.addEdge(c, d, vcout[d]);
  }
  a.correct();
}
template <class G, class K, class V>
inline auto louvainAggregate(vector<K>& vcs, vector<V>& vcout, const G& x, const vector<K>& vcom) {
  G a; louvainAggregate(a, vcs, vcout, x, vcom);
  return a;
}




// LOUVAIN-LOOKUP-COMMUNITIES
// --------------------------

/**
 * Update community membership in a tree-like fashion (to handle aggregation).
 * @param a output community each vertex belongs to (updated)
 * @param vcom community each vertex belongs to (at this aggregation level)
 */
template <class K>
void louvainLookupCommunities(vector<K>& a, const vector<K>& vcom) {
  for (auto& v : a)
    v = vcom[v];
}




// LOUVAIN-AFFECTED-VERTICES-DELTA-SCREENING
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
template <class G, class K, class V>
auto louvainAffectedVerticesDeltaScreening(const G& x, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom, const vector<V>& vtot, const vector<V>& ctot, V M, V R=V(1)) {
  K S = x.span();
  vector<K> vcs; vector<V> vcout(S);
  vector<bool> vertices(S), neighbors(S), communities(S);
  for (const auto& [u, v] : deletions) {
    if (vcom[u] != vcom[v]) continue;
    vertices[u]  = true;
    neighbors[u] = true;
    communities[vcom[v]] = true;
  }
  for (size_t i=0; i<insertions.size();) {
    K u = get<0>(insertions[i]);
    louvainClearScan(vcs, vcout);
    for (; i<insertions.size() && get<0>(insertions[i])==u; ++i) {
      K v = get<1>(insertions[i]);
      V w = get<2>(insertions[i]);
      if (vcom[u] == vcom[v]) continue;
      louvainScanCommunity(vcs, vcout, u, v, w, vcom);
    }
    auto [c, e] = louvainChooseCommunity(x, u, vcom, vtot, ctot, vcs, vcout, M, R);
    vertices[u]  = true;
    neighbors[u] = true;
    communities[c] = true;
  }
  x.forEachVertexKey([&](auto u) {
    if (neighbors[u]) x.forEachEdgeKey(u, [&](auto v) { vertices[v] = true; });
    if (communities[vcom[u]]) vertices[u] = true;
  });
  return vertices;
}




// LOUVAIN-AFFECTED-VERTICES-FRONTIER
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
template <class G, class K, class V>
auto louvainAffectedVerticesFrontier(const G& x, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom) {
  K S = x.span();
  vector<bool> vertices(S);
  for (const auto& [u, v] : deletions) {
    if (vcom[u] != vcom[v]) continue;
    vertices[u]  = true;
  }
  for (const auto& [u, v, w] : insertions) {
    if (vcom[u] == vcom[v]) continue;
    vertices[u]  = true;
  }
  return vertices;
}
