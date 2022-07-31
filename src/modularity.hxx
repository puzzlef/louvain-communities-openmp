#pragma once
#include <cmath>
#include <vector>

using std::pow;
using std::vector;




// MODULARITY
// ----------

/**
 * Find the modularity of a community C.
 * @param cin total weight of edges within community C
 * @param ctot total weight of edges of community C
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @returns modularity [-0.5, 1]
 * @see https://www.youtube.com/watch?v=0zuiLBOIcsw
 */
template <class T>
inline T modularityCommunity(T cin, T ctot, T M, T R=T(1)) {
  return cin/(2*M) - R*pow(ctot/(2*M), 2);
}


/**
 * Find the modularity of a set of communities.
 * @param cin total weight of edges within each community
 * @param ctot total weight of edges of each community
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @returns modularity [-0.5, 1]
 */
template <class T>
T modularityCommunities(const vector<T>& cin, const vector<T>& ctot, T M, T R=T(1)) {
  T a = T();
  for (size_t i=0, I=cin.size(); i<I; i++)
    a += modularityCommunity(cin[i], ctot[i], M, R);
  return a;
}


/**
 * Find the modularity of a graph, based on community membership function.
 * @param x original graph
 * @param fc community membership function of each vertex (u)
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @returns modularity [-0.5, 1]
 */
template <class G, class FC, class T>
auto modularity(const G& x, FC fc, T M, T R=T(1)) {
  size_t S = x.span();
  vector<T> cin(S), ctot(S);
  x.forEachVertexKey([&](auto u) {
    size_t c = fc(u);
    x.forEachEdge(u, [&](auto v, auto w) {
      size_t d = fc(v);
      if (c==d) cin[c] += w;
      ctot[c] += w;
    });
  });
  return modularityCommunities(cin, ctot, M, R);
}

/**
 * Find the modularity of a graph, where each vertex is its own community.
 * @param x original graph
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @returns modularity [-0.5, 1]
 */
template <class G, class T>
inline auto modularity(const G& x, T M, T R=T(1)) {
  auto fc = [](auto u) { return u; };
  return modularity(x, fc, M, R);
}




// DELTA-MODULARITY
// ----------------

/**
 * Find the change in modularity when moving a vertex from community D to C.
 * @param vcout total weight of edges from vertex v to community C
 * @param vdout total weight of edges from vertex v to community D
 * @param vtot total weight of edges from vertex v
 * @param ctot total weight of edges from community C
 * @param dtot total weight of edges from community C
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @returns delta-modularity [-0.5, 1]
 * @see https://gist.github.com/wolfram77/a3c95cd94a38a100f9b075594a823928
 */
template <class T>
inline T deltaModularity(T vcout, T vdout, T vtot, T ctot, T dtot, T M, T R=T(1)) {
  return (vcout-vdout)/M - R*vtot*(vtot+ctot-dtot)/(2*M*M);
}
