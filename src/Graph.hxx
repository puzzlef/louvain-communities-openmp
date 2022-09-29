#pragma once
#include <utility>
#include <vector>
#include <ostream>
#include <iostream>
#include "_main.hxx"

using std::pair;
using std::vector;
using std::ostream;
using std::cout;




// GRAPH-*
// -------
// Helps create graphs.

#ifndef GRAPH_TYPES
#define GRAPH_SHORT_TYPES_FROM(G) \
  using K = typename G::key_type; \
  using V = typename G::vertex_value_type; \
  using E = typename G::edge_value_type;
#define GRAPH_TYPES(K, V, E) \
  using key_type = K; \
  using vertex_key_type   = K; \
  using vertex_value_type = V; \
  using vertex_pair_type  = pair<K, V>; \
  using edge_key_type     = K; \
  using edge_value_type   = E; \
  using edge_pair_type    = pair<K, E>;
#endif


#ifndef GRAPH_SIZE
#define GRAPH_SIZE(K, V, E, N, M, vexists)  \
  inline K span()  const noexcept { return vexists.size(); } \
  inline K order() const noexcept { return N; } \
  inline size_t size() const noexcept { return M; }
#define GRAPH_SIZE_FROM(K, V, E, x) \
  inline K span()  const noexcept { return x.span(); } \
  inline K order() const noexcept { return x.order(); } \
  inline size_t size() const noexcept { return x.size(); }
#define GRAPH_EMPTY(K, V, E) \
  inline bool empty()   const noexcept { return order() == 0; }
#define GRAPH_SIZES(K, V, E, N, M, vexists) \
  GRAPH_SIZE(K, V, E, N, M, vexists) \
  GRAPH_EMPTY(K, V, E)
#define GRAPH_SIZES_FROM(K, V, E, x) \
  GRAPH_SIZE_FROM(K, V, E, x) \
  GRAPH_EMPTY(K, V, E)
#endif


#ifndef GRAPH_DIRECTED
#define GRAPH_DIRECTED(K, V, E, de) \
  inline bool directed()   const noexcept { return de; }
#define GRAPH_DIRECTED_FROM(K, V, E, x) \
  inline bool directed()   const noexcept { return x.directed(); }
#define GRAPH_UNDIRECTED(K, V, E) \
  inline bool undirected() const noexcept { return !directed(); }
#define GRAPH_DIRECTEDNESS(K, V, E, de) \
  GRAPH_DIRECTED(K, V, E, de) \
  GRAPH_UNDIRECTED(K, V, E)
#define GRAPH_DIRECTEDNESS_FROM(K, V, E, x) \
  GRAPH_DIRECTED_FROM(K, V, E, x) \
  GRAPH_UNDIRECTED(K, V, E)
#endif


#ifndef GRAPH_ENTRIES
#define GRAPH_CVERTICES(K, V, E, vexists, vvalues) \
  inline auto cvertexKeys() const noexcept { \
    auto vkeys = rangeIterable(span()); \
    return conditionalIterable(vkeys, vexists); \
  } \
  inline auto cvertexValues() const noexcept { \
    return conditionalIterable(vvalues, vexists); \
  } \
  inline auto cvertices() const noexcept { \
    auto vkeys = rangeIterable(span()); \
    auto pairs = pairIterable(vkeys, vvalues); \
    return conditionalIterable(pairs, vexists); \
  }
#define GRAPH_CEDGES(K, V, E, eto, enone) \
  inline auto cedgeKeys(const K& u) const noexcept { \
    return u<span()? eto[u].ckeys()   : enone.ckeys(); \
  } \
  inline auto cedgeValues(const K& u) const noexcept { \
    return u<span()? eto[u].cvalues() : enone.cvalues(); \
  } \
  inline auto cedges(const K& u) const noexcept { \
    return u<span()? eto[u].cpairs()  : enone.cpairs(); \
  }
#define GRAPH_CINEDGES(K, V, E, efrom, enone) \
  inline auto cinEdgeKeys(const K& v) const noexcept { \
    return v<span()? efrom[v].ckeys()   : enone.ckeys(); \
  } \
  inline auto cinEdgeValues(const K& v) const noexcept { \
    return v<span()? efrom[v].cvalues() : enone.cvalues(); \
  } \
  inline auto cinEdges(const K& v) const noexcept { \
    return v<span()? efrom[v].cpairs()  : enone.cpairs(); \
  }
#define GRAPH_CINEDGES_SEARCH(K, V, E, eto) \
  inline auto cinEdgeKeys(const K& v) const noexcept { \
    auto vkeys = rangeIterable(span()); \
    auto fedge = [&](const K& u) { return eto[u].has(v); }; \
    return filterIterable(vkeys, fedge); \
  } \
  inline auto cinEdgeValues(const K& v) const noexcept { \
    auto fvals = [&](const K& u) { return eto[u][v]; }; \
    return transformIterable(cinEdgeKeys(v), fvals); \
  } \
  inline auto cinEdges(const K& v) const noexcept { \
    return pairIterable(cinEdgeKeys(v), cinEdgeValues(v)); \
  }

#define GRAPH_CVERTICES_FROM(K, V, E, x) \
  inline auto cvertexKeys()   const noexcept { return x.cvertexKeys(); } \
  inline auto cvertexValues() const noexcept { return x.cvertexValues(); } \
  inline auto cvertices()     const noexcept { return x.cvertices(); }
#define GRAPH_CEDGES_FROM(K, V, E, x, name) \
  inline auto cedgeKeys(const K& u)   const noexcept { return x.c##name##Keys(u); } \
  inline auto cedgeValues(const K& u) const noexcept { return x.c##name##Values(u); } \
  inline auto cedges(const K& u)      const noexcept { return x.c##name##s(u); }
#define GRAPH_CINEDGES_FROM(K, V, E, x, name) \
  inline auto cinEdgeKeys(const K& u)   const noexcept { return x.c##name##Keys(u); } \
  inline auto cinEdgeValues(const K& u) const noexcept { return x.c##name##Values(u); } \
  inline auto cinEdges(const K& u)      const noexcept { return x.c##name##s(u); }

#define GRAPH_VERTICES(K, V, E) \
  inline auto vertexKeys()    const noexcept { return cvertexKeys(); } \
  inline auto vertexValues()  const noexcept { return cvertexValues(); } \
  inline auto vertices()      const noexcept { return cvertices(); }
#define GRAPH_EDGES(K, V, E) \
  inline auto edgeKeys(const K& u)    const noexcept { return cedgeKeys(u); } \
  inline auto edgeValues(const K& u)  const noexcept { return cedgeValues(u); } \
  inline auto edges(const K& u)       const noexcept { return cedges(u); }
#define GRAPH_INEDGES(K, V, E) \
  inline auto inEdgeKeys(const K& v)    const noexcept { return cinEdgeKeys(v); } \
  inline auto inEdgeValues(const K& v)  const noexcept { return cinEdgeValues(v); } \
  inline auto inEdges(const K& v)       const noexcept { return cinEdges(v); }

#define GRAPH_CENTRIES(K, V, E, vexists, vvalues, eto, efrom, enone) \
  GRAPH_CVERTICES(K, V, E, vexists, vvalues) \
  GRAPH_CEDGES(K, V, E, eto, enone) \
  GRAPH_CINEDGES(K, V, E, efrom, enone)
#define GRAPH_CENTRIES_FROM(K, V, E, x, ename, iname) \
  GRAPH_CVERTICES_FROM(K, V, E, x) \
  GRAPH_CEDGES_FROM(K, V, E, x, ename) \
  GRAPH_CINEDGES_FROM(K, V, E, x, iname)
#define GRAPH_ENTRIES(K, V, E) \
  GRAPH_VERTICES(K, V, E) \
  GRAPH_EDGES(K, V, E) \
  GRAPH_INEDGES(K, V, E)
#endif


#ifndef GRAPH_FOREACH
#define GRAPH_CFOREACH_VERTEX(K, V, E, vexists, vvalues) \
  template <class F> \
  inline void cforEachVertexKey(F fn) const noexcept { \
    for (K u = K(); u < span(); ++u) \
      if (vexists[u]) fn(u); \
  } \
  template <class F> \
  inline void cforEachVertexValue(F fn) const noexcept { \
    for (K u = K(); u < span(); ++u) \
      if (vexists[u]) fn(vvalues[u]); \
  } \
  template <class F> \
  inline void cforEachVertex(F fn) const noexcept { \
    for (K u = K(); u < span(); ++u) \
      if (vexists[u]) fn(u, vvalues[u]); \
  }
#define GRAPH_CFOREACH_XEDGE(K, V, E, name, u, fn, eto) \
  template <class F> \
  inline void cforEach##name##Key(const K& u, F fn) const noexcept { \
    if (u < span()) eto[u].cforEachKey(fn); \
  } \
  template <class F> \
  inline void cforEach##name##Value(const K& u, F fn) const noexcept { \
    if (u < span()) eto[u].cforEachValue(fn); \
  } \
  template <class F> \
  inline void cforEach##name(const K& u, F fn) const noexcept { \
    if (u < span()) eto[u].cforEach(fn); \
  }
#define GRAPH_CFOREACH_INEDGE_SEARCH(K, V, E, eto) \
  template <class F> \
  inline void cforEachInEdgeKey(const K& v, F fn) const noexcept { \
    for (K u = K(); u < span(); ++u) \
      if (eto[u].has(v)) fn(u); \
  } \
  template <class F> \
  inline void cforEachInEdgeValue(const K& v, F fn) const noexcept { \
    for (K u = K(); u < span(); ++u) \
      if (eto[u].has(v)) fn(eto[u][v]); \
  } \
  template <class F> \
  inline void cforEachInEdge(const K& v, F fn) const noexcept { \
    for (K u = K(); u < span(); ++u) \
      if (eto[u].has(v)) fn(u, eto[u][v]); \
  }
#define GRAPH_CFOREACH_EDGE(K, V, E, eto) \
  GRAPH_CFOREACH_XEDGE(K, V, E, Edge, u, fn, eto)
#define GRAPH_CFOREACH_INEDGE(K, V, E, efrom) \
  GRAPH_CFOREACH_XEDGE(K, V, E, InEdge, v, fn, efrom)

#define GRAPH_CFOREACH_VERTEX_FROM(K, V, E, x) \
  template <class F> \
  inline void cforEachVertexKey(F fn)   const noexcept { x.cforEachVertexKey(fn); } \
  template <class F> \
  inline void cforEachVertexValue(F fn) const noexcept { x.cforEachVertexValue(fn); } \
  template <class F> \
  inline void cforEachVertex(F fn)      const noexcept { x.cforEachVertex(fn); }
#define GRAPH_CFOREACH_XEDGE_FROM(K, V, E, name, x, ename) \
  template <class F> \
  inline void cforEach##name##Key(const K& u, F fn)   const noexcept { x.cforEach##ename##Key(u, fn); } \
  template <class F> \
  inline void cforEach##name##Value(const K& u, F fn) const noexcept { x.cforEach##ename##Value(u, fn); } \
  template <class F> \
  inline void cforEach##name(const K& u, F fn)        const noexcept { x.cforEach##ename(u, fn); }
#define GRAPH_CFOREACH_EDGE_FROM(K, V, E, x, name) \
  GRAPH_CFOREACH_XEDGE_FROM(K, V, E, Edge, x, name)
#define GRAPH_CFOREACH_INEDGE_FROM(K, V, E, x, name) \
  GRAPH_CFOREACH_XEDGE_FROM(K, V, E, InEdge, x, name)

#define GRAPH_FOREACH_VERTEX(K, V, E) \
  template <class F> \
  inline void forEachVertexKey(F fn)   const noexcept { cforEachVertexKey(fn); } \
  template <class F> \
  inline void forEachVertexValue(F fn) const noexcept { cforEachVertexValue(fn); } \
  template <class F> \
  inline void forEachVertex(F fn)      const noexcept { cforEachVertex(fn); }
#define GRAPH_FOREACH_XEDGE(K, V, E, name) \
  template <class F> \
  inline void forEach##name##Key(const K& u, F fn)    const noexcept { cforEach##name##Key(u, fn); } \
  template <class F> \
  inline void forEach##name##Value(const K& u, F fn)  const noexcept { cforEach##name##Value(u, fn); } \
  template <class F> \
  inline void forEach##name(const K& u, F fn)         const noexcept { cforEach##name(u, fn); }
#define GRAPH_FOREACH_EDGE(K, V, E) \
  GRAPH_FOREACH_XEDGE(K, V, E, Edge)
#define GRAPH_FOREACH_INEDGE(K, V, E) \
  GRAPH_FOREACH_XEDGE(K, V, E, InEdge)

#define GRAPH_CFOREACH(K, V, E, vexists, vvalues, eto, efrom) \
  GRAPH_CFOREACH_VERTEX(K, V, E, vexists, vvalues) \
  GRAPH_CFOREACH_EDGE(K, V, E, eto) \
  GRAPH_CFOREACH_INEDGE(K, V, E, efrom)
#define GRAPH_CFOREACH_FROM(K, V, E, x, ename, iname) \
  GRAPH_CFOREACH_VERTEX_FROM(K, V, E, x) \
  GRAPH_CFOREACH_EDGE_FROM(K, V, E, x, ename) \
  GRAPH_CFOREACH_INEDGE_FROM(K, V, E, x, iname)
#define GRAPH_FOREACH(K, V, E) \
  GRAPH_FOREACH_VERTEX(K, V, E) \
  GRAPH_FOREACH_EDGE(K, V, E) \
  GRAPH_FOREACH_INEDGE(K, V, E)
#endif


#ifndef GRAPH_HAS
#define GRAPH_HAS(K, V, E, vexists, eto) \
  inline bool hasVertex(const K& u) const noexcept { \
    return u < span() && vexists[u]; \
  } \
  inline bool hasEdge(const K& u, const K& v) const noexcept { \
    return u < span() && eto[u].has(v); \
  }
#define GRAPH_HAS_FROM(K, V, E, x, u, v, ve, ee) \
  inline bool hasVertex(const K& u)           const noexcept { return x.ve; } \
  inline bool hasEdge(const K& u, const K& v) const noexcept { return x.ee; }
#endif


#ifndef GRAPH_DEGREES
#define GRAPH_XDEGREE(K, V, E, name, eto) \
  inline K name(const K& u) const noexcept { \
    return u < span()? eto[u].size() : 0; \
  }
#define GRAPH_INDEGREE_SEARCH(K, V, E, eto) \
  inline K inDegree(const K& v) const noexcept { \
    auto fedge = [&](const K& u) { return eto[u].has(v); }; \
    return countIf(rangeIterable(span()), fedge); \
  }
#define GRAPH_DEGREE(K, V, E, eto) \
  GRAPH_XDEGREE(K, V, E, degree, eto)
#define GRAPH_INDEGREE(K, V, E, efrom) \
  GRAPH_XDEGREE(K, V, E, inDegree, efrom)

#define GRAPH_DEGREES(K, V, E, eto, efrom) \
  GRAPH_DEGREE(K, V, E, eto) \
  GRAPH_INDEGREE(K, V, E, efrom)
#define GRAPH_DEGREES_SEARCH(K, V, E, eto) \
  GRAPH_DEGREE(K, V, E, eto) \
  GRAPH_INDEGREE_SEARCH(K, V, E, eto)

#define GRAPH_DEGREES_FROM(K, V, E, x, u, v, de, ie) \
  inline K degree(const K& u)   const noexcept { return x.de; } \
  inline K inDegree(const K& v) const noexcept { return x.ie; }
#endif


#ifndef GRAPH_VALUES
#define GRAPH_VERTEX_VALUE(K, V, E, vvalues) \
  inline V vertexValue(const K& u) const noexcept { \
    return u < span()? vvalues[u] : V(); \
  }
#define GRAPH_EDGE_VALUE(K, V, E, eto) \
  inline E edgeValue(const K& u, const K& v) const noexcept { \
    return u < span()? eto[u].get(v) : E(); \
  }
#define GRAPH_VALUES(K, V, E, vvalues, eto) \
  GRAPH_VERTEX_VALUE(K, V, E, vvalues) \
  GRAPH_EDGE_VALUE(K, V, E, eto)

#define GRAPH_SET_VERTEX_VALUE(K, V, E, vvalues) \
  inline bool setVertexValue(const K& u, const V& d) noexcept { \
    if (!hasVertex(u)) return false; \
    vvalues[u] = d; \
    return true; \
  }
#define GRAPH_SET_EDGE_VALUE_X(K, V, E, ee) \
  inline bool setEdgeValue(const K& u, const K& v, const E& d) noexcept { \
    if (!hasVertex(u) || !hasVertex(v)) return false; \
    return ee; \
  }
#define GRAPH_SET_EDGE_VALUE(K, V, E, eto, efrom) \
  GRAPH_SET_EDGE_VALUE_X(K, V, E, eto[u].set(v, d) && efrom[v].set(u, d))
#define GRAPH_SET_EDGE_VALUE_SEARCH(K, V, E, eto) \
  GRAPH_SET_EDGE_VALUE_X(K, V, E, eto[u].set(v, d))
#define GRAPH_SET_VALUES(K, V, E, vvalues, eto, efrom) \
  GRAPH_SET_VERTEX_VALUE(K, V, E, vvalues) \
  GRAPH_SET_EDGE_VALUE(K, V, E, eto, efrom)
#define GRAPH_SET_VALUES_SEARCH(K, V, E, vvalues, eto) \
  GRAPH_SET_VERTEX_VALUE(K, V, E, vvalues) \
  GRAPH_SET_EDGE_VALUE_SEARCH(K, V, E, eto)

#define GRAPH_VALUES_FROM(K, V, E, x, u, v, ve, ee) \
  inline V vertexValue(const K& u)           const noexcept { return x.ve; } \
  inline E edgeValue(const K& u, const K& v) const noexcept { return x.ee; }
#define GRAPH_SET_VALUES_FROM(K, V, E, x, u, v, d, ve, ee) \
  inline bool setVertexValue(const K& u, const V& d)           noexcept { return x.ve; } \
  inline bool setEdgeValue(const K& u, const K& v, const E& d) noexcept { return x.ee; }
#endif


#ifndef GRAPH_BASE
#define GRAPH_BASE_X(K, V, E, be) \
  inline auto& base() noexcept { return be; } \
  inline const auto& cbase() const noexcept { return be; } \
  inline const auto& base()  const noexcept { return cbase(); }

#define GRAPH_BASE(K, V, E) \
  GRAPH_BASE_X(K, V, E, *this)
#define GRAPH_BASE_FROM(K, V, E, x) \
  GRAPH_BASE_X(K, V, E, x)
#endif


#ifndef GRAPH_CORRECT
#define GRAPH_CORRECT(K, V, E, M, unq, buf, u, e0, e1) \
  inline bool correct(bool unq=false) { \
    bool a = false; M = 0; \
    vector<pair<K, E>> buf; \
    cforEachVertexKey([&](const K& u) { \
      a |= e0; \
      a |= e1; \
      M += degree(u); \
    }); \
    return a; \
  }
#endif


#ifndef GRAPH_RESIZE
#define GRAPH_RESIZE_X(K, V, E, n, vexists, vvalues, eto, extra) \
  inline bool resize(size_t n) { \
    vexists.resize(n); \
    vvalues.resize(n); \
    eto.resize(n); \
    extra; \
    return true; \
  }
#define GRAPH_RESIZE(K, V, E, vexists, vvalues, eto, efrom) \
  GRAPH_RESIZE_X(K, V, E, n, vexists, vvalues, eto, efrom.resize(n))
#define GRAPH_RESIZE_SEARCH(K, V, E, vexists, vvalues, eto) \
  GRAPH_RESIZE_X(K, V, E, n, vexists, vvalues, eto,)
#endif


#ifndef GRAPH_CLEAR
#define GRAPH_CLEAR_X(K, V, E, N, M, vexists, vvalues, eto, extra) \
  inline bool clear() noexcept { \
    if (empty()) return false; \
    N = 0; M = 0; \
    vexists.clear(); \
    vvalues.clear(); \
    eto.clear(); \
    extra; \
    return true; \
  }
#define GRAPH_CLEAR(K, V, E, N, M, vexists, vvalues, eto, efrom) \
  GRAPH_CLEAR_X(K, V, E, N, M, vexists, vvalues, eto, efrom.clear())
#define GRAPH_CLEAR_SEARCH(K, V, E, N, M, vexists, vvalues, eto) \
  GRAPH_CLEAR_X(K, V, E, N, M, vexists, vvalues, eto,)
#endif


#ifndef GRAPH_ADD_VERTEX
#define GRAPH_ADD_VERTEX(K, V, E, N, vexists, vvalues) \
  inline bool addVertex(const K& u, const V& d=V()) { \
    if (hasVertex(u)) return false; \
    if (u >= span()) resize(u+1); \
    vexists[u] = true; \
    vvalues[u] = d; \
    ++N; \
    return true; \
  }
#endif


#ifndef GRAPH_ADD_EDGE
#define GRAPH_ADD_EDGE_X(K, V, E, u, v, d, M, ee) \
  inline bool addEdge(const K& u, const K& v, const E& d=E()) { \
    addVertex(u); addVertex(v); \
    if (ee) return false; \
    ++M; \
    return true; \
  }
#define GRAPH_ADD_EDGE(K, V, E, M, eto, efrom) \
  GRAPH_ADD_EDGE_X(K, V, E, u, v, d, M, !eto[u].add(v, d) || !efrom[v].add(u, d))
#define GRAPH_ADD_EDGE_SEARCH(K, V, E, M, eto) \
  GRAPH_ADD_EDGE_X(K, V, E, u, v, d, M, !eto[u].add(v, d))
#endif


#ifndef GRAPH_REMOVE_EDGE
#define GRAPH_REMOVE_EDGE_X(K, V, E, u, v, M, ee) \
  inline bool removeEdge(const K& u, const K& v) { \
    if (!hasVertex(u) || !hasVertex(v)) return false; \
    if (ee) return false; \
    --M; \
    return true; \
  }
#define GRAPH_REMOVE_EDGE(K, V, E, M, eto, efrom) \
  GRAPH_REMOVE_EDGE_X(K, V, E, u, v, M, !eto[u].remove(v) || !efrom[v].remove(u))
#define GRAPH_REMOVE_EDGE_SEARCH(K, V, E, M, eto) \
  GRAPH_REMOVE_EDGE_X(K, V, E, u, v, M, !eto[u].remove(v))
#endif


#ifndef GRAPH_REMOVE_EDGES
#define GRAPH_REMOVE_EDGES(K, V, E, M, eto, efrom) \
  inline bool removeEdges(const K& u) { \
    if (!hasVertex(u)) return false; \
    eto[u].forEachKey([&](const K& v) { efrom[v].remove(u); --M; }); \
    eto[u].clear(); \
    return true; \
  }
#define GRAPH_REMOVE_EDGES_SEARCH(K, V, E, M, eto) \
  inline bool removeEdges(const K& u) { \
    if (!hasVertex(u)) return false; \
    M -= eto[u].size(); \
    eto[u].clear(); \
    return true; \
  }
#endif


#ifndef GRAPH_REMOVE_INEDGES
#define GRAPH_REMOVE_INEDGES(K, V, E, M, eto, efrom) \
  inline bool removeInEdges(const K& v) { \
    if (!hasVertex(v)) return false; \
    efrom[v].forEachKey([&](const K& u) { eto[u].remove(v); --M; }); \
    efrom[v].clear(); \
    return true; \
  }
#define GRAPH_REMOVE_INEDGES_SEARCH(K, V, E, M, eto) \
  inline bool removeInEdges(const K& v) { \
    if (!hasVertex(v)) return false; \
    for (K u = K(); u < span(); ++u) \
      if (eto[u].remove(v)) --M; \
    return true; \
  }
#endif


#ifndef GRAPH_REMOVE_VERTEX
#define GRAPH_REMOVE_VERTEX(K, V, E, N, vexists, vvalues) \
  inline bool removeVertex(const K& u) { \
    if (!hasVertex(u)) return false; \
    removeEdges(u); \
    removeInEdges(u); \
    vexists[u] = false; \
    vvalues[u] = V(); \
    --N; \
    return true; \
  }
#endif


#ifndef GRAPH_CORRECT_FROM
#define GRAPH_CORRECT_FROM(K, V, E, x) \
  inline bool correct(bool unq=false) noexcept { return x.correct(unq); }
#define GRAPH_CLEAR_FROM(K, V, E, x) \
  inline bool clear() noexcept { return x.clear(); }
#define GRAPH_RESIZE_FROM(K, V, E, x) \
  inline bool resize() { return x.resize(); }
#define GRAPH_ADD_FROM(K, V, E, x, u, v, d, ve, ee) \
  inline bool addVertex(const K& u, const V& d=V())           { return x.ve; } \
  inline bool addEdge(const K& u, const K& v, const E& d=E()) { return x.ee; }
#define GRAPH_REMOVE_FROM(K, V, E, x, u, v, ve, ee) \
  inline bool removeVertex(const K& u)           { return x.ve; } \
  inline bool removeEdge(const K& u, const K& v) { return x.ee; }
#define GRAPH_REMOVE_EDGES_FROM(K, V, E, x, u, v, ee, ie) \
  inline bool removeEdges(const K& u)            { return x.ee; } \
  inline bool removeInEdges(const K& v)          { return x.ie; }
#endif


#ifndef GRAPH_WRITE
#define GRAPH_WRITE(K, V, E, Bitset, Graph) \
  template <class K, class V, class E, tclass2 Bitset> \
  inline void write(ostream& a, const Graph<K, V, E, Bitset>& x, bool det=false) { writeGraph(a, x, det); } \
  template <class K, class V, class E, tclass2 Bitset> \
  inline ostream& operator<<(ostream& a, const Graph<K, V, E, Bitset>& x) { write(a, x); return a; }
#define GRAPH_WRITE_VIEW(G, Graph) \
  template <class G> \
  inline void write(ostream& a, const Graph<G>& x, bool det=false) { writeGraph(a, x, det); } \
  template <class G> \
  inline ostream& operator<<(ostream& a, const Graph<G>& x) { write(a, x); return a; }
#endif




// DI-GRAPH
// --------
// Directed graph that memorizes in- and out-edges for each vertex.

template <class K=int, class V=NONE, class E=NONE, tclass2 Bitset=ROrderedBitset>
class DiGraph {
  // Data.
  protected:
  size_t N = 0, M = 0;
  vector<bool> vexists;
  vector<V>    vvalues;
  vector<Bitset<K, E>> eto;
  vector<Bitset<K, E>> efrom;
  Bitset<K, E> enone;

  // Types.
  public:
  GRAPH_TYPES(K, V, E)


  // Property operations.
  public:
  GRAPH_SIZES(K, V, E, N, M, vexists)
  GRAPH_DIRECTEDNESS(K, V, E, true)


  // Scan operations.
  public:
  GRAPH_CVERTICES(K, V, E, vexists, vvalues)
  GRAPH_CEDGES(K, V, E, eto, enone)
  GRAPH_CINEDGES(K, V, E, efrom, enone)
  GRAPH_VERTICES(K, V, E)
  GRAPH_EDGES(K, V, E)
  GRAPH_INEDGES(K, V, E)

  public:
  GRAPH_CFOREACH_VERTEX(K, V, E, vexists, vvalues)
  GRAPH_CFOREACH_EDGE(K, V, E, eto)
  GRAPH_CFOREACH_INEDGE(K, V, E, efrom)
  GRAPH_FOREACH_VERTEX(K, V, E)
  GRAPH_FOREACH_EDGE(K, V, E)
  GRAPH_FOREACH_INEDGE(K, V, E)


  // Access operations.
  public:
  GRAPH_BASE(K, V, E)
  GRAPH_HAS(K, V, E, vexists, eto)
  GRAPH_DEGREES(K, V, E, eto, efrom)
  GRAPH_VALUES(K, V, E, vvalues, eto)
  GRAPH_SET_VALUES(K, V, E, vvalues, eto, efrom)


  // Update operations.
  public:
  GRAPH_CORRECT(K, V, E, M, unq, buf, u, eto[u].correct(unq, buf), efrom[u].correct(unq, buf))
  GRAPH_CLEAR(K, V, E, N, M, vexists, vvalues, eto, efrom)
  GRAPH_RESIZE(K, V, E, vexists, vvalues, eto, efrom)
  GRAPH_ADD_VERTEX(K, V, E, N, vexists, vvalues)
  GRAPH_ADD_EDGE(K, V, E, M, eto, efrom)
  GRAPH_REMOVE_EDGE(K, V, E, M, eto, efrom)
  GRAPH_REMOVE_EDGES(K, V, E, M, eto, efrom)
  GRAPH_REMOVE_INEDGES(K, V, E, M, eto, efrom)
  GRAPH_REMOVE_VERTEX(K, V, E, N, vexists, vvalues)
};

template <class K=int, class V=NONE, class E=NONE>
using UnorderedDiGraph = DiGraph<K, V, E, UnorderedBitset>;
template <class K=int, class V=NONE, class E=NONE>
using OrderedDiGraph   = DiGraph<K, V, E, OrderedBitset>;
template <class K=int, class V=NONE, class E=NONE>
using POrderedDiGraph  = DiGraph<K, V, E, POrderedBitset>;
template <class K=int, class V=NONE, class E=NONE>
using ROrderedDiGraph  = DiGraph<K, V, E, ROrderedBitset>;




// OUT-DI-GRAPH
// ------------
// Directed graph that memorizes only out-edges for each vertex.

template <class K=int, class V=NONE, class E=NONE, tclass2 Bitset=ROrderedBitset>
class OutDiGraph {
  // Data.
  protected:
  size_t N = 0, M = 0;
  vector<bool> vexists;
  vector<V>    vvalues;
  vector<Bitset<K, E>> eto;
  Bitset<K, E> enone;

  // Types.
  public:
  GRAPH_TYPES(K, V, E)


  // Property operations.
  public:
  GRAPH_SIZES(K, V, E, N, M, vexists)
  GRAPH_DIRECTEDNESS(K, V, E, true)


  // Scan operations.
  public:
  GRAPH_CVERTICES(K, V, E, vexists, vvalues)
  GRAPH_CEDGES(K, V, E, eto, enone)
  GRAPH_CINEDGES_SEARCH(K, V, E, eto)
  GRAPH_VERTICES(K, V, E)
  GRAPH_EDGES(K, V, E)
  GRAPH_INEDGES(K, V, E)

  public:
  GRAPH_CFOREACH_VERTEX(K, V, E, vexists, vvalues)
  GRAPH_CFOREACH_EDGE(K, V, E, eto)
  GRAPH_CFOREACH_INEDGE_SEARCH(K, V, E, eto)
  GRAPH_FOREACH_VERTEX(K, V, E)
  GRAPH_FOREACH_EDGE(K, V, E)
  GRAPH_FOREACH_INEDGE(K, V, E)


  // Access operations.
  public:
  GRAPH_BASE(K, V, E)
  GRAPH_HAS(K, V, E, vexists, eto)
  GRAPH_DEGREES_SEARCH(K, V, E, eto)
  GRAPH_VALUES(K, V, E, vvalues, eto)
  GRAPH_SET_VALUES_SEARCH(K, V, E, vvalues, eto)


  // Update operations.
  public:
  GRAPH_CORRECT(K, V, E, M, unq, buf, u, eto[u].correct(unq, buf), false)
  GRAPH_CLEAR_SEARCH(K, V, E, N, M, vexists, vvalues, eto)
  GRAPH_RESIZE_SEARCH(K, V, E, vexists, vvalues, eto)
  GRAPH_ADD_VERTEX(K, V, E, N, vexists, vvalues)
  GRAPH_ADD_EDGE_SEARCH(K, V, E, M, eto)
  GRAPH_REMOVE_EDGE_SEARCH(K, V, E, M, eto)
  GRAPH_REMOVE_EDGES_SEARCH(K, V, E, M, eto)
  GRAPH_REMOVE_INEDGES_SEARCH(K, V, E, M, eto)
  GRAPH_REMOVE_VERTEX(K, V, E, N, vexists, vvalues)
};

template <class K=int, class V=NONE, class E=NONE>
using UnorderedOutDiGraph = OutDiGraph<K, V, E, UnorderedBitset>;
template <class K=int, class V=NONE, class E=NONE>
using OrderedOutDiGraph   = OutDiGraph<K, V, E, OrderedBitset>;
template <class K=int, class V=NONE, class E=NONE>
using POrderedOutDiGraph  = OutDiGraph<K, V, E, POrderedBitset>;
template <class K=int, class V=NONE, class E=NONE>
using ROrderedOutDiGraph  = OutDiGraph<K, V, E, ROrderedBitset>;




// GRAPH
// -----
// Undirected graph.

template <class K=int, class V=NONE, class E=NONE, tclass2 Bitset=ROrderedBitset>
class Graph : public OutDiGraph<K, V, E, Bitset> {
  using G = OutDiGraph<K, V, E, Bitset>;


  // Property operations.
  public:
  inline size_t size() const noexcept { return G::size()/2; }
  GRAPH_DIRECTEDNESS(K, V, E, false)


  // Scan operations.
  public:
  GRAPH_CINEDGES_FROM(K, V, E, (*this), edge)
  GRAPH_INEDGES(K, V, E)

  public:
  GRAPH_CFOREACH_INEDGE_FROM(K, V, E, (*this), Edge)
  GRAPH_FOREACH_INEDGE(K, V, E)


  // Access operations.
  public:
  GRAPH_BASE(K, V, E)
  inline size_t inDegree(const K& v) const noexcept { return degree(v); }
  inline bool setEdgeValue(const K& u, const K& v, const E& d) noexcept {
    return G::setEdgeValue(u, v, d) && G::setEdgeValue(v, u, d);
  }


  // Update operations.
  public:
  inline bool addEdge(const K& u, const K& v, const E& d=E()) {
    return G::addEdge(u, v, d) && G::addEdge(v, u, d);
  }
  inline bool removeEdge(const K& u, const K& v) {
    return G::removeEdge(u, v) && G::removeEdge(v, u);
  }
  inline bool removeEdges(const K& u) {
    forEachEdgeKey(u, [&](const K& v) { G::removeEdge(v, u); });
    return G::removeEdges(u);
  }
  inline bool removeInEdges(const K& v) {
    return removeEdges(v);
  }
};

template <class K=int, class V=NONE, class E=NONE>
using UnorderedGraph = Graph<K, V, E, UnorderedBitset>;
template <class K=int, class V=NONE, class E=NONE>
using OrderedGraph   = Graph<K, V, E, OrderedBitset>;
template <class K=int, class V=NONE, class E=NONE>
using POrderedGraph  = Graph<K, V, E, POrderedBitset>;
template <class K=int, class V=NONE, class E=NONE>
using ROrderedGraph  = Graph<K, V, E, ROrderedBitset>;




// GRAPH-VIEW
// ----------

template <class G>
class GraphView {
  // Data.
  protected:
  G& x;

  // Types.
  private:
  GRAPH_SHORT_TYPES_FROM(G)
  public:
  GRAPH_TYPES(K, V, E)


  // Property operations.
  public:
  GRAPH_SIZES_FROM(K, V, E, x)
  GRAPH_DIRECTEDNESS_FROM(K, V, E, x)


  // Scan operations.
  public:
  GRAPH_CVERTICES_FROM(K, V, E, x)
  GRAPH_CEDGES_FROM(K, V, E, x, edge)
  GRAPH_CINEDGES_FROM(K, V, E, x, inEdge)
  GRAPH_VERTICES(K, V, E)
  GRAPH_EDGES(K, V, E)
  GRAPH_INEDGES(K, V, E)

  public:
  GRAPH_CFOREACH_VERTEX_FROM(K, V, E, x)
  GRAPH_CFOREACH_EDGE_FROM(K, V, E, x, Edge)
  GRAPH_CFOREACH_INEDGE_FROM(K, V, E, x, InEdge)
  GRAPH_FOREACH_VERTEX(K, V, E)
  GRAPH_FOREACH_EDGE(K, V, E)
  GRAPH_FOREACH_INEDGE(K, V, E)


  // Access operations.
  public:
  GRAPH_BASE_FROM(K, V, E, x)
  GRAPH_HAS_FROM(K, V, E, x, u, v, hasVertex(u), hasEdge(u, v))
  GRAPH_DEGREES_FROM(K, V, E, x, u, v, degree(u), inDegree(v))
  GRAPH_VALUES_FROM(K, V, E, x, u, v, vertexValue(u), edgeValue(u, v))
  GRAPH_SET_VALUES_FROM(K, V, E, x, u, v, d, setVertexValue(u, d), setEdgeValue(u, v, d))


  // Update operations.
  public:
  GRAPH_CORRECT_FROM(K, V, E, x)
  GRAPH_CLEAR_FROM(K, V, E, x)
  GRAPH_RESIZE_FROM(K, V, E, x)
  GRAPH_ADD_FROM(K, V, E, x, u, v, d, addVertex(u, d), addEdge(u, v, d))
  GRAPH_REMOVE_FROM(K, V, E, x, u, v, removeVertex(u), removeEdge(u, v))
  GRAPH_REMOVE_EDGES_FROM(K, V, E, x, u, v, removeEdges(u), removeInEdges(v))


  // Lifetime operations.
  public:
  GraphView(G& x) : x(x) {}
};




// TRANSPOSED-GRAPH-VIEW
// ---------------------

template <class G>
class TransposedGraphView {
  // Data.
  protected:
  G& x;

  // Types.
  private:
  GRAPH_SHORT_TYPES_FROM(G)
  public:
  GRAPH_TYPES(K, V, E)


  // Property operations.
  public:
  GRAPH_SIZES_FROM(K, V, E, x)
  GRAPH_DIRECTEDNESS_FROM(K, V, E, x)


  // Scan operations.
  public:
  GRAPH_CVERTICES_FROM(K, V, E, x)
  GRAPH_CEDGES_FROM(K, V, E, x, inEdge)
  GRAPH_CINEDGES_FROM(K, V, E, x, edge)
  GRAPH_VERTICES(K, V, E)
  GRAPH_EDGES(K, V, E)
  GRAPH_INEDGES(K, V, E)

  public:
  GRAPH_CFOREACH_VERTEX_FROM(K, V, E, x)
  GRAPH_CFOREACH_EDGE_FROM(K, V, E, x, InEdge)
  GRAPH_CFOREACH_INEDGE_FROM(K, V, E, x, Edge)
  GRAPH_FOREACH_VERTEX(K, V, E)
  GRAPH_FOREACH_EDGE(K, V, E)
  GRAPH_FOREACH_INEDGE(K, V, E)


  // Access operations.
  public:
  GRAPH_BASE_FROM(K, V, E, x)
  GRAPH_HAS_FROM(K, V, E, x, u, v, hasVertex(u), hasEdge(v, u))
  GRAPH_DEGREES_FROM(K, V, E, x, u, v, inDegree(u), degree(v))
  GRAPH_VALUES_FROM(K, V, E, x, u, v, vertexValue(u), edgeValue(v, u))
  GRAPH_SET_VALUES_FROM(K, V, E, x, u, v, d, setVertexValue(u, d), setEdgeValue(v, u, d))


  // Update operations.
  public:
  GRAPH_CORRECT_FROM(K, V, E, x)
  GRAPH_CLEAR_FROM(K, V, E, x)
  GRAPH_RESIZE_FROM(K, V, E, x)
  GRAPH_ADD_FROM(K, V, E, x, u, v, d, addVertex(u, d), addEdge(v, u, d))
  GRAPH_REMOVE_FROM(K, V, E, x, u, v, removeVertex(u), removeEdge(v, u))
  GRAPH_REMOVE_EDGES_FROM(K, V, E, x, u, v, removeInEdges(u), removeEdges(v))


  // Lifetime operations.
  public:
  TransposedGraphView(G& x) : x(x) {}
};




// RETYPE
// ------

template <class K, class V, class E, tclass2 B, class KA=K, class VA=V, class EA=E>
constexpr auto retype(const DiGraph<K, V, E, B>& x, KA _k=KA(), VA _v=VA(), EA _e=E()) {
  return DiGraph<KA, VA, EA, B>();
}
template <class K, class V, class E, tclass2 B, class KA=K, class VA=V, class EA=E>
constexpr auto retype(const OutDiGraph<K, V, E, B>& x, KA _k=KA(), VA _v=VA(), EA _e=E()) {
  return OutDiGraph<KA, VA, EA, B>();
}
template <class K, class V, class E, tclass2 B, class KA=K, class VA=V, class EA=E>
constexpr auto retype(const Graph<K, V, E, B>& x, KA _k=KA(), VA _v=VA(), EA _e=E()) {
  return Graph<KA, VA, EA, B>();
}




// WTITE
// -----

template <class G>
void writeGraphSizes(ostream& a, const G& x) {
  a << "order: " << x.order() << " size: " << x.size();
  a << (x.directed()? " [directed]" : " [undirected]") << " {}";
}

template <class G>
void writeGraphDetailed(ostream& a, const G& x) {
  a << "order: " << x.order() << " size: " << x.size();
  a << (x.directed()? " [directed]" : " [undirected]") << " {\n";
  x.forEachVertex([&](auto u, auto d) {
    a << u << ":" << d << " ->";
    x.forEachEdge(u, [&](auto v, auto w) {
      a << " " << v << ":" << w;
    });
    a << "\n";
  });
  a << "}";
}

template <class G>
inline void writeGraph(ostream& a, const G& x, bool detailed=false) {
  if (detailed) writeGraphDetailed(a, x);
  else writeGraphSizes(a, x);
}

GRAPH_WRITE(K, V, E, Bitset, DiGraph)
GRAPH_WRITE(K, V, E, Bitset, OutDiGraph)
GRAPH_WRITE(K, V, E, Bitset, Graph)
GRAPH_WRITE_VIEW(G, GraphView)
GRAPH_WRITE_VIEW(G, TransposedGraphView)
