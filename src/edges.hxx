#pragma once
#include <type_traits>
#include <vector>
#include "_main.hxx"

using std::remove_reference_t;
using std::vector;




// EDGE-KEYS
// ---------

template <class G, class K>
inline auto edgeKeys(const G& x, K u) {
  return copyVector(x.edgeKeys(u));
}

template <class G, class K>
inline auto edgeKey(const G& x, K u) {
  for (auto v : x.edgeKeys(u))
    return v;
  return K(-1);
}




// EDGE-VALUES
// -----------

template <class G, class K>
inline auto edgeValues(const G& x, K u) {
  return copyVector(x.edgeValues(u));
}




// EDGE-DATA
// ---------

template <class G, class J, class FM>
auto edgeData(const G& x, const J& ks, FM fm) {
  using K = typename G::key_type;
  using E = typename G::edge_value_type;
  using T = remove_reference_t<decltype(fm(K(), K(), E()))>;
  vector<T> a;
  for (auto u : ks)
    x.forEachEdge(u, [&](auto v, auto w) { a.push_back(fm(u, v, w)); });
  return a;
}
template <class G, class J>
inline auto edgeData(const G& x, const J& ks) {
  auto fm = [&](auto u, auto v, auto w) { return w; };
  return edgeData(x, ks, fm);
}
template <class G>
inline auto edgeData(const G& x) {
  return edgeData(x, x.vertexKeys());
}
