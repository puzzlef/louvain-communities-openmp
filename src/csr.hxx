#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"

using std::vector;
using std::transform;




// SOURCE-OFFSETS
// --------------

template <class G, class J, class T>
auto sourceOffsetsAs(const G& x, const J& ks, T _) {
  vector<T> a; T i = 0;
  a.reserve(x.order()+1);
  for (auto u : ks) {
    a.push_back(i);
    i += x.degree(u);
  }
  a.push_back(i);
  return a;
}
template <class G, class T>
inline auto sourceOffsetsAs(const G& x, T _) {
  return sourceOffsetsAs(x, x.vertexKeys(), _);
}

template <class G, class J>
inline auto sourceOffsets(const G& x, const J& ks) {
  return sourceOffsetsAs(x, ks, size_t());
}
template <class G>
inline auto sourceOffsets(const G& x) {
  return sourceOffsetsAs(x, size_t());
}




// DESTINATION-INDICES
// -------------------

template <class G, class J, class T>
auto destinationIndicesAs(const G& x, const J& ks, T _) {
  auto ids = valueIndicesUnorderedMap(ks); vector<T> a;
  for (auto u : ks)
    x.forEachEdgeKey(u, [&](auto v) { a.push_back(T(ids[v])); });
  return a;
}
template <class G, class T>
auto destinationIndicesAs(const G& x, T _) {
  return destinationIndicesAs(x, x.vertexKeys(), _);
}

template <class G, class J>
auto destinationIndices(const G& x, const J& ks) {
  using K = typename G::key_type;
  return destinationIndicesAs(x, ks, K());
}
template <class G>
inline auto destinationIndices(const G& x) {
  using K = typename G::key_type;
  return destinationIndicesAs(x, K());
}
