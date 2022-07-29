#pragma once
#include <type_traits>
#include <vector>
#include "_main.hxx"

using std::remove_reference_t;
using std::vector;




// VERTEX-KEYS
// -----------

template <class G>
inline auto vertexKeys(const G& x) {
  return copyVector(x.vertexKeys());
}




// VERTEX-VALUES
// -------------

template <class G>
inline auto vertexValues(const G& x) {
  return copyVector(x.vertexValues());
}




// VERTEX-DATA
// -----------

template <class G, class J, class FM>
inline auto vertexData(const G& x, const J& ks, FM fm) {
  using K = typename G::key_type;
  using V = typename G::vertex_value_type;
  using T = remove_reference_t<decltype(fm(K(), V()))>;
  vector<T> a;
  for (auto u : ks)
    a.push_back(fm(u, x.vertexValue(u)));
  return a;
}
template <class G, class J>
inline auto vertexData(const G& x, const J& ks) {
  auto fm = [&](auto u, auto d) { return d; };
  return vertexData(x, ks, fm);
}
template <class G>
inline auto vertexData(const G& x) {
  return copyVector(x.vertexValues());
}




// CREATE-CONTAINER
// ----------------

template <class G, class T>
inline auto createContainer(const G& x, const T& _) {
  return vector<T>(x.span());
}
template <class G, class T>
inline auto createCompressedContainer(const G& x, const T& _) {
  return vector<T>(x.order());
}




// DECOMPRESS-CONTAINER
// --------------------

template <class G, class T, class J>
inline void decompressContainerW(vector<T>& a, const G& x, const vector<T>& vs, const J& ks) {
  scatterValuesW(a, vs, ks);
}
template <class G, class T>
inline void decompressContainerW(vector<T>& a, const G& x, const vector<T>& vs) {
  decompressContainerW(a, x, vs, x.vertexKeys());
}

template <class G, class T, class J>
inline auto decompressContainer(const G& x, const vector<T>& vs, const J& ks) {
  auto a = createContainer(x, T());
  decompressContainerW(a, x, vs, ks);
  return a;
}
template <class G, class T>
inline auto decompressContainer(const G& x, const vector<T>& vs) {
  return decompressContainer(x, vs, x.vertexKeys());
}




// COMPRESS-CONTAINER
// ------------------

template <class G, class T, class J>
inline void compressContainerW(vector<T>& a, const G& x, const vector<T>& vs, const J& ks) {
  gatherValuesW(a, vs, ks);
}
template <class G, class T>
inline void compressContainerW(vector<T>& a, const G& x, const vector<T>& vs) {
  return compressContainerW(a, x, vs, x.vertexKeys());
}

template <class G, class T, class J>
inline auto compressContainer(const G& x, const vector<T>& vs, const J& ks) {
  auto a = createCompressedContainer(x, T());
  compressContainerW(a, x, vs, ks);
  return a;
}
template <class G, class T>
inline auto compressContainer(const G& x, const vector<T>& vs) {
  return compressContainer(x, vs, x.vertexKeys());
}




// VERTICES-EQUAL
// --------------

template <class G, class K>
inline bool verticesEqual(const G& x, K u, const G& y, K v) {
  if (x.degree(u) != y.degree(v)) return false;
  return equalValues(x.edgeKeys(u), y.edgeKeys(v));
}
template <class G, class H, class K>
inline bool verticesEqual(const G& x, const H& xt, K u, const G& y, const H& yt, K v) {
  return verticesEqual(x, u, y, u) && verticesEqual(xt, u, yt, u);
}
