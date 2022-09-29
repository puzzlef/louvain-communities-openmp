#include <random>

using std::uniform_real_distribution;




// ADD-RANDOM-EDGE
// ---------------

template <class G, class R, class K, class V, class FE>
bool addRandomEdge(const G& x, R& rnd, K span, V w, FE fe) {
  uniform_real_distribution<> dis(0.0, 1.0);
  K u = K(dis(rnd) * span);
  K v = K(dis(rnd) * span);
  fe(u, v, w);
  return true;
}

template <class G, class R, class K, class V>
bool addRandomEdge(G& a, R& rnd, K span, V w) {
  auto fe = [&](auto u, auto v, auto w) { a.addEdge(u, v, w); };
  return addRandomEdge(a, rnd, span, w, fe);
}


template <class G, class R, class K, class V, class FE>
bool addRandomEdgeByDegree(const G& x, R& rnd, K span, V w, FE fe) {
  uniform_real_distribution<> dis(0.0, 1.0);
  double deg = x.size() / x.span();
  K un = K(dis(rnd) * deg * span);
  K vn = K(dis(rnd) * deg * span);
  K u = K(), v = K(), n = K();
  x.forEachVertexKey([&](auto t) {
    if (un<0 && un > n+x.degree(t)) u = t;
    if (vn<0 && vn > n+x.degree(t)) v = t;
    if (un>0 && vn>=0) return;
    n += x.degree(t);
  });
  if (!u) u = K(un/deg);
  if (!v) v = K(vn/deg);
  fe(u, v, w);
  return true;
}

template <class G, class R, class K, class V>
bool addRandomEdgeByDegree(G& a, R& rnd, K span, V w) {
  auto fe = [&](auto u, auto v, auto w) { a.addEdge(u, v, w); };
  return addRandomEdgeByDegree(a, rnd, span, w, fe);
}




// REMOVE-RANDOM-EDGE
// ------------------

template <class G, class R, class K, class FE>
bool removeRandomEdgeFrom(const G& x, R& rnd, K u, FE fe) {
  uniform_real_distribution<> dis(0.0, 1.0);
  if (x.degree(u) == 0) return false;
  K vi = K(dis(rnd) * x.degree(u)), i = 0;
  bool removed = false;
  x.forEachEdgeKey(u, [&](auto v) {
    if (removed) return;
    if (i++ == vi) { fe(u, v); removed = true; }
  });
  return removed;
}

template <class G, class R, class K>
bool removeRandomEdgeFrom(G& a, R& rnd, K u) {
  auto fe = [&](auto u, auto v) { a.removeEdge(u, v); };
  return removeRandomEdgeFrom(a, rnd, u, fe);
}


template <class G, class R, class FE>
bool removeRandomEdge(const G& x, R& rnd, FE fe) {
  using K = typename G::key_type;
  uniform_real_distribution<> dis(0.0, 1.0);
  K u = K(dis(rnd) * x.span());
  return removeRandomEdgeFrom(x, rnd, u, fe);
}

template <class G, class R>
bool removeRandomEdge(G& a, R& rnd) {
  auto fe = [&](auto u, auto v) { a.removeEdge(u, v); };
  return removeRandomEdge(a, rnd, fe);
}


template <class G, class R, class FE>
bool removeRandomEdgeByDegree(const G& x, R& rnd, FE fe) {
  using K = typename G::key_type;
  uniform_real_distribution<> dis(0.0, 1.0);
  K v = K(dis(rnd) * x.size()), n = 0;
  bool attempted = false, removed = false;
  x.forEachVertexKey([&](auto u) {
    if (attempted) return;
    if (v > n+x.degree(u)) n += x.degree(u);
    else { removed = removeRandomEdge(x, rnd, u, fe); attempted = true; }
  });
  return removed;
}

template <class G, class R>
bool removeRandomEdgeByDegree(G& a, R& rnd) {
  auto fe = [&](auto u, auto v) { a.removeEdge(u, v); };
  return removeRandomEdgeByDegree(a, rnd, fe);
}
