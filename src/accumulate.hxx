#pragma once




// ACCUMULATE-VERTICES
// -------------------

template <class G, class FV, class T>
auto accumulateVertices(const G& x, FV fv, T init) {
  T acc = init;
  x.forEachVertex([&](auto u, auto d) { acc = fv(acc, d, u); });
  return acc;
}

template <class G, class T>
auto accumulateVertices(const G& x, T init) {
  auto fv = [&](T acc, T v, auto u) { return acc+v; };
  return accumulateVertices(x, fv, init);
}




// ACCUMULATE-EDGES
// ----------------

template <class G, class FV, class FE, class T>
auto accumulateEdges(const G& x, FV fv, FE fe, T init, T initv) {
  T acc = init;
  x.forEachVertexKey([&](auto u) {
    T accv = initv;
    x.forEachEdge(u, [&](auto v, auto w) {
      accv = fe(accv, w, u, v);
    });
    acc = fv(acc, accv, u);
  });
  return acc;
}
