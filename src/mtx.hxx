#pragma once
#include <string>
#include <istream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "_main.hxx"
#include "Graph.hxx"

using std::string;
using std::istream;
using std::stringstream;
using std::ofstream;
using std::getline;
using std::max;




// READ-MTX
// --------

template <class FV, class FE>
void readMtxDo(istream& s, FV fv, FE fe) {
  string ln, h0, h1, h2, h3, h4;

  // read header
  while (1) {
    getline(s, ln);
    if (ln.find('%')!=0) break;
    if (ln.find("%%")!=0) continue;
    stringstream ls(ln);
    ls >> h0 >> h1 >> h2 >> h3 >> h4;
  }
  if (h1!="matrix" || h2!="coordinate") return;
  bool sym = h4=="symmetric" || h4=="skew-symmetric";

  // read rows, cols, size
  size_t r, c, sz;
  stringstream ls(ln);
  ls >> r >> c >> sz;
  size_t n = max(r, c);
  for (size_t u=1; u<=n; u++)
    fv(u);  // a.addVertex(u);

  // read edges (from, to)
  while (getline(s, ln)) {
    size_t u, v;
    double w = 1;
    ls = stringstream(ln);
    if (!(ls >> u >> v)) break;
    ls >> w;
    fe(u, v, w);  // a.addEdge(u, v);
    if (sym) fe(v, u, w);  // a.addEdge(v, u);
  }
}


template <class G>
void readMtxW(G& a, istream& s, bool unq=false) {
  using K = typename G::key_type;
  using E = typename G::edge_value_type;
  auto fv = [&](auto u) { a.addVertex(K(u)); };
  auto fe = [&](auto u, auto v, auto w) { a.addEdge(K(u), K(v), E(w)); };
  readMtxDo(s, fv, fe);
  a.correct(unq);
}
template <class G>
void readMtxW(G& a, const char *pth, bool unq=false) {
  string buf = readFile(pth);
  stringstream s(buf);
  return readMtxW(a, s, unq);
}


#define READ_MTX_RETURN(R, unq) \
  inline auto readMtx##R(istream& s) { \
    R<> a; readMtxW(a, s, unq); \
    return a; \
  } \
  inline auto readMtx##R(const char *pth) { \
    R<> a; readMtxW(a, pth, unq); \
    return a; \
  }

READ_MTX_RETURN(DiGraph, true)
READ_MTX_RETURN(OutDiGraph, true)
READ_MTX_RETURN(Graph, false)




// WRITE-MTX
// ---------

template <class G>
void writeMtx(ostream& a, const G& x) {
  a << "%%MatrixMarket matrix coordinate real asymmetric\n";
  a << x.order() << " " << x.order() << " " << x.size() << "\n";
  x.forEachVertexKey([&](auto u) {
    x.forEachEdge(u, [&](auto v, auto w) {
      a << u << " " << v << " " << w << "\n";
    });
  });
}
template <class G>
inline void writeMtx(string pth, const G& x) {
  string s0; stringstream s(s0);
  writeMtx(s, x);
  ofstream f(pth);
  f << s.rdbuf();
  f.close();
}
