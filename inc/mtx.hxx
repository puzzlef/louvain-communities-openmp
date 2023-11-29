#pragma once
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <string>
#include <istream>
#include <sstream>
#include <fstream>
#include "_main.hxx"
#include "Graph.hxx"
#include "update.hxx"

#ifdef OPENMP
#include <omp.h>
#endif

using std::tuple;
using std::string;
using std::istream;
using std::istringstream;
using std::ifstream;
using std::ofstream;
using std::move;
using std::max;
using std::getline;




// READ MTX HEADER
// ---------------
// Read header of MTX file.

/**
 * Read header of MTX file.
 * @param s input stream
 * @param symmetric is graph symmetric (updated)
 * @param rows number of rows (updated)
 * @param cols number of columns (updated)
 * @param size number of lines/edges (updated)
 */
inline void readMtxHeader(istream& s, bool& symmetric, size_t& rows, size_t& cols, size_t& size) {
  string line, h0, h1, h2, h3, h4;
  // Skip past the comments and read the graph type.
  while (true) {
    getline(s, line);
    if (line.find('%')!=0) break;
    if (line.find("%%")!=0) continue;
    istringstream sline(line);
    sline >> h0 >> h1 >> h2 >> h3 >> h4;
  }
  if (h1!="matrix" || h2!="coordinate") { symmetric = false; rows = 0; cols = 0; size = 0; return; }
  symmetric = h4=="symmetric" || h4=="skew-symmetric";
  // Read rows, cols, size.
  istringstream sline(line);
  sline >> rows >> cols >> size;
}
inline void readMtxHeader(const char* pth, bool& symmetric, size_t& rows, size_t& cols, size_t& size) {
  ifstream s(pth);
  readMtxHeader(s, symmetric, rows, cols, size);
}


/**
 * Read order of graph in MTX file.
 * @param s input stream
 * @returns number of vertices (1-N)
 */
inline size_t readMtxOrder(istream& s) {
  bool symmetric; size_t rows, cols, size;
  readMtxHeader(s, symmetric, rows, cols, size);
  return max(rows, cols);
}
inline size_t readMtxOrder(const char* pth) {
  ifstream s(pth);
  return readMtxOrder(s);
}


/**
 * Read size of graph in MTX file.
 * @param s input stream
 * @returns number of edges
 */
inline size_t readMtxSize(istream& s) {
  bool symmetric; size_t rows, cols, size;
  readMtxHeader(s, symmetric, rows, cols, size);
  return size;
}
inline size_t readMtxSize(const char* pth) {
  ifstream s(pth);
  return readMtxSize(s);
}


/**
 * Read span of graph in MTX file.
 * @param s input stream
 * @returns number of vertices + 1
 */
inline size_t readMtxSpan(istream& s) {
  return readMtxOrder(s) + 1;
}
inline size_t readMtxSpan(const char* pth) {
  ifstream s(pth);
  return readMtxSpan(s);
}




// READ MTX BODY DO
// ----------------
// Read body of MTX file.

/**
 * Read body of MTX file.
 * @param s input stream
 * @param weighted is it weighted?
 * @param symmetric is graph symmetric?
 * @param fb on body line (u, v, w)
 */
template <class FB>
inline void readMtxBodyDo(istream& s, bool weighted, bool symmetric, FB fb) {
  string line;
  while (getline(s, line)) {
    size_t u, v; double w = 1;
    istringstream sline(line);
    if (!(sline >> u >> v)) break;
    if (weighted) sline >> w;
    fb(u, v, w);
    if (symmetric) fb(v, u, w);
  }
}
template <class FB>
inline void readMtxBodyDo(const char *pth, bool weighted, bool symmetric, FB fb) {
  ifstream s(pth);
  readMtxBodyDo(s, weighted, symmetric, fb);
}


#ifdef OPENMP
/**
 * Read body of MTX file.
 * @param s input stream
 * @param weighted is it weighted?
 * @param symmetric is graph symmetric?
 * @param fb on body line (u, v, w)
 */
template <class FB>
inline void readMtxBodyDoOmp(istream& s, bool weighted, bool symmetric, FB fb) {
  const int THREADS = omp_get_max_threads();
  const int LINES   = 131072;
  const size_t CHUNK_SIZE = 1024;
  vector<string> lines(LINES);
  vector<tuple<size_t, size_t, double>> edges(LINES);
  while (true) {
    // Read several lines from the stream.
    int READ = 0;
    for (int i=0; i<LINES; ++i, ++READ)
      if (!getline(s, lines[i])) break;
    if (READ==0) break;
    // Parse lines using multiple threads.
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i=0; i<READ; ++i) {
      char *line = (char*) lines[i].c_str();
      size_t u = strtoull(line, &line, 10);
      size_t v = strtoull(line, &line, 10);
      double w = weighted? strtod(line, &line) : 0;
      edges[i] = {u, v, w? w : 1};
    }
    // Notify parsed lines.
    #pragma omp parallel
    {
      for (int i=0; i<READ; ++i) {
        const auto& [u, v, w] = edges[i];
        fb(u, v, w);
        if (symmetric) fb(v, u, w);
      }
    }
  }
}
template <class FB>
inline void readMtxBodyDoOmp(const char *pth, bool weighted, bool symmetric, FB fb) {
  ifstream s(pth);
  readMtxBodyDoOmp(s, weighted, symmetric, fb);
}
#endif




// READ MTX DO
// -----------
// Read contents of MTX file.

/**
 * Read contents of MTX file.
 * @param s input stream
 * @param weighted is it weighted?
 * @param fh on header (symmetric, rows, cols, size)
 * @param fb on body line (u, v, w)
 */
template <class FH, class FB>
inline void readMtxDo(istream& s, bool weighted, FH fh, FB fb) {
  bool symmetric; size_t rows, cols, size;
  readMtxHeader(s, symmetric, rows, cols, size);
  fh(symmetric, rows, cols, size);
  size_t n = max(rows, cols);
  if (n==0) return;
  readMtxBodyDo(s, weighted, symmetric, fb);
}
template <class FH, class FB>
inline void readMtxDo(const char *pth, bool weighted, FH fh, FB fb) {
  ifstream s(pth);
  readMtxDo(s, weighted, fh, fb);
}


#ifdef OPENMP
/**
 * Read contents of MTX file.
 * @param s input stream
 * @param weighted is it weighted?
 * @param fh on header (symmetric, rows, cols, size)
 * @param fb on body line (u, v, w)
 */
template <class FH, class FB>
inline void readMtxDoOmp(istream& s, bool weighted, FH fh, FB fb) {
  bool symmetric; size_t rows, cols, size;
  readMtxHeader(s, symmetric, rows, cols, size);
  fh(symmetric, rows, cols, size);
  size_t n = max(rows, cols);
  if (n==0) return;
  readMtxBodyDoOmp(s, weighted, symmetric, fb);
}
template <class FH, class FB>
inline void readMtxDoOmp(const char *pth, bool weighted, FH fh, FB fb) {
  ifstream s(pth);
  readMtxDoOmp(s, weighted, fh, fb);
}
#endif




// READ MTX IF
// -----------
// Read MTX file as graph if test passes.

/**
 * Read MTX file as graph if test passes.
 * @param a output graph (updated)
 * @param s input stream
 * @param weighted is it weighted?
 * @param fv include vertex? (u, d)
 * @param fe include edge? (u, v, w)
 */
template <class G, class FV, class FE>
inline void readMtxIfW(G &a, istream& s, bool weighted, FV fv, FE fe) {
  using K = typename G::key_type;
  using V = typename G::vertex_value_type;
  using E = typename G::edge_value_type;
  auto fh = [&](auto symmetric, auto rows, auto cols, auto size) { addVerticesIfU(a, K(1), K(max(rows, cols)+1), V(), fv); };
  auto fb = [&](auto u, auto v, auto w) { if (fe(K(u), K(v), K(w))) a.addEdge(K(u), K(v), E(w)); };
  readMtxDo(s, weighted, fh, fb);
  a.update();
}
template <class G, class FV, class FE>
inline void readMtxIfW(G &a, const char *pth, bool weighted, FV fv, FE fe) {
  ifstream s(pth);
  readMtxIfW(a, s, weighted, fv, fe);
}


#ifdef OPENMP
/**
 * Read MTX file as graph if test passes.
 * @param a output graph (updated)
 * @param s input stream
 * @param weighted is it weighted?
 * @param fv include vertex? (u, d)
 * @param fe include edge? (u, v, w)
 */
template <class G, class FV, class FE>
inline void readMtxIfOmpW(G &a, istream& s, bool weighted, FV fv, FE fe) {
  using K = typename G::key_type;
  using V = typename G::vertex_value_type;
  using E = typename G::edge_value_type;
  auto fh = [&](auto symmetric, auto rows, auto cols, auto size) { addVerticesIfU(a, K(1), K(max(rows, cols)+1), V(), fv); };
  auto fb = [&](auto u, auto v, auto w) { if (fe(K(u), K(v), K(w))) addEdgeOmpU(a, K(u), K(v), E(w)); };
  readMtxDoOmp(s, weighted, fh, fb);
  updateOmpU(a);
}
template <class G, class FV, class FE>
inline void readMtxIfOmpW(G &a, const char *pth, bool weighted, FV fv, FE fe) {
  ifstream s(pth);
  readMtxIfOmpW(a, s, weighted, fv, fe);
}
#endif




// READ MTX
// --------
// Read MTX file as graph.

/**
 * Read MTX file as graph.
 * @param a output graph (updated)
 * @param s input stream
 * @param weighted is it weighted?
 */
template <class G>
inline void readMtxW(G& a, istream& s, bool weighted=false) {
  auto fv = [](auto u, auto d)         { return true; };
  auto fe = [](auto u, auto v, auto w) { return true; };
  readMtxIfW(a, s, weighted, fv, fe);
}
template <class G>
inline void readMtxW(G& a, const char *pth, bool weighted=false) {
  ifstream s(pth);
  readMtxW(a, s, weighted);
}


#ifdef OPENMP
/**
 * Read MTX file as graph.
 * @param a output graph (updated)
 * @param s input stream
 * @param weighted is it weighted?
 */
template <class G>
inline void readMtxOmpW(G& a, istream& s, bool weighted=false) {
  auto fv = [](auto u, auto d)         { return true; };
  auto fe = [](auto u, auto v, auto w) { return true; };
  readMtxIfOmpW(a, s, weighted, fv, fe);
}
template <class G>
inline void readMtxOmpW(G& a, const char *pth, bool weighted=false) {
  ifstream s(pth);
  readMtxOmpW(a, s, weighted);
}
#endif




// READ EDGELIST HEADER
// --------------------
// Read header of EDGELIST file.

/**
 * Read header of EDGELIST file.
 * @param s input stream
 * @param nodes number of nodes (updated)
 * @param edges number of edges (updated)
 * @returns following line
 */
inline string readEdgelistHeader(istream& s, size_t& nodes, size_t& edges) {
  string line, h0, h1, h2;
  nodes = size_t(-1);
  edges = size_t(-1);
  while (getline(s, line)) {
    if (line.find('#')!=0) break;
    if (line.find("Nodes:")==size_t(-1)) continue;
    if (line.find("Edges:")==size_t(-1)) continue;
    istringstream sline(line);
    sline >> h0 >> h1 >> nodes >> h2 >> edges;
  }
  return line;
}
inline string readEdgelistHeader(const char* pth, size_t& nodes, size_t& edges) {
  ifstream s(pth);
  return readEdgelistHeader(s, nodes, edges);
}


/**
 * Read order of graph in EDGELIST file.
 * @param s input stream
 * @returns number of vertices (0-N?)
 */
inline size_t readEdgelistOrder(istream& s) {
  size_t nodes, edges;
  readEdgelistHeader(s, nodes, edges);
  return nodes;
}
inline size_t readEdgelistOrder(const char* pth) {
  ifstream s(pth);
  return readEdgelistOrder(s);
}


/**
 * Read size of graph in EDGELIST file.
 * @param s input stream
 * @returns number of edges
 */
inline size_t readEdgelistSize(istream& s) {
  size_t nodes, edges;
  readEdgelistHeader(s, nodes, edges);
  return edges;
}
inline size_t readEdgelistSize(const char* pth) {
  ifstream s(pth);
  return readEdgelistSize(s);
}


/**
 * Read span of graph in EDGELIST file.
 * @param s input stream
 * @returns number of vertices + 1
 */
inline size_t readEdgelistSpan(istream& s) {
  return readEdgelistOrder(s) + 1;
}
inline size_t readEdgelistSpan(const char* pth) {
  ifstream s(pth);
  return readEdgelistSpan(s);
}




// READ EDGELIST DO
// ----------------
// Read contents of EDGELIST file.

/**
 * Read contents of EDGELIST file.
 * @param s input stream
 * @param weighted is it weighted?
 * @param symmetric is it symmetric?
 * @param fh on header (nodes, edges)
 * @param fb on body line (u, v, w)
 */
template <class FH, class FB>
inline void readEdgelistDo(istream& s, bool weighted, bool symmetric, FH fh, FB fb) {
  size_t nodes, edges;
  string line = readEdgelistHeader(s, nodes, edges);
  fh(nodes, edges);
  if (nodes==0) return;
  istringstream sline(line);
  readMtxBodyDo(sline, weighted, symmetric, fb);
  readMtxBodyDo(s,     weighted, symmetric, fb);
}
template <class FH, class FB>
inline void readEdgelistDo(const char *pth, bool weighted, bool symmetric, FH fh, FB fb) {
  ifstream s(pth);
  readEdgelistDo(s, weighted, symmetric, fh, fb);
}


#ifdef OPENMP
/**
 * Read contents of EDGELIST file.
 * @param s input stream
 * @param weighted is it weighted?
 * @param symmetric is it symmetric?
 * @param fh on header (nodes, edges)
 * @param fb on body line parallel (u, v, w)
 */
template <class FH, class FB>
inline void readEdgelistDoOmp(istream& s, bool weighted, bool symmetric, FH fh, FB fb) {
  size_t nodes, edges;
  string line = readEdgelistHeader(s, nodes, edges);
  fh(nodes, edges);
  if (nodes==0) return;
  istringstream sline(line);
  #pragma omp parallel for schedule(static)
  for (int i=0; i<1; ++i)
    readMtxBodyDo(sline, weighted, symmetric, fb);
  readMtxBodyDoOmp(s, weighted, symmetric, fb);
}
template <class FH, class FB>
inline void readEdgelistDoOmp(const char *pth, bool weighted, bool symmetric, FH fh, FB fb) {
  ifstream s(pth);
  readEdgelistDoOmp(s, weighted, symmetric, fh, fb);
}
#endif




// READ EDGELIFT IF
// ----------------
// Read EDGELIFT file as graph if test passes.

/**
 * Read EDGELIFT file as graph if test passes.
 * @param a output graph (updated)
 * @param s input stream
 * @param weighted is it weighted?
 * @param symmetric is it symmetric?
 * @param fv include vertex? (u, d)
 * @param fe include edge? (u, v, w)
 */
template <class G, class FV, class FE>
inline void readEdgelistIfW(G &a, istream& s, bool weighted, bool symmetric, FV fv, FE fe) {
  using K = typename G::key_type;
  using V = typename G::vertex_value_type;
  using E = typename G::edge_value_type;
  // Not sure if the vertex-id is 0-based or 1-based, accomodate both.
  // Reserve enough space for the vertices, as there is likely gaps in the vertex-ids.
  auto fh = [&](auto nodes, auto edges) { if (nodes!=size_t(-1) && nodes>a.span()) a.respan(2*nodes+1); };
  auto fb = [&](auto u, auto v, auto w) { if (fe(K(u), K(v), K(w))) a.addEdge(K(u), K(v), E(w)); };
  readEdgelistDo(s, weighted, symmetric, fh, fb);
  a.update();
}
template <class G, class FV, class FE>
inline void readEdgelistIfW(G &a, const char *pth, bool weighted, bool symmetric, FV fv, FE fe) {
  ifstream s(pth);
  readEdgelistIfW(a, s, weighted, symmetric, fv, fe);
}


#ifdef OPENMP
/**
 * Read EDGELIST file as graph if test passes.
 * @param a output graph (updated)
 * @param s input stream
 * @param weighted is it weighted?
 * @param symmetric is it symmetric?
 * @param fv include vertex? (u, d)
 * @param fe include edge? (u, v, w)
 */
template <class G, class FV, class FE>
inline void readEdgelistIfOmpW(G &a, istream& s, bool weighted, bool symmetric, FV fv, FE fe) {
  using K = typename G::key_type;
  using V = typename G::vertex_value_type;
  using E = typename G::edge_value_type;
  // Not sure if the vertex-id is 0-based or 1-based, accomodate both.
  // Reserve enough space for the vertices, as there is likely gaps in the vertex-ids.
  auto fh = [&](auto nodes, auto edges) { if (nodes!=size_t(-1) && nodes>a.span()) a.respan(2*nodes+1); };
  auto fb = [&](auto u, auto v, auto w) { if (fe(K(u), K(v), K(w))) addEdgeOmpU(a, K(u), K(v), E(w)); };
  readEdgelistDoOmp(s, weighted, symmetric, fh, fb);
  updateOmpU(a);
}
template <class G, class FV, class FE>
inline void readEdgelistIfOmpW(G &a, const char *pth, bool weighted, bool symmetric, FV fv, FE fe) {
  ifstream s(pth);
  readEdgelistIfOmpW(a, s, weighted, symmetric, fv, fe);
}
#endif




// READ EDGELIST
// -------------
// Read EDGELIST file as graph.

/**
 * Read EDGELIST file as graph.
 * @param a output graph (updated)
 * @param s input stream
 * @param weighted is it weighted?
 * @param symmetric is it symmetric?
 */
template <class G>
inline void readEdgelistW(G& a, istream& s, bool weighted=false, bool symmetric=false) {
  auto fv = [](auto u, auto d)         { return true; };
  auto fe = [](auto u, auto v, auto w) { return true; };
  readEdgelistIfW(a, s, weighted, symmetric, fv, fe);
}
template <class G>
inline void readEdgelistW(G& a, const char *pth, bool weighted=false, bool symmetric=false) {
  ifstream s(pth);
  readEdgelistW(a, s, weighted, symmetric);
}


#ifdef OPENMP
/**
 * Read EDGELIST file as graph.
 * @param a output graph (updated)
 * @param s input stream
 * @param weighted is it weighted?
 * @param symmetric is it symmetric?
 */
template <class G>
inline void readEdgelistOmpW(G& a, istream& s, bool weighted=false, bool symmetric=false) {
  auto fv = [](auto u, auto d)         { return true; };
  auto fe = [](auto u, auto v, auto w) { return true; };
  readEdgelistIfOmpW(a, s, weighted, symmetric, fv, fe);
}
template <class G>
inline void readEdgelistOmpW(G& a, const char *pth, bool weighted=false, bool symmetric=false) {
  ifstream s(pth);
  readEdgelistOmpW(a, s, weighted, symmetric);
}
#endif
