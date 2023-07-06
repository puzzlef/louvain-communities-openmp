#include <cstdint>
#include <cstdio>
#include <utility>
#include <random>
#include <iterator>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "src/main.hxx"

#ifdef OPENMP
#include <omp.h>
#endif

using namespace std;




// Fixed config
#ifndef TYPE
#define TYPE float
#endif
#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif
#ifndef REPEAT_METHOD
#define REPEAT_METHOD 5
#endif




// HELPERS
// -------

/**
 * Calculate the modularity of obtained communities.
 * @param x original graph
 * @param a obtained louvain result
 * @param M total edge weight of the graph
 * @returns modularity
 */
template <class G, class K>
inline double getModularity(const G& x, const LouvainResult<K>& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityByOmp(x, fc, M, 1.0);
}


/**
 * Read (ground-truth) community vertices from a file.
 * @param comv community vertices (output)
 * @param pth path to the file
 */
template <class K>
inline void readCommunityVerticesOmpW(vector2d<K>& comv, const char *pth) {
  ifstream s(pth);
  readLinesOmpDo(s, [&](auto c, const auto& line) {
    const char *str = line.c_str();
    while (1) {
      const char *ptr = str;
      size_t u = strtoull(str, (char**) &ptr, 10);
      if (ptr==str) break;
      str = ptr;
      comv[c].push_back(K(u));
    }
  });
  // Keep vertices in each community sorted.
  #pragma omp parallel for schedule(dynamic, 2048)
  for (size_t c=0; c<comv.size(); ++c)
    sort(comv[c].begin(), comv[c].end());
}


/**
 * Get community memberships of each vertex from (ground-truth) community vertices.
 * @param x original graph
 * @param comv community vertices
 * @returns community memberships of each vertex
 */
template <class G, class K>
inline vector2d<K> communityMembershipsOmp(const G& x, const vector2d<K>& comv) {
  size_t S = x.span();
  vector2d<K> a(S);
  #pragma omp parallel
  {
    for (size_t c=0; c<comv.size(); ++c) {
      for (auto u : comv[c])
        if (belongsOmp(u)) a[u].push_back(K(c));
    }
  }
  // // Each vertex must belong to at least its own community.
  // #pragma omp parallel for schedule(dynamic, 2048)
  // for (K u=0; u<S; ++u) {
  //   if (!x.hasVertex(u)) continue;
  //   if (a[u].empty()) a[u].push_back(K(u));
  // }
  return a;
}


/**
 * Obtain community vertices from community membership of each vertex.
 * @param x original graph
 * @param vcom community membership of each vertex
 * @returns community vertices
 */
template <class G, class K>
inline vector2d<K> communityVerticesOmp(const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  vector2d<K> a(S);
  #pragma omp parallel
  {
    for (size_t u=0; u<S; ++u) {
      if (!x.hasVertex(u)) continue;
      K c = vcom[u];
      if (belongsOmp(c)) a[c].push_back(K(u));
    }
  }
  // Keep vertices in each community sorted.
  #pragma omp parallel for schedule(dynamic, 2048)
  for (size_t c=0; c<a.size(); ++c)
    sort(a[c].begin(), a[c].end());
  return a;
}


/**
 * Obtain various properties of community vertices.
 * @param x original graph
 * @param comv community vertices
 * @returns (number of communities, minimum community size, maximum community size, average community size)
 */
template <class K>
inline auto communityVerticesPropertiesOmp(const vector2d<K>& comv) {
  size_t amin = numeric_limits<size_t>::max();
  size_t amax = 0, aavg = 0, anum = 0;
  #pragma omp parallel for schedule(static) reduction(min:amin) reduction(max:amax) reduction(+:aavg,anum)
  for (size_t c=0; c<comv.size(); ++c) {
    if (comv[c].empty()) continue;
    amin = min(amin, comv[c].size());
    amax = max(amax, comv[c].size());
    aavg += comv[c].size();
    ++anum;
  }
  if (amin==numeric_limits<size_t>::max()) amin = 0;
  return make_tuple(anum, amin, amax, anum? double(aavg) / anum : 0.0);
}


/**
 * Obtain various properties of community memberships.
 * @param x original graph
 * @param vcom community membership of each vertex
 * @returns (number of vertices, minimum community memberships per vertex, maximum community memberships per vertex, average community memberships per vertex)
 */
template <class G, class K>
inline auto communityMembershipsPropertiesOmp(const G& x, const vector2d<K>& vcom) {
  size_t S = x.span();
  size_t amin = numeric_limits<size_t>::max();
  size_t amax = 0, aavg = 0, anum = 0;
  #pragma omp parallel for schedule(static) reduction(min:amin) reduction(max:amax) reduction(+:aavg,anum)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    amin = min(amin, vcom[u].size());
    amax = max(amax, vcom[u].size());
    aavg += vcom[u].size();
    ++anum;
  }
  if (amin==numeric_limits<size_t>::max()) amin = 0;
  return make_tuple(anum, amin, amax, anum? double(aavg) / anum : 0.0);
}


/**
 * Calculate F1-scores for a given vertex wrt ground-truth communities.
 * @param buf scratch buffer (updated)
 * @param x original graph
 * @param u vertex
 * @param vcom community membership of each vertex
 * @param comv community vertices
 * @param gtvcom ground-truth community membership of each vertex
 * @param gtcomv ground-truth community vertices
 * @returns (minimum F1-score, maximum F1-score, average F1-score)
 */
template <class G, class K>
inline auto vertexF1Scores(vector<K>& buf, const G& x, K u, const vector<K>& vcom, const vector2d<K>& comv, const vector2d<K>& gtvcom, const vector2d<K>& gtcomv) {
  K c = vcom[u];
  double amin = numeric_limits<double>::max();
  double amax = 0, aavg = 0;
  // Each vertex must belong to at least its own community.
  if (gtvcom[u].empty()) {
    size_t nc  = comv[c].size();
    size_t nd  = 1;
    size_t ncd = 1;
    assert(nc > 0);
    double f1  = 2.0 * ncd / (nc + nd);
    return make_tuple(f1, f1, f1);
  }
  for (K d : gtvcom[u]) {
    buf.clear();
    set_intersection(comv[c].begin(), comv[c].end(), gtcomv[d].begin(), gtcomv[d].end(), back_inserter(buf));
    size_t nc  = comv[c].size();
    size_t nd  = gtcomv[d].size();
    size_t ncd = buf.size();
    // TODO: Remove asserts.
    assert(nc > 0);
    assert(nd > 0);
    assert(ncd > 0);
    assert(ncd <= nc);
    assert(ncd <= nd);
    double f1  = 2.0 * ncd / (nc + nd);
    amin = min(amin, f1);
    amax = max(amax, f1);
    aavg += f1;
  }
  return make_tuple(amin, amax, aavg / gtvcom[u].size());
}


/**
 * Calculate overall F1-scores wrt ground-truth communities.
 * @param x original graph
 * @param vcom community membership of each vertex
 * @param comv community vertices
 * @param gtvcom ground-truth community membership of each vertex
 * @param gtcomv ground-truth community vertices
 * @returns average of (minimum F1-score, maximum F1-score, average F1-score)
 */
template <class G, class K>
inline auto overallF1ScoresOmp(const G& x, const vector<K>& vcom, const vector2d<K>& comv, const vector2d<K>& gtvcom, const vector2d<K>& gtcomv) {
  size_t S = x.span();
  size_t N = x.order();
  int    T = omp_get_max_threads();
  if (N==0) return make_tuple(0.0, 0.0, 0.0);
  double amin = 0, amax = 0, aavg = 0;
  vector<vector<K>*> buf(T);
  for (int t=0; t<T; ++t)
    buf[t] = new vector<K>();
  #pragma omp parallel for schedule(dynamic, 2048) reduction(+:amin,amax,aavg)
  for (K u=0; u<S; ++u) {
    int t = omp_get_thread_num();
    if (!x.hasVertex(u)) continue;
    auto [f1min, f1max, f1avg] = vertexF1Scores(*buf[t], x, u, vcom, comv, gtvcom, gtcomv);
    amin += f1min;
    amax += f1max;
    aavg += f1avg;
  }
  for (int t=0; t<T; ++t)
    delete buf[t];
  return make_tuple(amin/N, amax/N, aavg/N);
}




// PERFORM EXPERIMENT
// ------------------

template <class G, class K>
void runExperiment(const G& x, const vector2d<K>& gtcomv, const vector2d<K>& gtvcom) {
  using V = typename G::edge_value_type;
  int repeat  = REPEAT_METHOD;
  int retries = 5;
  vector<K> *init = nullptr;
  double M = edgeWeightOmp(x)/2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto flog = [&](const auto& ans, const char *technique) {
    printf(
      "{%03d threads} -> "
      "{%09.1fms, %09.1fms preproc, %09.1fms firstpass, %09.1fms locmove, %09.1fms aggr, %zu affected, %04d iters, %03d passes, %01.9f modularity} %s\n",
      MAX_THREADS,
      ans.time, ans.preprocessingTime, ans.firstPassTime, ans.localMoveTime, ans.aggregationTime,
      ans.affectedVertices, ans.iterations, ans.passes, getModularity(x, ans, M), technique
    );
  };
  // Find static Louvain.
  auto b1 = louvainStaticOmp(x, init, {repeat});
  flog(b1, "louvainStaticOmp");
  // Calculate F1-scores for communities found by static Louvain.
  const vector<K>& vcom = b1.membership;
  auto comv = communityVerticesOmp(x, vcom);
  auto [f1min, f1max, f1avg] = overallF1ScoresOmp(x, vcom, comv, gtvcom, gtcomv);
  printf("F1 scores: min=%.4f, max=%.4f, avg=%.4f\n", f1min, f1max, f1avg);
}


int main(int argc, char **argv) {
  using K = uint32_t;
  using V = TYPE;
  install_sigsegv();
  char  *file       = argv[1];
  size_t span       = argc>2? stoull(argv[2]) : 0;
  bool weighted     = argc>3? stoi(argv[3]) : false;
  bool symmetric    = argc>4? stoi(argv[4]) : false;
  char *groundTruth = argc>5? argv[5] : nullptr;
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<K, None, V> x;
  if (span) x.respan(span);
  readEdgelistOmpW(x, file, weighted); LOG(""); println(x);
  if (!symmetric) { x = symmetricizeOmp(x); LOG(""); print(x); printf(" (symmetricize)\n"); }
  vector2d<K> gtcomv, gtvcom;
  if (groundTruth) {
    gtcomv.resize(2 * x.span());
    LOG("Loading ground truth communities %s ...\n", groundTruth);
    readCommunityVerticesOmpW(gtcomv, groundTruth);
    auto [cnum, cmin, cmax, cavg] = communityVerticesPropertiesOmp(gtcomv);
    LOG("Community vertices distribution: num=%zu, min=%zu, max=%zu, avg=%.2f\n", cnum, cmin, cmax, cavg);
    gtvcom = communityMembershipsOmp(x, gtcomv);
    auto [vnum, vmin, vmax, vavg] = communityMembershipsPropertiesOmp(x, gtvcom);
    LOG("Vertex memberships distribution: num=%zu, min=%zu, max=%zu, avg=%.2f\n", vnum, vmin, vmax, vavg);
  }
  runExperiment(x, gtcomv, gtvcom);
  printf("\n");
  return 0;
}
