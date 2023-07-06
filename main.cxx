#include <cstdint>
#include <cstdio>
#include <utility>
#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "src/main.hxx"

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

template <class T>
inline auto minMaxAverageSize(const vector2d<T>& xs) {
  size_t smin = numeric_limits<size_t>::min();
  size_t smax = 0, ssum = 0;
  for (const auto& x : xs) {
    smin = min(smin, x.size());
    smax = max(smax, x.size());
    ssum += x.size();
  }
  return make_tuple(smin, smax, double(ssum) / xs.size());
}


template <class G, class K>
inline double getModularity(const G& x, const LouvainResult<K>& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityByOmp(x, fc, M, 1.0);
}


template <class K>
inline void readGroundTruthCommunitiesOmpW(vector2d<K>& comv, const char *pth) {
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
    sortValues(comv[c]);
  });
}


template <class G, class K>
inline vector2d<K> communityMembershipsOmp(const G& x, const vector2d<K>& comv) {
  vector2d<K> a(x.span());
  #pragma omp parallel
  {
    for (size_t c=0; c<comv.size(); ++c) {
      for (auto u : comv[c])
        if (belongsOmp(u)) a[u].push_back(K(c));
    }
  }
  return a;
}


template <class G, class K>
inline vector2d<K> communityVerticesOmp(const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  vector2d<K> a(S);
  #pragma omp parallel
  {
    for (size_t u=0; u<x.span(); ++u) {
      if (!x.hasVertex(u)) continue;
      K c = vcom[u];
      if (belongsOmp(c)) a[c].push_back(K(u));
    }
  }
  #pragma omp parallel for schedule(dynamic, 2048)
  for (size_t i=0; i<a.size(); ++i)
    sortValues(a[i]);
  return a;
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
    readGroundTruthCommunitiesOmpW(gtcomv, groundTruth);
    LOG("Loaded %zu ground truth communities.\n", gtcomv.size());
    auto [cmin, cmax, cavg] = minMaxAverageSize(gtcomv);
    LOG("Community size distribution: min=%zu, max=%zu, avg=%.2f.\n", cmin, cmax, cavg);
    gtvcom = communityMembershipsOmp(x, gtcomv);
    LOG("Obtained ground truth communities each vertex belongs to.\n");
    auto [vmin, vmax, vavg] = minMaxAverageSize(gtvcom);
    LOG("Vertex membership distribution: min=%zu, max=%zu, avg=%.2f.\n", vmin, vmax, vavg);
  }
  runExperiment(x, gtcomv, gtvcom);
  printf("\n");
  return 0;
}
