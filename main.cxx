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
  readLinesOmpDo(s, [&](auto l, const auto& line) {
    const char *str = line.c_str();
    while (1) {
      const char *ptr = str;
      size_t u = strtoull(str, (char**) &ptr, 10);
      if (ptr==str) break;
      str = ptr;
      comv[l].push_back(K(u));
    }
  });
}

template <class G, class K>
inline vector2d<K> communityMembershipsOmp(const vector2d<K>& comv, const G& x) {
  vector2d<K> a(x.span());
  #pragma omp parallel
  {
    for (size_t i=0; i<comv.size(); ++i) {
      for (auto u : comv[i])
        if (belongsOmp(u)) a[u].push_back(K(i));
    }
  }
  return a;
}




// PERFORM EXPERIMENT
// ------------------

template <class G>
void runExperiment(const G& x) {
  using K = typename G::key_type;
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
  char  *file    = argv[1];
  size_t span    = argc>2? stoull(argv[2]) : 0;
  bool weighted  = argc>3? stoi(argv[3]) : false;
  bool symmetric = argc>4? stoi(argv[4]) : false;
  char  *truth   = argc>5? argv[5] : nullptr;
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<K, None, V> x;
  if (span) x.respan(span);
  readEdgelistOmpW(x, file, weighted); LOG(""); println(x);
  if (!symmetric) { x = symmetricizeOmp(x); LOG(""); print(x); printf(" (symmetricize)\n"); }
  vector2d<K> comv, vcoms;
  if (!truth) {
    LOG("Loading ground truth communities %s ...\n", truth);
    readGroundTruthCommunitiesOmpW(comv, truth);
    LOG("Loaded %zu ground truth communities.\n", comv.size());
    auto [cmin, cmax, cavg] = minMaxAverageSize(comv);
    LOG("Community size distribution: min=%zu, max=%zu, avg=%.2f.\n", cmin, cmax, cavg);
    vcoms = communityMembershipsOmp(comv, x);
    LOG("Obtained ground truth communities each vertex belongs to.\n");
    auto [vmin, vmax, vavg] = minMaxAverageSize(vcoms);
    LOG("Vertex membership distribution: min=%zu, max=%zu, avg=%.2f.\n", vmin, vmax, vavg);
  }
  runExperiment(x);
  printf("\n");
  return 0;
}
