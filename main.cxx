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

template <class G, class K>
inline double getModularity(const G& x, const LouvainResult<K>& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityByOmp(x, fc, M, 1.0);
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
  auto b1 = louvainStaticOmp(x, init, {repeat, 1.0, 1e-2, 0.8, 100, 10, 10});
  flog(b1, "louvainStaticOmpMax10");
  auto b2 = louvainStaticOmp(x, init, {repeat, 1.0, 1e-2, 0.8, 100, 20, 10});
  flog(b2, "louvainStaticOmpMax20");
  auto b3 = louvainStaticOmp(x, init, {repeat, 1.0, 1e-2, 0.8, 100, 50, 10});
  flog(b3, "louvainStaticOmpMax50");
  auto b4 = louvainStaticOmp(x, init, {repeat, 1.0, 1e-2, 0.8, 100, 100, 10});
  flog(b4, "louvainStaticOmpMax100");
}


int main(int argc, char **argv) {
  using K = uint32_t;
  using V = TYPE;
  install_sigsegv();
  char *file     = argv[1];
  bool symmetric = argc>2? stoi(argv[2]) : false;
  bool weighted  = argc>3? stoi(argv[3]) : false;
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<K, None, V> x;
  readMtxOmpW(x, file, weighted); LOG(""); println(x);
  if (!symmetric) { x = symmetricizeOmp(x); LOG(""); print(x); printf(" (symmetricize)\n"); }
  runExperiment(x);
  printf("\n");
  return 0;
}
