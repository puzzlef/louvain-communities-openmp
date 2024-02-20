#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef TYPE
/** Type of edge weights. */
#define TYPE float
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 5
#endif
#pragma endregion




#pragma region METHODS
#pragma region HELPERS
/**
 * Obtain the modularity of community structure on a graph.
 * @param x original graph
 * @param a rak result
 * @param M sum of edge weights
 * @returns modularity
 */
template <class G, class K>
inline double getModularity(const G& x, const LouvainResult<K>& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityBy(x, fc, M, 1.0);
}
#pragma endregion




#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 * @param x original graph
 */
template <class G>
void runExperiment(const G& x) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  int repeat = REPEAT_METHOD;
  double   M = edgeWeightOmp(x)/2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto flog = [&](const auto& ans, const char *technique) {
    printf(
      "{%09.1fms, %09.1fms mark, %09.1fms init, %09.1fms firstpass, %09.1fms locmove, %09.1fms aggr, %09.1fms split, %.3e aff, %04d iters, %03d passes, %01.9f modularity, %zu/%zu disconnected} %s\n",
      ans.time, ans.markingTime, ans.initializationTime, ans.firstPassTime, ans.localMoveTime, ans.aggregationTime, ans.splittingTime,
      double(ans.affectedVertices), ans.iterations, ans.passes, getModularity(x, ans, M),
      countValue(communitiesDisconnectedOmp(x, ans.membership), char(1)),
      communities(x, ans.membership).size(), technique
    );
  };
  // Get community memberships on original graph (static).
  {
    auto a0 = louvainStaticOmp(x, {repeat});
    flog(a0, "louvainStaticOmp");
  }
  {
    auto a1 = louvainSplitLastStaticOmp<1>(x, {repeat});
    flog(a1, "louvainSplitLastStaticOmp1");
    auto a2 = louvainSplitLastStaticOmp<2>(x, {repeat});
    flog(a2, "louvainSplitLastStaticOmp2");
    // auto a3 = louvainSplitLastStaticOmp<3>(x, {repeat});
    // flog(a3, "louvainSplitLastStaticOmp3");
    auto a4 = louvainSplitLastStaticOmp<4>(x, {repeat});
    flog(a4, "louvainSplitLastStaticOmp4");
  }
  {
    auto a5 = louvainSplitIterationStaticOmp<1>(x, {repeat});
    flog(a5, "louvainSplitIterationStaticOmp1");
    auto a6 = louvainSplitIterationStaticOmp<2>(x, {repeat});
    flog(a6, "louvainSplitIterationStaticOmp2");
    // auto a7 = louvainSplitIterationStaticOmp<3>(x, {repeat});
    // flog(a7, "louvainSplitIterationStaticOmp3");
    auto a8 = louvainSplitIterationStaticOmp<4>(x, {repeat});
    flog(a8, "louvainSplitIterationStaticOmp4");
  }
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
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
#pragma endregion
#pragma endregion
