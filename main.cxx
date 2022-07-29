#include <utility>
#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include <omp.h>
#include "src/main.hxx"

using namespace std;




template <class G>
void runLouvain(const G& x, int repeat) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  int maxThreads = 12;
  V   resolution = V(1);
  V   tolerance  = V(1e-2);
  V   passTolerance = V(0);
  V   toleranceDeclineFactor = V(10);
  omp_set_num_threads(maxThreads);
  printf("OMP_NUM_THREADS=%d\n", maxThreads);
  auto M = edgeWeight(x)/2;
  auto Q = modularity(x, M, 1.0f);
  printf("[%01.6f modularity] noop\n", Q);
  // Run louvain sequential algorithm.
  do {
    LouvainResult<K> a = louvainSeq<false>(x, {repeat, resolution, tolerance, passTolerance, toleranceDeclineFactor});
    auto fc = [&](auto u) { return a.membership[u]; };
    auto Q  = modularity(x, fc, M, 1.0f);
    printf("[%09.3f ms; %04d iterations; %03d passes; %01.6f modularity] louvainSeq\n", a.time, a.iterations, a.passes, Q);
  } while(0);
  for (int chunkSize=1; chunkSize<=65536; chunkSize*=2) {
    omp_set_schedule(omp_sched_static, chunkSize);
    LouvainResult<K> a = louvainOmp<false>(x, {repeat, resolution, tolerance, passTolerance, toleranceDeclineFactor});
    auto fc = [&](auto u) { return a.membership[u]; };
    auto Q  = modularity(x, fc, M, 1.0f);
    printf("[%09.3f ms; %04d iterations; %03d passes; %01.6f modularity] louvainOmp {sch_kind: static, chunk_size: %d}\n", a.time, a.iterations, a.passes, Q, chunkSize);
  }
  for (int chunkSize=1; chunkSize<=65536; chunkSize*=2) {
    omp_set_schedule(omp_sched_dynamic, chunkSize);
    LouvainResult<K> a = louvainOmp<false>(x, {repeat, resolution, tolerance, passTolerance, toleranceDeclineFactor});
    auto fc = [&](auto u) { return a.membership[u]; };
    auto Q  = modularity(x, fc, M, 1.0f);
    printf("[%09.3f ms; %04d iterations; %03d passes; %01.6f modularity] louvainOmp {sch_kind: dynamic, chunk_size: %d}\n", a.time, a.iterations, a.passes, Q, chunkSize);
  }
  for (int chunkSize=1; chunkSize<=65536; chunkSize*=2) {
    omp_set_schedule(omp_sched_guided, chunkSize);
    LouvainResult<K> a = louvainOmp<false>(x, {repeat, resolution, tolerance, passTolerance, toleranceDeclineFactor});
    auto fc = [&](auto u) { return a.membership[u]; };
    auto Q  = modularity(x, fc, M, 1.0f);
    printf("[%09.3f ms; %04d iterations; %03d passes; %01.6f modularity] louvainOmp {sch_kind: guided, chunk_size: %d}\n", a.time, a.iterations, a.passes, Q, chunkSize);
  }
  for (int chunkSize=1; chunkSize<=65536; chunkSize*=2) {
    omp_set_schedule(omp_sched_auto, chunkSize);
    LouvainResult<K> a = louvainOmp<false>(x, {repeat, resolution, tolerance, passTolerance, toleranceDeclineFactor});
    auto fc = [&](auto u) { return a.membership[u]; };
    auto Q  = modularity(x, fc, M, 1.0f);
    printf("[%09.3f ms; %04d iterations; %03d passes; %01.6f modularity] louvainOmp {sch_kind: auto, chunk_size: %d}\n", a.time, a.iterations, a.passes, Q, chunkSize);
  }
}


int main(int argc, char **argv) {
  using K = int;
  using V = float;
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  OutDiGraph<K, None, V> x; V w = 1;
  printf("Loading graph %s ...\n", file);
  readMtxW(x, file); println(x);
  auto y  = symmetricize(x); print(y); printf(" (symmetricize)\n");
  auto fl = [](auto u) { return true; };
  // selfLoopU(y, w, fl); print(y); printf(" (selfLoopAllVertices)\n");
  runLouvain(y, repeat);
  printf("\n");
  return 0;
}
