#include <utility>
#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




// You can define datatype with -DTYPE=...
#ifndef TYPE
#define TYPE float
#endif
// You can define number of threads with -DMAX_THREADS=...
#ifndef MAX_THREADS
#define MAX_THREADS 12
#endif




template <class G, class K, class V>
double getModularity(const G& x, const LouvainResult<K>& a, V M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularity(x, fc, M, V(1));
}


template <class G, class R, class K, class V>
auto addRandomEdges(G& a, R& rnd, K span, V w, int batchSize) {
  int retries = 5;
  vector<tuple<K, K, V>> insertions;
  auto fe = [&](auto u, auto v, auto w) {
    a.addEdge(u, v, w);
    a.addEdge(v, u, w);
    insertions.push_back(make_tuple(u, v, w));
    insertions.push_back(make_tuple(v, u, w));
  };
  for (int i=0; i<batchSize; ++i)
    retry([&]() { return addRandomEdge(a, rnd, span, w, fe); }, retries);
  a.correct();
  return insertions;
}


template <class G, class R>
auto removeRandomEdges(G& a, R& rnd, int batchSize) {
  using K = typename G::key_type;
  int retries = 5;
  vector<tuple<K, K>> deletions;
  auto fe = [&](auto u, auto v) {
    a.removeEdge(u, v);
    a.removeEdge(v, u);
    deletions.push_back(make_tuple(u, v));
    deletions.push_back(make_tuple(v, u));
  };
  for (int i=0; i<batchSize; ++i)
    retry([&]() { return removeRandomEdge(a, rnd, fe); }, retries);
  a.correct();
  return deletions;
}




template <class G>
void runLouvain(const G& x, int repeat) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  vector<K> *init = nullptr;
  random_device dev;
  default_random_engine rnd(dev());
  int retries  = 5;
  V resolution = V(1);
  V tolerance  = V(1e-2);
  V passTolerance = V(0);
  V toleranceDeclineFactor = V(10);
  auto M = edgeWeight(x)/2;
  auto Q = modularity(x, M, 1.0f);
  printf("[%01.6f modularity] noop\n", Q);
  LouvainOptions<V> o = {repeat, resolution, tolerance, passTolerance, toleranceDeclineFactor};

  // Get last pass community memberships (static).
  LouvainResult<K> al = louvainSeqStatic(x, init, o);
  printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainSeqStatic\n", 0.0, al.time, al.iterations, al.passes, getModularity(x, al, M));
  // Batch of additions only (dynamic).
  for (int batchSize=500, i=0; batchSize<=100000; batchSize*=i&1? 5:2, ++i) {
    for (int batchCount=1; batchCount<=5; ++batchCount) {
      auto y = duplicate(x);
      auto insertions = addRandomEdges(y, rnd, x.span(), V(1), batchSize); vector<tuple<K, K>> deletions;
      LouvainResult<K> bk = louvainSeqStatic(y, init, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainSeqStatic\n",                double(batchSize), bk.time, bk.iterations, bk.passes, getModularity(y, bk, M));
      LouvainResult<K> bl = louvainOmpStatic(y, init, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainOmpStatic\n",                double(batchSize), bl.time, bl.iterations, bl.passes, getModularity(y, bl, M));
      LouvainResult<K> cl = louvainOmpStatic(y, &al.membership, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainOmpNaiveDynamic\n",          double(batchSize), cl.time, cl.iterations, cl.passes, getModularity(y, cl, M));
      LouvainResult<K> dl = louvainOmpDynamicDeltaScreening(y, deletions, insertions, &al.membership, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainOmpDynamicDeltaScreening\n", double(batchSize), dl.time, dl.iterations, dl.passes, getModularity(y, dl, M));
      LouvainResult<K> el = louvainOmpDynamicFrontier(y, deletions, insertions, &al.membership, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainOmpDynamicFrontier\n",       double(batchSize), el.time, el.iterations, el.passes, getModularity(y, el, M));
    }
  }
  // Batch of deletions only (dynamic).
  for (int batchSize=500, i=0; batchSize<=100000; batchSize*=i&1? 5:2, ++i) {
    for (int batchCount=1; batchCount<=5; ++batchCount) {
      auto y = duplicate(x);
      auto deletions = removeRandomEdges(y, rnd, batchSize); vector<tuple<K, K, V>> insertions;
      LouvainResult<K> bk = louvainSeqStatic(y, init, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainSeqStatic\n",                double(-batchSize), bk.time, bk.iterations, bk.passes, getModularity(y, bk, M));
      LouvainResult<K> bl = louvainOmpStatic(y, init, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainOmpStatic\n",                double(-batchSize), bl.time, bl.iterations, bl.passes, getModularity(y, bl, M));
      LouvainResult<K> cl = louvainOmpStatic(y, &al.membership, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainOmpNaiveDynamic\n",          double(-batchSize), cl.time, cl.iterations, cl.passes, getModularity(y, cl, M));
      LouvainResult<K> dl = louvainOmpDynamicDeltaScreening(y, deletions, insertions, &al.membership, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainOmpDynamicDeltaScreening\n", double(-batchSize), dl.time, dl.iterations, dl.passes, getModularity(y, dl, M));
      LouvainResult<K> el = louvainOmpDynamicFrontier(y, deletions, insertions, &al.membership, o);
      printf("[%1.0e batch_size; %09.3f ms; %04d iters.; %03d passes; %01.9f modularity] louvainOmpDynamicFrontier\n",       double(-batchSize), el.time, el.iterations, el.passes, getModularity(y, el, M));
    }
  }
}


int main(int argc, char **argv) {
  using K = int;
  using V = TYPE;
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  OutDiGraph<K, None, V> x; V w = 1;
  printf("Loading graph %s ...\n", file);
  readMtxW(x, file); println(x);
  auto y = symmetricize(x); print(y); printf(" (symmetricize)\n");
  // auto fl = [](auto u) { return true; };
  // selfLoopU(y, w, fl); print(y); printf(" (selfLoopAllVertices)\n");
  omp_set_num_threads(MAX_THREADS);
  printf("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  runLouvain(y, repeat);
  printf("\n");
  return 0;
}
