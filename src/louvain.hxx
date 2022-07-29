#pragma once
#include <vector>
#include <utility>
#include "_main.hxx"

using std::vector;
using std::move;




template <class T>
struct LouvainOptions {
  int repeat;
  T   resolution;
  T   tolerance;
  T   phaseTolerance;
  int maxIterations;
  int maxPhaseIterations;

  LouvainOptions(int repeat=1, T resolution=1, T tolerance=0, T phaseTolerance=0, int maxIterations=500, int maxPhaseIterations=500) :
  repeat(repeat), resolution(resolution), tolerance(tolerance), phaseTolerance(phaseTolerance), maxIterations(maxIterations), maxPhaseIterations(maxPhaseIterations) {}
};


template <class K>
struct LouvainResult {
  vector<K> membership;
  int   iterations;
  float time;

  LouvainResult(vector<K>&& membership, int iterations=0, float time=0) :
  membership(membership), iterations(iterations), time(time) {}

  LouvainResult(vector<K>& membership, int iterations=0, float time=0) :
  membership(move(membership)), iterations(iterations), time(time) {}
};
