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
  T   passTolerance;
  T   toleranceDeclineFactor;
  int maxIterations;
  int maxPasses;

  LouvainOptions(int repeat=1, T resolution=1, T tolerance=1e-2, T passTolerance=0, T toleranceDeclineFactor=10, int maxIterations=500, int maxPasses=500) :
  repeat(repeat), resolution(resolution), tolerance(tolerance), passTolerance(passTolerance), toleranceDeclineFactor(toleranceDeclineFactor), maxIterations(maxIterations), maxPasses(maxPasses) {}
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
