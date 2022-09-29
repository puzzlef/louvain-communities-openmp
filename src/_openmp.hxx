#pragma once
#include <cmath>
#include <utility>
#include <algorithm>
#include <array>
#include <vector>
#include <map>
#include <omp.h>
#include "_vector.hxx"

using std::array;
using std::vector;
using std::map;
using std::copy;
using std::swap;
using std::move;
using std::abs;
using std::max;
using std::sqrt;




// VECTOR OPERATIONS
// -----------------

#define SIZE_MIN_OMPM 100000
#define SIZE_MIN_OMPR 100000




// COPY-VALUES
// -----------

template <class T, class TA>
size_t copyValuesOmp(const T *x, TA *a, size_t N) {
  if (N<SIZE_MIN_OMPM) return copyValues(x, a, N);
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<N; ++i)
    a[i] = x[i];
  return N;
}
template <class T, class TA>
inline size_t copyValuesOmp(const vector<T>& x, vector<TA>& a) {
  return copyValuesOmp(x.data(), a.data(), x.size());
}
template <class T, class TA>
inline size_t copyValuesOmp(const vector<T>& x, vector<TA>& a, size_t i, size_t N) {
  return copyValuesOmp(x.data()+i, a.data()+i, N);
}


template <class T, class TA>
inline size_t copyValuesOmpW(TA *a, const T *x, size_t N) {
  return copyValuesOmp(x, a, N);
}
template <class T, class TA>
inline size_t copyValuesOmpW(vector<TA>& a, const vector<T>& x) {
  return copyValuesOmp(x, a);
}
template <class T, class TA>
inline size_t copyValuesOmpW(vector<TA>& a, const vector<T>& x, size_t i, size_t N) {
  return copyValuesOmp(x, a, i, N);
}




// FILL-VALUE
// ----------

template <class T, class V>
void fillValueOmpU(T *a, size_t N, const V& v) {
  if (N<SIZE_MIN_OMPM) { fillValueU(a, N, v); return; }
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<N; ++i)
    a[i] = v;
}
template <class T, class V>
inline void fillValueOmpU(vector<T>& a, const V& v) {
  fillValueOmpU(a.data(), a.size(), v);
}
template <class T, class V>
inline void fillValueOmpU(vector<T>& a, size_t i, size_t N, const V& v) {
  fillValueOmpU(a.data()+i, N, v);
}




// SUM-VALUES
// ----------

template <class T, class V=T>
V sumValuesOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return sumValues(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; ++i)
    a += x[i];
  return a;
}
template <class T, class V=T>
inline V sumValuesOmp(const vector<T>& x, V a=V()) {
  return sumValuesOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V sumValuesOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return sumValuesOmp(x.data()+i, N, a);
}




// SUM-ABS-VALUES
// --------------

template <class T, class V=T>
V sumAbsValuesOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return sumAbsValues(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; ++i)
    a += abs(x[i]);
  return a;
}
template <class T, class V=T>
inline V sumAbsValuesOmp(const vector<T>& x, V a=V()) {
  return sumAbsValuesOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V sumAbsValuesOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return sumAbsValuesOmp(x.data()+i, N, a);
}




// SUM-SQR-VALUES
// --------------

template <class T, class V=T>
V sumSqrValuesOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return sumSqrValues(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; ++i)
    a += x[i]*x[i];
  return a;
}
template <class T, class V=T>
inline V sumSqrValuesOmp(const vector<T>& x, V a=V()) {
  return sumSqrValuesOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V sumSqrValuesOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return sumSqrValuesOmp(x.data()+i, N, a);
}




// ADD-VALUE
// ---------

template <class T, class V>
void addValueOmp(T *a, size_t N, const V& v) {
  if (N<SIZE_MIN_OMPM) { addValue(a, N, v); return; }
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<N; ++i)
    a[i] += v;
}
template <class T, class V>
inline void addValueOmp(vector<T>& a, const V& v) {
  addValueOmp(a.data(), a.size(), v);
}
template <class T, class V>
inline void addValueOmp(vector<T>& a, size_t i, size_t N, const V& v) {
  addValueOmp(a.data()+i, N, v);
}




// MAX-VALUE
// ---------

template <class T, class V=T>
V maxValueOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return maxValue(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; ++i)
    a = max(a, x[i]);
  return a;
}
template <class T, class V=T>
inline V maxValueOmp(const vector<T>& x, V a=V()) {
  return maxValueOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V maxValueOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return maxValueOmp(x.data()+i, N, a);
}




// MAX-ABS-VALUE
// -------------

template <class T, class V=T>
V maxAbsValueOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return maxAbsValue(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; ++i)
    a = max(a, abs(x[i]));
  return a;
}
template <class T, class V=T>
inline V maxAbsValueOmp(const vector<T>& x, V a=V()) {
  return maxAbsValueOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V maxAbsValueOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return maxAbsValueOmp(x.data()+i, N, a);
}




// CONSTRAIN-MAX
// -------------

template <class T, class V>
void constrainMaxOmp(T *a, size_t N, const V& v) {
  if (N<SIZE_MIN_OMPM) { constrainMax(a, N, v); return; }
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<N; ++i)
    a[i] = max(a[i], v);
}
template <class T, class V>
inline void constrainMaxOmp(vector<T>& a, const V& v) {
  constrainMaxOmp(a.data(), a.size(), v);
}
template <class T, class V>
inline void constrainMaxOmp(vector<T>& a, size_t i, size_t N, const V& v) {
  constrainMaxOmp(a.data()+i, N, v);
}




// MIN-VALUE
// ---------

template <class T, class V=T>
V minValueOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return minValue(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; ++i)
    a = min(a, x[i]);
  return a;
}
template <class T, class V=T>
inline V minValueOmp(const vector<T>& x, V a=V()) {
  return minValueOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V minValueOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return minValueOmp(x.data()+i, N, a);
}




// MIN-ABS-VALUE
// -------------

template <class T, class V=T>
V minAbsValueOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return minAbsValue(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; ++i)
    a = min(a, abs(x[i]));
  return a;
}
template <class T, class V=T>
inline V minAbsValueOmp(const vector<T>& x, V a=V()) {
  return minAbsValueOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V minAbsValueOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return minAbsValueOmp(x.data()+i, N, a);
}




// CONSTRAIN-MIN
// -------------

template <class T, class V>
void constrainMinOmp(T *a, size_t N, const V& v) {
  if (N<SIZE_MIN_OMPM) { constrainMin(a, N, v); return; }
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<N; ++i)
    a[i] = min(a[i], v);
}
template <class T, class V>
inline void constrainMinOmp(vector<T>& a, const V& v) {
  constrainMinOmp(a.data(), a.size(), v);
}
template <class T, class V>
inline void constrainMinOmp(vector<T>& a, size_t i, size_t N, const V& v) {
  constrainMinOmp(a.data()+i, N, v);
}




// L1-NORM
// -------

template <class TX, class TY, class V=TX>
V l1NormOmp(const TX *x, const TY *y, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return l1Norm(x, y, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}
template <class TX, class TY, class V=TX>
inline V l1NormOmp(const vector<TX>& x, const vector<TY>& y, V a=V()) {
  return l1NormOmp(x.data(), y.data(), x.size(), a);
}
template <class TX, class TY, class V=TX>
inline V l1NormOmp(const vector<TX>& x, const vector<TY>& y, size_t i, size_t N, V a=V()) {
  return l1NormOmp(x.data()+i, y.data()+i, N, a);
}


template <class T, class V=T>
V l1NormOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return l1Norm(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; i++)
    a += abs(x[i]);
  return a;
}
template <class T, class V=T>
inline V l1NormOmp(const vector<T>& x, V a=V()) {
  return l1NormOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V l1NormOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return l1NormOmp(x.data()+i, N, a);
}




// L2-NORM
// -------

template <class TX, class TY, class V=TX>
V l2NormOmp(const TX *x, const TY *y, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return l2Norm(x, y, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; i++)
    a += (x[i] - y[i]) * (x[i] - y[i]);
  return sqrt(a);
}
template <class TX, class TY, class V=TX>
inline V l2NormOmp(const vector<TX>& x, const vector<TY>& y, V a=V()) {
  return l2NormOmp(x.data(), y.data(), x.size(), a);
}
template <class TX, class TY, class V=TX>
inline V l2NormOmp(const vector<TX>& x, const vector<TY>& y, size_t i, size_t N, V a=V()) {
  return l2NormOmp(x.data()+i, y.data()+i, N, a);
}


template <class T, class V=T>
V l2NormOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return l2Norm(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; i++)
    a += x[i] * x[i];
  return sqrt(a);
}
template <class T, class V=T>
inline V l2NormOmp(const vector<T>& x, V a=V()) {
  return l2NormOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V l2NormOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return l2NormOmp(x.data()+i, N, a);
}




// LI-NORM (INFINITY)
// ------------------

template <class TX, class TY, class V=TX>
V liNormOmp(const TX *x, const TY *y, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return liNorm(x, y, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; i++)
    a = max(a, abs(x[i] - y[i]));
  return a;
}
template <class TX, class TY, class V=TX>
inline V liNormOmp(const vector<TX>& x, const vector<TY>& y, V a=V()) {
  return liNormOmp(x.data(), y.data(), x.size(), a);
}
template <class TX, class TY, class V=TX>
inline V liNormOmp(const vector<TX>& x, const vector<TY>& y, size_t i, size_t N, V a=V()) {
  return liNormOmp(x.data()+i, y.data()+i, N, a);
}


template <class T, class V=T>
V liNormOmp(const T *x, size_t N, V a=V()) {
  if (N<SIZE_MIN_OMPR) return liNorm(x, N, a);
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (size_t i=0; i<N; i++)
    a = max(a, abs(x[i]));
  return a;
}
template <class T, class V=T>
inline V liNormOmp(const vector<T>& x, V a=V()) {
  return liNormOmp(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V liNormOmp(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return liNormOmp(x.data()+i, N, a);
}




// MULTIPLY-VALUES
// ---------------

template <class TX, class TY, class TA>
void multiplyValuesOmp(const TX *x, const TY *y, TA *a, size_t N) {
  if (N<SIZE_MIN_OMPM) { multiplyValues(x, y, a, N); return; }
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<N; i++)
    a[i] = x[i] * y[i];
}
template <class TX, class TY, class TA>
inline void multiplyValuesOmp(const vector<TX>& x, const vector<TY>& y, vector<TA>& a) {
  multiplyValuesOmp(x.data(), y.data(), a.data(), x.size());
}
template <class TX, class TY, class TA>
inline void multiplyValuesOmp(const vector<TX>& x, const vector<TY>& y, vector<TA>& a, size_t i, size_t N) {
  multiplyValuesOmp(x.data()+i, y.data()+i, a.data()+i, N);
}


template <class TA, class TX, class TY>
inline void multiplyValuesOmpW(TA *a, const TX *x, const TY *y, size_t N) {
  multiplyValuesOmp(x, y, a, N);
}
template <class TX, class TY, class TA>
inline void multiplyValuesOmpW(vector<TA>& a, const vector<TX>& x, const vector<TY>& y) {
  multiplyValuesOmp(x, y, a);
}
template <class TX, class TY, class TA>
inline void multiplyValuesOmpW(vector<TA>& a, const vector<TX>& x, const vector<TY>& y, size_t i, size_t N) {
  multiplyValuesOmp(x, y, a, i, N);
}




// MULTIPLY-VALUE
// --------------

template <class T, class TA, class V>
void multiplyValueOmp(const T *x, const TA *a, size_t N, const V& v) {
  if (N<SIZE_MIN_OMPM) { multiplyValue(x, a, N, v); return; }
  #pragma omp parallel for schedule(auto)
  for (size_t i=0; i<N; i++)
    a[i] = TA(x[i] * v);
}
template <class T, class TA, class V>
inline void multiplyValueOmp(const vector<T>& x, vector<TA>& a, const V& v) {
  multiplyValueOmp(x.data(), a.data(), x.size(), v);
}
template <class T, class TA, class V>
inline void multiplyValueOmp(const vector<T>& x, vector<TA>& a, size_t i, size_t N, const V& v) {
  multiplyValueOmp(x.data()+i, a.data()+i, N, v);
}
