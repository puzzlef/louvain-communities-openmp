#pragma once
#include <ctime>
#include <type_traits>
#include <utility>
#include <iterator>
#include <array>
#include <vector>
#include <string>
#include <istream>
#include <ostream>
#include <iostream>
#include <chrono>

#ifdef OPENMP
#include <omp.h>
#endif

using std::pair;
using std::array;
using std::vector;
using std::string;
using std::istream;
using std::ostream;
using std::is_fundamental;
using std::iterator_traits;
using std::time_t;
using std::tm;
using std::localtime;
using std::chrono::time_point;
using std::chrono::system_clock;
using std::cout;




// READ LINES
// ----------

/**
 * Read lines from a stream and apply a function to each line.
 * @param s input stream
 * @param fp process function (line_index, line)
 */
template <class FP>
inline void readLinesDo(istream& s, FP fp) {
  string line;
  for (size_t l=0; getline(s, line);) {
    if (line[0]=='#') continue;
    fp(l++, line);
  }
}


#ifdef OPENMP
template <class FP>
inline void readLinesOmpDo(istream& s, FP fp) {
  const int LINES = 131072;
  vector<string> lines(LINES);
  for (size_t l=0;;) {
    // Read several lines from the stream.
    int READ = 0;
    for (int i=0; i<LINES;) {
      if (!getline(s, lines[i])) break;
      if (lines[i][0]=='#') continue;
      ++i; ++READ;
    }
    if (READ==0) break;
    // Process lines using multiple threads.
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i=0; i<READ; ++i)
      fp(l+i, lines[i]);
    l += READ;
  }
}
#endif




// WRITE
// -----

template <class I>
inline void write_values(ostream& a, I ib, I ie) {
  using T = typename iterator_traits<I>::value_type;
  if (is_fundamental<T>::value) {
    a << "{";
    for (; ib < ie; ++ib)
      a << " " << *ib;
    a << " }";
  }
  else {
    a << "{\n";
    for (; ib < ie; ++ib)
      a << "  " << *ib << "\n";
    a << "}";
  }
}
template <class J>
inline void writeValues(ostream& a, const J& x) {
  write_values(a, x.begin(), x.end());
}

template <class K, class V>
inline void write(ostream& a, const pair<K, V>& x) {
  a << x.first << ": " << x.second;
}
template <class T, size_t N>
inline void write(ostream& a, const array<T, N>& x) {
  writeValues(a, x);
}
template <class T>
inline void write(ostream& a, const vector<T>& x) {
  writeValues(a, x);
}

template <class K, class V>
inline ostream& operator<<(ostream& a, const pair<K, V>& x) {
  write(a, x); return a;
}
template <class T, size_t N>
inline ostream& operator<<(ostream& a, const array<T, N>& x) {
  write(a, x); return a;
}
template <class T>
inline ostream& operator<<(ostream& a, const vector<T>& x) {
  write(a, x); return a;
}




// WRITE TIME
// ----------

inline void writeTime(ostream& a, const time_t& x) {
  const int BUF = 64;
  char  buf[BUF];
  tm* t = localtime(&x);
  sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d",
    t->tm_year + 1900,
    t->tm_mon  + 1,
    t->tm_mday,
    t->tm_hour,
    t->tm_min,
    t->tm_sec
  );
  a << buf;
}
inline void writeTimePoint(ostream& a, const time_point<system_clock>& x) {
  writeTime(a, system_clock::to_time_t(x));
}

inline ostream& operator<<(ostream& a, const time_t& x) {
  writeTime(a, x); return a;
}
inline ostream& operator<<(ostream& a, const time_point<system_clock>& x) {
  writeTimePoint(a, x); return a;
}




// PRINT*
// ------

template <class T>
inline void print(const T& x)   { cout << x; }
template <class T>
inline void println(const T& x) { cout << x << "\n"; }
inline void println()           { cout << "\n"; }
