#pragma once
#include <utility>
#include <array>
#include <string>
#include <vector>
#include <ostream>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <iterator>

using std::pair;
using std::array;
using std::string;
using std::vector;
using std::ios;
using std::ostream;
using std::ifstream;
using std::is_fundamental;
using std::iterator_traits;
using std::cout;




// READ-FILE
// ---------

string readFile(const char *pth) {
  string a; ifstream f(pth);
  f.seekg(0, ios::end);
  a.resize(f.tellg());
  f.seekg(0);
  f.read((char*) a.data(), a.size());
  return a;
}




// WRITE
// -----

template <class I>
void write_values(ostream& a, I ib, I ie) {
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
void writeValues(ostream& a, const J& x) {
  write_values(a, x.begin(), x.end());
}

template <class K, class V>
void write(ostream& a, const pair<K, V>& x) {
  a << x.first << ": " << x.second;
}
template <class T, size_t N>
void write(ostream& a, const array<T, N>& x) {
  writeValues(a, x);
}
template <class T>
void write(ostream& a, const vector<T>& x) {
  writeValues(a, x);
}

template <class K, class V>
ostream& operator<<(ostream& a, const pair<K, V>& x) {
  write(a, x); return a;
}
template <class T, size_t N>
ostream& operator<<(ostream& a, const array<T, N>& x) {
  write(a, x); return a;
}
template <class T>
ostream& operator<<(ostream& a, const vector<T>& x) {
  write(a, x); return a;
}




// PRINT*
// ------

template <class T>
void print(const T& x) { cout << x; }

template <class T>
void println(const T& x) { cout << x << "\n"; }
void println() { cout << "\n"; }
