#pragma once
#include <vector>
#include "_main.hxx"




auto dendrogramLevel(const vector2d<int>& x, int l) {
  vector<int> a = x[0];
  for (int k=1; k<l; k++) {
    for (int& v : a)
      v = x[k][v];
  }
  return a;
}
