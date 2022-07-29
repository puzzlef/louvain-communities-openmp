#pragma once
#include <utility>
#include <type_traits>
#include <iterator>
#include <algorithm>
#include <functional>
#include <vector>
#include <unordered_map>
#include "_queue.hxx"

using std::remove_reference_t;
using std::iterator_traits;
using std::vector;
using std::unordered_map;
using std::hash;
using std::move;
using std::distance;
using std::for_each;
using std::any_of;
using std::all_of;
using std::find;
using std::find_if;
using std::lower_bound;
using std::count;
using std::count_if;
using std::back_inserter;
using std::copy;
using std::transform;
using std::remove;
using std::remove_if;
using std::sort;
using std::reverse;
using std::set_difference;
using std::merge;
using std::inplace_merge;




// FIRST
// -----
// First position.

template <class I>
inline auto first_value(I ib, I ie) {
  using T = typename iterator_traits<I>::value_type;
  T a = ib != ie? *ib : T();
  return a;
}
template <class J>
inline auto firstValue(const J& x) {
  return first_value(x.begin(), x.end());
}




// FOR-EACH
// --------
// Perform a sale.

template <class J, class F>
inline void forEach(J& x, F fn) {
  for_each(x.begin(), x.end(), fn);
}
template <class J, class F>
inline void forEach(const J& x, F fn) {
  for_each(x.begin(), x.end(), fn);
}
template <class J, class F>
inline void cforEach(const J& x, F fn) {
  for_each(x.begin(), x.end(), fn);
}




// ANY-OF
// ------
// Is anything useful there?

template <class J, class F>
inline bool anyOf(const J& x, F fn) {
  return any_of(x.begin(), x.end(), fn);
}




// ALL-OF
// ------
// Is everything there?

template <class J, class F>
inline bool allOf(const J& x, F fn) {
  return all_of(x.begin(), x.end(), fn);
}




// FIND-*
// ------
// Find a business or its address.

template <class I, class T>
inline auto find_value(I ib, I ie, const T& v) {
  return find(ib, ie, v);
}
template <class J, class T>
inline size_t findValue(const J& x, const T& v) {
  auto   it = find_value(x.begin(), x.end(), v);
  return it - x.begin();
}
template <class J, class T>
inline size_t findValueAt(const J& x, const T& v) {
  auto   it = find_value(x.begin(), x.end(), v);
  return it != x.end()? it - x.begin() : size_t(-1);
}


template <class J, class F>
inline size_t findIf(const J& x, F fn) {
  auto   it = find_if(x.begin(), x.end(), fn);
  return it - x.begin();
}
template <class J, class F>
inline size_t findIfAt(const J& x, F fn) {
  auto   it = find_if(x.begin(), x.end(), fn);
  return it != x.end()? it - x.begin() : size_t(-1);
}




// LOWER-BOUND/FIND
// ----------------
// Find closest business, or its address.

template <class J, class T>
inline size_t lowerBound(const J& x, const T& v) {
  auto   it = lower_bound(x.begin(), x.end(), v);
  return it - x.begin();
}
template <class J, class T, class FL>
inline size_t lowerBound(const J& x, const T& v, FL fl) {
  auto   it = lower_bound(x.begin(), x.end(), v, fl);
  return it - x.begin();
}


template <class I, class T>
inline auto lower_find(I ib, I ie, const T& v) {
  auto   it = lower_bound(ib, ie, v);
  return it != ie && *it == v? it : ie;
}
template <class I, class T, class FL, class FE>
inline auto lower_find(I ib, I ie, const T& v, FL fl, FE fe) {
  auto   it = lower_bound(ib, ie, v, fl);
  return it != ie && fe(*it, v)? it : ie;
}
template <class J, class T>
inline size_t lowerFind(const J& x, const T& v) {
  auto   it = lower_find(x.begin(), x.end(), v);
  return it - x.begin();
}
template <class J, class T, class FL, class FE>
inline size_t lowerFind(const J& x, const T& v, FL fl, FE fe) {
  auto   it = lower_find(x.begin(), x.end(), v, fl, fe);
  return it - x.begin();
}
template <class J, class T>
inline size_t lowerFindAt(const J& x, const T& v) {
  auto   it = lower_find(x.begin(), x.end(), v) - x.begin();
  return it != x.end()? it - x.begin() : size_t(-1);
}
template <class J, class T, class FL, class FE>
inline size_t lowerFindAt(const J& x, const T& v, FL fl, FE fe) {
  auto   it = lower_find(x.begin(), x.end(), v, fl, fe) - x.begin();
  return it != x.end()? it - x.begin() : size_t(-1);
}




// EQUAL-*
// -------
// Check if values match.

template <class IX, class IY>
inline bool equal_values(IX xb, IX xe, IY yb) {
  return equal(xb, xe, yb);
}
template <class IX, class IY>
inline bool equal_values(IX xb, IX xe, IY yb, IY ye) {
  return equal(xb, xe, yb, ye);
}
template <class IX, class IY, class FE>
inline bool equal_values(IX xb, IX xe, IY yb, FE fe) {
  return equal(xb, xe, yb, fe);
}
template <class IX, class IY, class FE>
inline bool equal_values(IX xb, IX xe, IY yb, IY ye, FE fe) {
  return equal(xb, xe, yb, ye, fe);
}
template <class JX, class JY>
inline bool equalValues(const JX& x, const JY& y) {
  return equal_values(x.begin(), x.end(), y.begin(), y.end());
}
template <class JX, class JY, class FE>
inline bool equalValues(const JX& x, const JY& y, FE fe) {
  return equal_values(x.begin(), x.end(), y.begin(), y.end(), fe);
}




// COUNT-*
// -------
// Count businesses in a sector.

template <class I, class T>
inline size_t count_value(I ib, I ie, const T& v) {
  return count(ib, ie, v);
}
template <class J, class T>
inline size_t countValue(const J& x, const T& v) {
  return count_value(x.begin(), x.end(), v);
}


template <class J, class F>
inline size_t countIf(const J& x, F fn) {
  return count_if(x.begin(), x.end(), fn);
}




// COUNT-EACH
// ----------
// Count businesses in each sector.

template <class I, class M, class FM>
inline auto count_each(I ib, I ie, M& a, FM fm) {
  for_each(ib, ie, [&](const auto& v) { ++a[fm(v)]; });
  return a;
}
template <class I, class M>
inline auto count_each(I ib, I ie, M& a) {
  auto fm = [](const auto& v) { return v; };
  return count_each(ib, ie, a, fm);
}
template <class J, class M, class FM>
inline auto countEach(const J& x, M& a, FM fm) {
  return count_each(x.begin(), x.end(), a, fm);
}
template <class J, class M>
inline auto countEach(const J& x, M& a) {
  return count_each(x.begin(), x.end(), a);
}


template <class I, class FM>
inline auto count_each_unordered_map(I ib, I ie, FM fm) {
  using K = remove_reference_t<decltype(fm(*ib))>;
  unordered_map<K, size_t> a;
  return count_each(ib, ie, a, fm);
}
template <class I>
inline auto count_each_unordered_map(I ib, I ie) {
  auto fm = [](const auto& v) { return v; };
  return count_each_unordered_map(ib, ie, fm);
}
template <class J, class FM>
inline auto countEachUnorderedMap(const J& x, FM fm) {
  return count_each_unordered_map(x.begin(), x.end(), fm);
}
template <class J>
inline auto countEachUnorderedMap(const J& x) {
  return count_each_unordered_map(x.begin(), x.end());
}




// GROUP-VALUES
// ------------
// Group businesses in each sector.

template <class I, class M, class FM>
inline auto group_values(I ib, I ie, M& a, FM fm) {
  for_each(ib, ie, [&](const auto& v) { a[fm(v)].push_back(v); });
  return a;
}
template <class I, class M>
inline auto group_values(I ib, I ie, M& a) {
  auto fm = [](const auto& v) { return v; };
  return group_values(ib, ie, a, fm);
}
template <class J, class M, class FM>
inline auto groupValues(const J& x, M& a, FM fm) {
  return group_values(x.begin(), x.end(), a, fm);
}
template <class J, class M>
inline auto groupValues(const J& x, M& a) {
  return group_values(x.begin(), x.end(), a);
}


template <class I, class FM>
inline auto group_values_unordered_map(I ib, I ie, FM fm) {
  using K = remove_reference_t<decltype(fm(*ib))>;
  using V = typename iterator_traits<I>::value_type;
  unordered_map<K, vector<V>> a;
  return group_values(ib, ie, a, fm);
}
template <class I>
inline auto group_values_unordered_map(I ib, I ie) {
  auto fm = [](const auto& v) { return v; };
  return group_values_unordered_map(ib, ie, fm);
}
template <class J, class FM>
inline auto groupValuesUnorderedMap(const J& x, FM fm) {
  return group_values_unordered_map(x.begin(), x.end(), fm);
}
template <class J>
inline auto groupValuesUnorderedMap(const J& x) {
  return group_values_unordered_map(x.begin(), x.end());
}


template <class I, class FM>
inline auto group_values_vector(I ib, I ie, FM fm) {
  using V = typename iterator_traits<I>::value_type; vector<vector<V>> a;
  auto gs = group_values_unordered_map(ib, ie, fm);
  for (const auto& [k, g] : gs)
    a.push_back(move(g));
  return a;
}
template <class I>
inline auto group_values_vector(I ib, I ie) {
  auto fm = [](const auto& v) { return v; };
  return group_values_vector(ib, ie, fm);
}
template <class J, class FM>
inline auto groupValuesVector(const J& x, FM fm) {
  return group_values_vector(x.begin(), x.end(), fm);
}
template <class J>
inline auto groupValuesVector(const J& x) {
  return group_values_vector(x.begin(), x.end());
}




// COPY-*
// ------

template <class I, class IA>
inline auto copy_values(I ib, I ie, IA ab) {
  return copy(ib, ie, ab);
}
template <class J, class JA>
inline size_t copyValues(const J& x, JA& a) {
  auto   it = copy_values(x.begin(), x.end(), a.begin());
  return it - a.begin();
}


template <class I, class T>
inline auto copy_append(I ib, I ie, vector<T>& a) {
  return a.insert(a.end(), ib, ie);
}
template <class J, class T>
inline size_t copyAppend(const J& x, vector<T>& a) {
  auto   it = copy_append(x.begin(), x.end(), a);
  return it - a.begin();
}


template <class I, class T>
inline auto copy_write(I ib, I ie, vector<T>& a) {
  a.clear();
  return copy_append(ib, ie, a);
}
template <class J, class T>
inline size_t copyWrite(const J& x, vector<T>& a) {
  auto   it = copy_write(x.begin(), x.end(), a);
  return it - a.begin();
}


template <class I>
inline auto copy_vector(I ib, I ie) {
  using T = typename iterator_traits<I>::value_type; vector<T> a;
  copy_append(ib, ie, a);
  return a;
}
template <class J>
inline auto copyVector(const J& x) {
  return copy_vector(x.begin(), x.end());
}




// COPY-AT
// -------
// Requires random access!

template <class IX, class II, class IA>
auto copy_at(IX xb, IX xe, II ib, II ie, IA ab) {
  for (; ib < ie; ++ib)
    *(ab++) = *(xb + *(ib));
  return ab;
}
template <class JX, class JI, class JA>
inline size_t copyAt(const JX& x, const JI& is, JA& a) {
  auto   it = copy_at(x.begin(), x.end(), is.begin(), is.end(), a.begin());
  return it - a.begin();
}

template <class IX, class II, class T>
inline auto copy_at_append(IX xb, IX xe, II ib, II ie, vector<T>& a) {
  return copy_at(xb, xe, ib, ie, back_inserter(a));
}
template <class JX, class JI, class T>
inline size_t copyAtAppend(const JX& x, const JI& is, vector<T>& a) {
  auto   it = copy_at_append(x.begin(), x.end(), is.begin(), is.end(), a);
  return it - a.begin();
}

template <class IX, class II>
inline auto copy_at_vector(IX xb, IX xe, II ib, II ie) {
  using T = typename iterator_traits<IX>::value_type; vector<T> a;
  copy_at_append(xb, xe, ib, ie, a);
  return a;
}
template <class JX, class JI>
inline auto copyAtVector(const JX& x, const JI& is) {
  return copy_at_vector(x.begin(), x.end(), is.begin(), is.end());
}




// HASH-VALUE
// ----------

template <class I>
size_t hash_value(I ib, I ie) {
  // From boost::hash_combine.
  using T = typename iterator_traits<I>::value_type; size_t a = 0;
  for (; ib != ie; ++ib)
    a ^= hash<T>{}(*ib) + 0x9e3779b9 + (a<<6) + (a>>2);
  return a;
}
template <class J>
inline size_t hashValue(const J& x) {
  return hash_value(x.begin(), x.end());
}


template <class I, class IB>
inline size_t hash_unordered(I ib, I ie, IB bb) {
  IB be = copy(ib, ie, bb); sort(bb, be);
  return hash_value(bb, be);
}
template <class J, class JB>
inline size_t hashUnordered(const J& x, JB& buf) {
  return hash_unordered(x.begin(), x.end(), buf.begin());
}
template <class J, class T>
inline size_t hashUnordered(const J& x, vector<T>& buf) {
  size_t s = distance(x.begin(), x.end());
  if (buf.size() < s) buf.resize(s);
  return hash_unordered(x.begin(), x.end(), buf.begin());
}




// INDICES
// -------
// Keep the address of each business (yellow pages).

template <class I, class M>
auto value_indices(I ib, I ie, M& a) {
  size_t i = 0;
  for (; ib != ie; ++ib)
    a[*ib] = i++;
  return a;
}
template <class J, class M>
inline auto valueIndices(const J& x, M& a) {
  return value_indices(x.begin(), x.end(), a);
}

template <class I>
inline auto value_indices_unordered_map(I ib, I ie) {
  using K = typename iterator_traits<I>::value_type;
  unordered_map<K, size_t> a;
  return value_indices(ib, ie, a);
}
template <class J>
inline auto valueIndicesUnorderedMap(const J& x) {
  return value_indices_unordered_map(x.begin(), x.end());
}




// TRANSFORM
// ---------
// Switch around your portfolio.

template <class IX, class IA, class FM>
inline auto transform_values(IX xb, IX xe, IA ab, FM fm) {
  return transform(xb, xe, ab, fm);
}
template <class IX, class IY, class IA, class FM>
inline auto transform_values(IX xb, IX xe, IY yb, IA ab, FM fm) {
  return transform(xb, xe, yb, ab, fm);
}
template <class JX, class JA, class FM>
inline size_t transformValues(const JX& x, JA& a, FM fm) {
  auto   it = transform_values(x.begin(), x.end(), a.begin(), fm);
  return it - a.begin();
}
template <class JX, class JY, class JA, class FM>
inline size_t transformValues(const JX& x, const JY& y, JA& a, FM fm) {
  auto   it = transform_values(x.begin(), x.end(), y.begin(), a.begin(), fm);
  return it - a.begin();
}


template <class IX, class T, class FM>
inline auto transform_append(IX xb, IX xe, vector<T>& a, FM fm) {
  return transform_values(xb, xe, back_inserter(a), fm);
}
template <class IX, class IY, class T, class FM>
inline auto transform_append(IX xb, IX xe, IY yb, vector<T>& a, FM fm) {
  return transform_values(xb, xe, yb, back_inserter(a), fm);
}
template <class JX, class T, class FM>
inline size_t transformAppend(const JX& x, vector<T>& a, FM fm) {
  auto   it = transform_append(x.begin(), x.end(), a, fm);
  return it - a.begin();
}
template <class JX, class JY, class T, class FM>
inline size_t transformAppend(const JX& x, const JY& y, vector<T>& a, FM fm) {
  auto   it = transform_append(x.begin(), x.end(), y.begin(), a, fm);
  return it - a.begin();
}


template <class IX, class FM>
inline auto transform_vector(IX xb, IX xe, FM fm) {
  using T = remove_reference_t<decltype(fm(*xb))>; vector<T> a;
  transform_append(xb, xe, a, fm);
  return a;
}
template <class IX, class IY, class FM>
inline auto transform_vector(IX xb, IX xe, IY yb, FM fm) {
  using T = remove_reference_t<decltype(fm(*xb, *yb))>; vector<T> a;
  transform_append(xb, xe, yb, a, fm);
  return a;
}
template <class JX, class FM>
inline auto transformVector(const JX& x, FM fm) {
  return transform_vector(x.begin(), x.end(), fm);
}
template <class JX, class JY, class FM>
inline auto transformVector(const JX& x, const JY& y, FM fm) {
  return transform_vector(x.begin(), x.end(), y.begin(), fm);
}




// REMOVE
// ------
// Remove overpriced stocks from your portfolio.

template <class I, class T>
inline auto remove_value(I ib, I ie, const T& v) {
  return remove(ib, ie, v);
}
template <class J, class T>
inline size_t removeValue(const J& x, const T& v) {
  auto it = remove_value(x.begin(), x.end(), v);
  return it - x.begin();
}


template <class J, class F>
inline size_t removeIf(const J& x, F fn) {
  auto it = remove_if(x.begin(), x.end(), fn);
  return it - x.begin();
}




// FILTER-IF
// ---------

template <class I, class F>
auto filter_if(I ib, I ie, I ia, F fn) {
  for (; ib!=ie; ++ib) {
    if (!fn(*ib)) continue;
    if (ia!=ib) *ia = move(*ib);
    ++ia;
  }
  return ia;
}
template <class I, class F>
inline auto filter_if(I ib, I ie, F fn) {
  return filter_if(ib, ie, ib, fn);
}

template <class J, class F>
inline size_t filterIf(J& a, F fn) {
  auto it = filter_if(a.begin(), a.end(), fn);
  return it - a.begin();
}


template <class I, class F>
inline auto pairs_filter_if(I ib, I ie, I ia, F fn) {
  auto ft = [&](const auto& p) { return fn(p.first, p.second); };
  return filter_if(ib, ie, ia, ft);
}
template <class I, class F>
inline auto pairs_filter_if_key(I ib, I ie, I ia, F fn) {
  auto ft = [&](const auto& p) { return fn(p.first); };
  return filter_if(ib, ie, ia, ft);
}
template <class I, class F>
inline auto pairs_filter_if_value(I ib, I ie, I ia, F fn) {
  auto ft = [&](const auto& p) { return fn(p.second); };
  return filter_if(ib, ie, ia, ft);
}

template <class I, class F>
inline auto pairs_filter_if(I ib, I ie, F fn) {
  return pairs_filter_if(ib, ie, ib, fn);
}
template <class I, class F>
inline auto pairs_filter_if_key(I ib, I ie, F fn) {
  return pairs_filter_if_key(ib, ie, ib, fn);
}
template <class I, class F>
inline auto pairs_filter_if_value(I ib, I ie, F fn) {
  return pairs_filter_if_value(ib, ie, ib, fn);
}

template <class J, class F>
inline auto pairsFilterIf(J& a, F fn) {
  auto it = pairs_filter_if(a.begin(), a.end(), fn);
  return it - a.begin();
}
template <class J, class F>
inline auto pairsFilterIfKey(J& a, F fn) {
  auto it = pairs_filter_if_key(a.begin(), a.end(), fn);
  return it - a.begin();
}
template <class J, class F>
inline auto pairsFilterIfValue(J& a, F fn) {
  auto it = pairs_filter_if_value(a.begin(), a.end(), fn);
  return it - a.begin();
}




// SORT
// ----
// Lift your legs up.

template <class I>
inline void reverse_values(I ib, I ie) {
  reverse(ib, ie);
}
template <class J>
inline void reverseValues(J& x) {
  reverse_values(x.begin(), x.end());
}




// SORT
// ----
// Arrange your portfolio by ROCE.

template <class I>
inline void sort_values(I ib, I ie) {
  sort(ib, ie);
}
template <class I, class FL>
inline void sort_values(I ib, I ie, FL fl) {
  sort(ib, ie, fl);
}
template <class J>
inline void sortValues(J& x) {
  sort_values(x.begin(), x.end());
}
template <class J, class FL>
inline void sortValues(J& x, FL fl) {
  sort_values(x.begin(), x.end(), fl);
}




// SET-DIFFERENCE-*
// ----------------

template <class JX, class JY, class JA>
inline size_t setDifference(const JX& x, const JY& y, JA& a) {
  auto   it = set_difference(x.begin(), x.end(), y.begin(), y.end(), a.begin());
  return it - a.begin();
}
template <class JX, class JY, class JA, class FE>
inline size_t setDifference(const JX& x, const JY& y, JA& a, FE fe) {
  auto   it = set_difference(x.begin(), x.end(), y.begin(), y.end(), a.begin(), fe);
  return it - a.begin();
}


template <class IX, class IY, class T>
inline auto set_difference_append(IX xb, IX xe, IY yb, IY ye, vector<T>& a) {
  return set_difference(xb, xe, yb, ye, back_inserter(a));
}
template <class IX, class IY, class T, class FE>
inline auto set_difference_append(IX xb, IX xe, IY yb, IY ye, vector<T>& a, FE fe) {
  return set_difference(xb, xe, yb, ye, back_inserter(a), fe);
}
template <class JX, class JY, class T>
inline size_t setDifferenceAppend(const JX& x, const JY& y, vector<T>& a) {
  auto   it = set_difference_append(x.begin(), x.end(), y.begin(), y.end(), a);
  return it - a.begin();
}
template <class JX, class JY, class T, class FE>
inline size_t setDifferenceAppend(const JX& x, const JY& y, vector<T>& a, FE fe) {
  auto   it = set_difference_append(x.begin(), x.end(), y.begin(), y.end(), a, fe);
  return it - a.begin();
}


template <class IX, class IY>
inline auto set_difference_vector(IX xb, IX xe, IY yb, IY ye) {
  using T = typename iterator_traits<IX>::value_type; vector<T> a;
  set_difference_append(xb, xe, yb, ye, a);
  return a;
}
template <class IX, class IY, class FE>
inline auto set_difference_vector(IX xb, IX xe, IY yb, IY ye, FE fe) {
  using T = typename iterator_traits<IX>::value_type; vector<T> a;
  set_difference_append(xb, xe, yb, ye, a, fe);
  return a;
}
template <class JX, class JY>
inline auto setDifferenceVector(const JX& x, const JY& y) {
  return set_difference_vector(x.begin(), x.end(), y.begin(), y.end());
}
template <class JX, class JY, class FE>
inline auto setDifferenceVector(const JX& x, const JY& y, FE fe) {
  return set_difference_vector(x.begin(), x.end(), y.begin(), y.end(), fe);
}




// UNIQUE-*
// --------

template <class I>
inline auto unique_values(I ib, I ie) {
  return unique(ib, ie);
}
template <class I, class FE>
inline auto unique_values(I ib, I ie, FE fe) {
  return unique(ib, ie, fe);
}
template <class J>
inline size_t uniqueValues(J& x) {
  auto   it = unique_values(x.begin(), x.end());
  return it - x.begin();
}
template <class J, class FE>
inline size_t uniqueValues(J& x, FE fe) {
  auto   it = unique_values(x.begin(), x.end(), fe);
  return it - x.begin();
}


template <class I>
inline auto sorted_unique(I ib, I ie) {
  sort(ib, ie);
  return unique(ib, ie);
}
template <class I, class FL, class FE>
inline auto sorted_unique(I ib, I ie, FL fl, FE fe) {
  sort(ib, ie, fl);
  return unique(ib, ie, fe);
}
template <class J>
inline size_t sortedUnique(J& x) {
  auto   it = sorted_unique(x.begin(), x.end());
  return it - x.begin();
}
template <class J, class FL, class FE>
inline size_t sortedUnique(J& x, FL fl, FE fe) {
  auto   it = sorted_unique(x.begin(), x.end(), fl, fe);
  return it - x.begin();
}




// MERGE-*
// -------

template <class IX, class IY, class IA>
inline auto merge_values(IX xb, IX xe, IY yb, IY ye, IA ab) {
  return merge(xb, xe, yb, ye, ab);
}
template <class IX, class IY, class IA, class FL>
inline auto merge_values(IX xb, IX xe, IY yb, IY ye, IA ab, FL fl) {
  return merge(xb, xe, yb, ye, ab, fl);
}
template <class JX, class JY, class JA>
inline size_t mergeValues(const JX& x, const JY& y, JA& a) {
  auto   it = merge_values(x.begin(), x.end(), y.begin(), y.end(), a.begin());
  return it = a.begin();
}
template <class JX, class JY, class JA, class FL>
inline auto mergeValues(const JX& x, const JY& y, JA& a, FL fl) {
  auto   it = merge_values(x.begin(), x.end(), y.begin(), y.end(), a.begin(), fl);
  return it - a.begin();
}


template <class IX, class IY, class IA, class FL, class FE>
auto merge_unique(IX xb, IX xe, IY yb, IY ye, IA ab, FL fl, FE fe) {
  // `ab` points to the previous target value, unlike `xb` and `yb`.
  if (xb < xe && yb < ye) *ab = fl(*yb, *xb)? *(yb++) : *(xb++);
  else if (xb < xe) *ab = *(xb++);
  else if (yb < ye) *ab = *(yb++);
  else return ab;
  for (; xb < xe && yb < ye;) {
    if (fl(*yb, *xb)) { if (!fe(*yb, *ab)) { *(++ab) = *yb; } ++yb; }
    else              { if (!fe(*xb, *ab)) { *(++ab) = *xb; } ++xb; }
  }
  for (; xb < xe; ++xb)
    if (!fe(*xb, *ab)) *(++ab) = *xb;
  for (; yb < ye; ++yb)
    if (!fe(*yb, *ab)) *(++ab) = *yb;
  return ++ab;
}
template <class IX, class IY, class IA>
inline auto merge_unique(IX xb, IX xe, IY yb, IY ye, IA ab) {
  auto fl = [](const auto& a, const auto& b) { return a < b; };
  auto fe = [](const auto& a, const auto& b) { return a == b; };
  return merge_unique(xb, xe, yb, ye, ab, fl, fe);
}
template <class JX, class JY, class JA, class FL, class FE>
inline size_t mergeUnique(const JX& x, const JY& y, JA& a, FL fl, FE fe) {
  auto   it = merge_unique(x.begin(), x.end(), y.begin(), y.end(), fl, fe);
  return it - a.begin();
}
template <class JX, class JY, class JA>
inline size_t mergeUnique(const JX& x, const JY& y, JA& a) {
  auto   it = merge_unique(x.begin(), x.end(), y.begin(), y.end(), a.begin());
  return it - a.begin();
}


template <class IX, class IY, class FL>
auto merge_into(IX xb, IX xe, IY yb, IY ye, FL fl) {
  // `x` and `y` must be separate arrays, as `x` expands.
  IX ie = xe + (ye - yb);
  IX ix = xe - 1;
  IY iy = ye - 1;
  IX it = ie - 1;
  for (; iy >= yb; --it) {
    if (fl(*iy, *ix)) { *it = *ix; --ix; }
    else              { *it = *iy; --iy; }
  }
  return ie;
}
template <class IX, class IY>
inline auto merge_into(IX xb, IX xe, IY yb, IY ye) {
  auto fl = [](const auto& a, const auto& b) { return a < b; };
  return merge_into(xb, xe, yb, ye, fl);
}
template <class JX, class JY, class FL>
inline size_t mergeInto(JX& x, const JY& y, FL fl) {
  auto   it = merge_into(x.begin(), x.end(), y.begin(), y.end(), fl);
  return it - x.begin();
}
template <class JX, class JY>
inline size_t mergeInto(JX& x, const JY& y) {
  auto   it = merge_into(x.begin(), x.end(), y.begin(), y.end());
  return it - x.begin();
}


template <class J>
inline void inplaceMerge(J& x, size_t m) {
  inplace_merge(x.begin(), x.begin()+m, x.end());
}
template <class J, class FL>
inline void inplaceMerge(J& x, size_t m, FL fl) {
  inplace_merge(x.begin(), x.begin()+m, x.end(), fl);
}




template <class IX, class IB, class FL, class FE>
auto inplace_merge_unique(IX xb, IX xm, IX xe, IB bb, IB be, FL fl, FE fe) {
  // `it` points to the previous target value, unlike `ib` and `im`.
  // `bb` -> `be` should have atleast 2 + (`xm` -> `xe`) space.
  IX   it = xb, ib = xb, im = xm;
  auto bq = bounded_deque_view(bb, be);
  if (ib < xm && im < xe) {
    bq.push_back(*(ib++));
    *it = fl(*im, bq.front())? *(im++) : bq.pop_front();
  }
  else if (ib < xm) ++ib;
  else if (im < xe) ++im;
  else return it;
  for (; im < xe;) {
    if (ib < xm) bq.push_back(*(ib++));
    if (bq.empty()) break;
    if (fe(*it, bq.front())) { bq.pop_front(); continue; }
    if (fe(*it, *im))        { ++im;           continue; }
    *(++it) = fl(*im, bq.front())? *(im++) : bq.pop_front();
  }
  for (; !bq.empty(); bq.pop_front()) {
    if (ib < xm) bq.push_back(*(ib++));
    if (!fe(*it, bq.front())) *(++it) = bq.front();
  }
  for (; im < xe; ++im)
    if (!fe(*it, *im)) *(++it) = *im;
  return ++it;
}
template <class IX, class IB>
inline auto inplace_merge_unique(IX xb, IX xm, IX xe, IB bb, IB be) {
  auto fl = [](const auto& a, const auto& b) { return a < b; };
  auto fe = [](const auto& a, const auto& b) { return a == b; };
  return inplace_merge_unique(xb, xm, xe, bb, be, fl, fe);
}
template <class JX, class JB, class FL, class FE>
inline size_t inplaceMergeUnique(JX& x, size_t m, JB& b, FL fl, FE fe) {
  auto   it = inplace_merge_unique(x.begin(), x.begin()+m, x.end(), b.begin(), b.end(), fl, fe);
  return it - x.begin();
}
template <class JX, class JB>
inline size_t inplaceMergeUnique(JX& x, size_t m, JB& b) {
  auto   it = inplace_merge_unique(x.begin(), x.begin()+m, x.end(), b.begin(), b.end());
  return it - x.begin();
}
template <class JX, class T, class FL, class FE>
inline size_t inplaceMergeUnique(JX& x, size_t m, vector<T>& buf, FL fl, FE fe) {
  size_t s = 2 + distance(x.begin()+m, x.end());
  if (buf.size() < s) buf.resize(s);
  auto   it = inplace_merge_unique(x.begin(), x.begin()+m, x.end(), buf.begin(), buf.end(), fl, fe);
  return it - x.begin();
}
template <class JX, class T>
inline auto inplaceMergeUnique(JX& x, size_t m, vector<T>& buf) {
  size_t s = 2 + distance(x.begin()+m, x.end());
  if (buf.size() < s) buf.resize(s);
  auto   it = inplace_merge_unique(x.begin(), x.begin()+m, x.end(), buf.begin(), buf.end());
  return it - x.begin();
}
