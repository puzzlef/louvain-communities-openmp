#pragma once
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "_main.hxx"

using std::pair;
using std::vector;
using std::unordered_map;
using std::out_of_range;
using std::make_pair;
using std::iter_swap;
using std::find_if;
using std::lower_bound;
using std::sort;
using std::inplace_merge;




// BITSET-*
// --------
// Helps create bitsets.

#ifndef BITSET_TYPES
#define BITSET_TYPES(K, V, data) \
  using key_type   = K; \
  using value_type = V; \
  using pair_type  = pair<K, V>; \
  using iterator       = decltype(data.begin()); \
  using const_iterator = decltype(data.cbegin());
#endif


#ifndef BITSET_ITERATOR
#define BITSET_ITERATOR(K, V, data) \
  ITERABLE_ITERATOR(inline, noexcept, data.begin(), data.end())

#define BITSET_CITERATOR(K, V, data) \
  ITERABLE_CITERATOR(inline, noexcept, data.begin(), data.end())
#endif


#ifndef BITSET_SIZE
#define BITSET_SIZE(K, V, data)  \
  inline size_t size() const noexcept { return data.size(); }

#define BITSET_EMPTY(K, V) \
  inline bool empty()  const noexcept { return size() == 0; }
#endif


#ifndef BITSET_FCOMPARE
#define BITSET_FCOMPARE(K, V, fn, op) \
  auto fn = [](const pair<K, V>& p, const pair<K, V>& q) { return p.first op q.first; };
#define BITSET_FCOMPARES(K, V) \
  BITSET_FCOMPARE(K, V, fl, <) \
  BITSET_FCOMPARE(K, V, fe, ==)
#endif


#ifndef BITSET_FIND
#define BITSET_FIND(K, V, data) \
  inline auto find(const K& k)        noexcept { return locate_match(k); } \
  inline auto cfind(const K& k) const noexcept { return locate_match(k); } \
  inline auto find(const K& k)  const noexcept { return cfind(k); }

#define BITSET_ENTRIES(K, V, data) \
  /* dont change the keys! */ \
  inline auto values() noexcept { \
    return staticTransformIterable(data, PairSecond<K, V>()); \
  } \
  inline auto pairs() noexcept { \
    return iterable(data); \
  } \
  inline auto ckeys() const noexcept { \
    return staticTransformIterable(data, ConstPairFirst<K, V>()); \
  } \
  inline auto cvalues() const noexcept { \
    return staticTransformIterable(data, ConstPairSecond<K, V>()); \
  } \
  inline auto cpairs() const noexcept { \
    return iterable(data); \
  } \
  inline auto keys()   const noexcept { return ckeys(); } \
  inline auto values() const noexcept { return cvalues(); } \
  inline auto pairs()  const noexcept { return cpairs(); }

#define BITSET_FOREACH(K, V, data) \
  /* dont change the keys! */ \
  template <class F> \
  inline void forEachValue(F fn) { \
    for (pair<K, V>& p : data) fn(p.second); \
  } \
  template <class F> \
  inline void forEachPair(F fn) { \
    for (pair<K, V>& p : data) fn(p); \
  } \
  template <class F> \
  inline void forEach(F fn) { \
    for (pair<K, V>& p : data) fn(p.first, p.second); \
  } \
  template <class F> \
  inline void cforEachKey(F fn) const { \
    for (const pair<K, V>& p : data) fn(p.first); \
  } \
  template <class F> \
  inline void cforEachValue(F fn) const { \
    for (const pair<K, V>& p : data) fn(p.second); \
  } \
  template <class F> \
  inline void cforEachPair(F fn) const { \
    for (const pair<K, V>& p : data) fn(p); \
  } \
  template <class F> \
  inline void cforEach(F fn) const { \
    for (const pair<K, V>& p : data) fn(p.first, p.second); \
  } \
  template <class F> \
  inline void forEachKey(F fn)   const { cforEachKey(fn); } \
  template <class F> \
  inline void forEachValue(F fn) const { cforEachValue(fn); } \
  template <class F> \
  inline void forEachPair(F fn)  const { cforEachPair(fn); } \
  template <class F> \
  inline void forEach(F fn)      const { cforEach(fn); }
#endif


#ifndef BITSET_HAS
#define BITSET_HAS(K, V) \
  inline bool has(const K& k) const noexcept { \
    return locate_match(k) != end(); \
  }

#define BITSET_GET(K, V) \
  inline V get(const K& k) const noexcept { \
    auto it = locate_match(k); \
    return it == end()? V() : (*it).second; \
  }

#define BITSET_SET(K, V) \
  inline bool set(const K& k, const V& v) noexcept { \
    auto it = locate_match(k); \
    if (it == end()) return false; \
    (*it).second = v; \
    return true; \
  }

#define BITSET_SUBSCRIPT(K, V) \
  inline V& operator[](const K& k) noexcept { \
    auto it = locate_match(k); \
    return (*it).second; \
  } \
  inline const V& operator[](const K& k) const noexcept { \
    auto it = locate_match(k); \
    return (*it).second; \
  }

#define BITSET_AT(K, V) \
  inline V& at(const K& k) { \
    auto it = locate_match(k); \
    if (it == end()) throw out_of_range("bitset key not present"); \
    return (*it).second; \
  } \
  inline const V& at(const K& k) const { \
    auto it = locate_match(k); \
    if (it == end()) throw out_of_range("bitset key not present"); \
    return (*it).second; \
  }
#endif


#ifndef BITSET_CORRECT_NONE
#define BITSET_CORRECT_NONE(K, V) \
  inline bool correct(bool unq, vector<pair<K, V>>& buf) { \
    return false; \
  } \
  inline bool correct(bool unq=false) { \
    return false; \
  }
#endif


#ifndef BITSET_ADD_UNCHECKED
#define BITSET_ADD_UNCHECKED_DEFAULT(K, V) \
  inline bool addUnchecked(const K& k, const V& v=V()) { \
    return add(k, v); \
  }
#endif


#ifndef BITSET_FILTER_IF
#define BITSET_FILTER_IF_USING(K, V, data, name, fname) \
  template <class F> \
  inline void filterIf##name(F fn) { \
    data.resize(pairsFilterIf##fname(data, fn)); \
  }

#define BITSET_FILTER_IF(K, V, data) \
  BITSET_FILTER_IF_USING(K, V, data,,) \
  BITSET_FILTER_IF_USING(K, V, data, Key, Key) \
  BITSET_FILTER_IF_USING(K, V, data, Value, Value)

#define BITSET_FILTER_IF_BIPARTITE_USING(K, V, data, part, name, fname) \
  template <class F> \
  inline void filterIf##name(F fn) { \
    auto ib = data.begin(); \
    auto im = ib + part; \
    auto ie = data.end(); \
    auto ic = pairs_filter_if##fname(ib, im, ib, fn); \
    auto id = pairs_filter_if##fname(im, ie, ic, fn); \
    data.resize(id-ib); \
  }

#define BITSET_FILTER_IF_BIPARTITE(K, V, data, part) \
  BITSET_FILTER_IF_BIPARTITE_USING(K, V, data, part,,) \
  BITSET_FILTER_IF_BIPARTITE_USING(K, V, data, part, Key, _key) \
  BITSET_FILTER_IF_BIPARTITE_USING(K, V, data, part, Value, _value)
#endif




// DENSE-BITSET
// ------------
// An integer set that stores values in a full-size vector.
// It maintains integers in insertion order.

#ifndef DENSE_BITSET_FIND
#define DENSE_BITSET_FIND(K, V, ids) \
  inline auto find(const K& k) noexcept { return ids.find(k); } \
  inline auto cfind(const K& k) const noexcept { return ids.cfind(k); } \
  inline auto find(const K& k)  const noexcept { return cfind(k); }
#endif


#ifndef DENSE_BITSET_ENTRIES
#define DENSE_BITSET_ENTRIES(K, V, data, ids) \
  /* dont change the keys! */ \
  inline auto values() noexcept { \
    auto fn = [&](K k) { return data[k]; }; \
    return transformIterable(ids, fn); \
  } \
  inline auto pairs() noexcept { \
    auto fn = [&](K k) { return make_pair(k, data[k]); }; \
    return transformIterable(ids, fn); \
  } \
  inline auto ckeys() const noexcept { \
    return iterable(ids); \
  } \
  inline auto cvalues() const noexcept { \
    auto fn = [&](K k) { return data[k]; }; \
    return transformIterable(ids, fn); \
  } \
  inline auto cpairs() const noexcept { \
    auto fn = [&](K k) { return make_pair(k, data[k]); }; \
    return transformIterable(ids, fn); \
  } \
  inline auto keys()   const noexcept { return ckeys(); } \
  inline auto values() const noexcept { return cvalues(); } \
  inline auto pairs()  const noexcept { return cpairs(); }
#endif


#ifndef DENSE_BITSET_FOREACH
#define DENSE_BITSET_FOREACH(K, V, data, ids) \
  /* dont change the keys! */ \
  template <class F> \
  inline void forEachValue(F fn) { \
    for (K k : ids) fn(data[k]); \
  } \
  template <class F> \
  inline void forEachPair(F fn) { \
    for (K k : ids) fn(make_pair(k, data[k])); \
  } \
  template <class F> \
  inline void forEach(F fn) { \
    for (K k : ids) fn(k, data[k]); \
  } \
  template <class F> \
  inline void cforEachKey(F fn) const { \
    for (K k : ids) fn(k); \
  } \
  template <class F> \
  inline void cforEachValue(F fn) const { \
    for (K k : ids) fn(data[k]); \
  } \
  template <class F> \
  inline void cforEachPair(F fn) const { \
    for (K k : ids) fn(make_pair(k, data[k])); \
  } \
  template <class F> \
  inline void cforEach(F fn) const { \
    for (K k : ids) fn(k, data[k]); \
  } \
  template <class F> \
  inline void forEachKey(F fn)   const { cforEachKey(fn); } \
  template <class F> \
  inline void forEachValue(F fn) const { cforEachValue(fn); } \
  template <class F> \
  inline void forEachPair(F fn)  const { cforEachPair(fn); } \
  template <class F> \
  inline void forEach(F fn)      const { cforEach(fn); }
#endif


#ifndef DENSE_BITSET_HAS
#define DENSE_BITSET_HAS(K, V, data) \
  inline bool has(const K& k) const noexcept { \
    return k<data.size() && !!data[k]; \
  }

#define DENSE_BITSET_GET(K, V, data) \
  inline V get(const K& k) const noexcept { \
    return k<data.size()? data[k] : V(); \
  }

#define DENSE_BITSET_SET(K, V, data) \
  inline bool set(const K& k, const V& v) noexcept { \
    if (k>=data.size() || !data[k]) return false; \
    data[k] = v; \
    return true; \
  }

#define DENSE_BITSET_SUBSCRIPT(K, V, data) \
  inline V& operator[](const K& k) noexcept { \
    return data[k]; \
  } \
  inline const V& operator[](const K& k) const noexcept { \
    return data[k]; \
  }

#define DENSE_BITSET_AT(K, V, data) \
  inline V& at(const K& k) { \
    if (k>=data.size() || !data[k]) throw out_of_range("bitset key not present"); \
    return data[k]; \
  } \
  inline const V& at(const K& k) const { \
    if (k>=data.size() || !data[k]) throw out_of_range("bitset key not present"); \
    return data[k]; \
  }
#endif


#ifndef DENSE_BITSET_FILTER_IF
#define DENSE_BITSET_FILTER_IF_USING(K, V, data, ids, name, fn, test) \
  template <class F> \
  void filterIf##name(F fn) { \
    auto ib = ids.begin(); \
    auto ie = ids.end(); \
    auto i = ib, j = ib; \
    for (; i<ie; ++i) { \
      if (!test) data[*i] = V(); \
      else { *j = *i; ++j; } \
    } \
    ids.erase(j, ie); \
  }

#define DENSE_BITSET_FILTER_IF(K, V, data, ids) \
  DENSE_BITSET_FILTER_IF_USING(K, V, data, ids,, fn, fn(*i, data[*i])) \
  DENSE_BITSET_FILTER_IF_USING(K, V, data, ids, Key, fn, fn(*i)) \
  DENSE_BITSET_FILTER_IF_USING(K, V, data, ids, Value, fn, fn(data[*i]))
#endif


template <class K=int, class V=NONE>
class DenseBitset {
  // Data.
  protected:
  vector<V> data;
  vector<K> ids;

  // Types.
  public:
  BITSET_TYPES(K, V, ids)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, ids)
  BITSET_CITERATOR(K, V, ids)


  // Size operations.
  public:
  BITSET_SIZE(K, V, ids)
  BITSET_EMPTY(K, V)


  // Search operations.
  DENSE_BITSET_FIND(K, V, ids)


  // Access operations.
  public:
  DENSE_BITSET_ENTRIES(K, V, data, ids)
  DENSE_BITSET_FOREACH(K, V, data, ids)
  DENSE_BITSET_HAS(K, V, data)
  DENSE_BITSET_GET(K, V, data)
  DENSE_BITSET_SET(K, V, data)
  DENSE_BITSET_SUBSCRIPT(K, V, data)
  DENSE_BITSET_AT(K, V, data)


  // Update operations.
  public:
  BITSET_CORRECT_NONE(K, V)
  DENSE_BITSET_FILTER_IF(K, V, data, ids)
  inline bool clear() noexcept {
    if (ids.empty()) return false;
    for (K k : ids)
      data[k] = V();
    ids.clear();
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    if (k>=data.size()) data.resize(k+1);
    if (!data[k]) ids.push_back(k);
    data[k] = v;
    return true;
  }
  BITSET_ADD_UNCHECKED_DEFAULT(K, V)

  inline bool remove(const K& k) {
    if (k>=data.size() || !data[k]) return false;
    ids.erase(remove_value(ids.begin(), ids.end(), k), ids.end());
    data[k] = V();
    return true;
  }


  // Lifetime operations.
  public:
  DenseBitset(size_t capacity=0): data(capacity), ids() {}
};

template <class K=int, class V=NONE>
inline auto denseBitset(K _k=K(), V _v=V(), size_t capacity=0) {
  return DenseBitset<K, V>(capacity);
}




// SPARSE-BITSET
// -------------
// An integer set that is based on unordered_map.
// It does not maintain integers in insertion order.

#ifndef SPARSE_BITSET_FIND
#define SPARSE_BITSET_FIND(K, V, data) \
  inline auto find(const K& k)        noexcept { return data.find(k); } \
  inline auto cfind(const K& k) const noexcept { return data.find(k); } \
  inline auto find(const K& k)  const noexcept { return cfind(k); }
#endif


#ifndef SPARSE_BITSET_FOREACH
#define SPARSE_BITSET_FOREACH_USING(f0, f1, K, V, data, name0, name1, fn, it, call) \
  template <class F> \
  f0 void name0##forEach##name1(F fn) f1 { \
    auto ib = data.begin(); \
    auto ie = data.end(); \
    for (auto it=ib; it!=ie; ++it) call; \
  }

#define SPARSE_BITSET_FOREACH(K, V, data) \
  /* dont change the keys! */ \
  SPARSE_BITSET_FOREACH_USING(inline,, K, V, data,, Value, fn, it, fn(it->second)) \
  SPARSE_BITSET_FOREACH_USING(inline,, K, V, data,, Pair,  fn, it, fn(make_pair(it->first, it->second))) \
  SPARSE_BITSET_FOREACH_USING(inline,, K, V, data,,,       fn, it, fn(it->first, it->second)) \
  SPARSE_BITSET_FOREACH_USING(inline, const, K, V, data, c, Key,   fn, it, fn(it->first)) \
  SPARSE_BITSET_FOREACH_USING(inline, const, K, V, data, c, Value, fn, it, fn(it->second)) \
  SPARSE_BITSET_FOREACH_USING(inline, const, K, V, data, c, Pair,  fn, it, fn(make_pair(it->first, it->second))) \
  SPARSE_BITSET_FOREACH_USING(inline, const, K, V, data, c,,       fn, it, fn(it->first, it->second)) \
  template <class F> \
  inline void forEachKey(F fn)   const { cforEachKey(fn); } \
  template <class F> \
  inline void forEachValue(F fn) const { cforEachValue(fn); } \
  template <class F> \
  inline void forEachPair(F fn)  const { cforEachPair(fn); } \
  template <class F> \
  inline void forEach(F fn)      const { cforEach(fn); }
#endif


#ifndef SPARSE_BITSET_HAS
#define SPARSE_BITSET_HAS(K, V, data) \
  inline bool has(const K& k) const noexcept { \
    return data.count(k)==0; \
  }

#define SPARSE_BITSET_GET(K, V, data) \
  inline V get(const K& k) const noexcept { \
    auto it = data.find(k); \
    return it==data.end()? V() : it->second; \
  }

#define SPARSE_BITSET_SET(K, V, data) \
  inline bool set(const K& k, const V& v) noexcept { \
    data[k] = v; \
    return true; \
  }

#define SPARSE_BITSET_SUBSCRIPT(K, V, data) \
  inline V& operator[](const K& k) noexcept { \
    return data[k]; \
  } \
  inline const V& operator[](const K& k) const noexcept { \
    return data[k]; \
  }

#define SPARSE_BITSET_AT(K, V, data) \
  inline V& at(const K& k) { \
    return data[k]; \
  } \
  inline const V& at(const K& k) const { \
    return data[k]; \
  }
#endif


#ifndef SPARSE_BITSET_FILTER_IF
#define SPARSE_BITSET_FILTER_IF_USING(K, V, data, name, fn, test) \
  template <class F> \
  void filterIf##name(F fn) { \
    for (auto it=data.begin(); it!=data.end();) { \
      if(!test) it = data.erase(it); \
      else ++it; \
    } \
  }

#define SPARSE_BITSET_FILTER_IF(K, V, data) \
  SPARSE_BITSET_FILTER_IF_USING(K, V, data,, fn, fn(it->first, it->second)) \
  SPARSE_BITSET_FILTER_IF_USING(K, V, data, Key, fn, fn(it->first)) \
  SPARSE_BITSET_FILTER_IF_USING(K, V, data, Value, fn, fn(it->second))
#endif


template <class K=int, class V=NONE>
class SparseBitset {
  // Data.
  protected:
  unordered_map<K, V> data;

  // Types.
  public:
  BITSET_TYPES(K, V, data)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, data)
  BITSET_CITERATOR(K, V, data)


  // Size operations.
  public:
  BITSET_SIZE(K, V, data)
  BITSET_EMPTY(K, V)


  // Search operations.
  public:
  SPARSE_BITSET_FIND(K, V, data)


  // Access operations.
  public:
  BITSET_ENTRIES(K, V, data)
  SPARSE_BITSET_FOREACH(K, V, data)
  SPARSE_BITSET_HAS(K, V, data)
  SPARSE_BITSET_GET(K, V, data)
  SPARSE_BITSET_SET(K, V, data)
  SPARSE_BITSET_SUBSCRIPT(K, V, data)
  SPARSE_BITSET_AT(K, V, data)

  // Update operations.
  public:
  BITSET_CORRECT_NONE(K, V)
  SPARSE_BITSET_FILTER_IF(K, V, data)
  inline bool clear() noexcept {
    if (empty()) return false;
    data.clear();
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    data[k] = v;
    return true;
  }
  BITSET_ADD_UNCHECKED_DEFAULT(K, V)

  inline bool remove(const K& k) {
    return data.erase(k);
  }
};

template <class K=int, class V=NONE>
inline auto sparseBitset(K _k=K(), V _v=V()) {
  return SparseBitset<K, V>();
}




// UNOREDERD-BITSET
// ----------------
// An integer set that constantly checks duplicates.
// It maintains integers in insertion order.

#define UNORDERED_BITSET_LOCATE(K, V, f0, f1) \
  f0 auto locate_match(const K& k) f1 { \
    auto fe = [&](const pair<K, V>& p) { return p.first == k; }; \
    return find_if(begin(), end(), fe); \
  }


template <class K=int, class V=NONE>
class UnorderedBitset {
  // Data.
  protected:
  vector<pair<K, V>> data;

  // Types.
  public:
  BITSET_TYPES(K, V, data)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, data)
  BITSET_CITERATOR(K, V, data)


  // Size operations.
  public:
  BITSET_SIZE(K, V, data)
  BITSET_EMPTY(K, V)


  // Search operations.
  protected:
  UNORDERED_BITSET_LOCATE(K, V, inline, noexcept)
  UNORDERED_BITSET_LOCATE(K, V, inline, const noexcept)
  public:
  BITSET_FIND(K, V, data)


  // Access operations.
  public:
  BITSET_ENTRIES(K, V, data)
  BITSET_FOREACH(K, V, data)
  BITSET_HAS(K, V)
  BITSET_GET(K, V)
  BITSET_SET(K, V)
  BITSET_SUBSCRIPT(K, V)
  BITSET_AT(K, V)


  // Update operations.
  public:
  BITSET_CORRECT_NONE(K, V)
  BITSET_FILTER_IF(K, V, data)
  inline bool clear() noexcept {
    if (empty()) return false;
    data.clear();
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    auto it = locate_match(k);
    if  (it == end()) data.push_back({k, v});
    else (*it).second = v;
    return true;
  }
  inline bool addUnchecked(const K& k, const V& v=V()) {
    data.push_back({k, v});
    return true;
  }

  inline bool remove(const K& k) {
    auto it = locate_match(k);
    if (it == end()) return false;
    iter_swap(it, end()-1);
    data.pop_back();
    return true;
  }
};

template <class K=int, class V=NONE>
inline auto unorderedBitset(K _k=K(), V _v=V()) {
  return UnorderedBitset<K, V>();
}




// BITSET (SORTED)
// ---------------
// An integer set that constantly checks duplicates.
// It maintains integers in ascending value order.

#define ORDERED_BITSET_LOCATE(K, V, f0, f1) \
  f0 auto locate_spot(const K& k) f1 { \
    auto fl = [](const pair<K, V>& p, const K& k) { return p.first < k; }; \
    return lower_bound(begin(), end(), k, fl); \
  } \
  f0 auto locate_match(const K& k) f1 { \
    auto it = locate_spot(k); \
    return it == end() || (*it).first != k? end() : it; \
  }


template <class K=int, class V=NONE>
class OrderedBitset {
  // Data.
  protected:
  vector<pair<K, V>> data;

  // Types.
  public:
  BITSET_TYPES(K, V, data)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, data)
  BITSET_CITERATOR(K, V, data)


  // Size operations.
  public:
  BITSET_SIZE(K, V, data)
  BITSET_EMPTY(K, V)


  // Search operations.
  protected:
  ORDERED_BITSET_LOCATE(K, V, inline, noexcept)
  ORDERED_BITSET_LOCATE(K, V, inline, const noexcept)
  public:
  BITSET_FIND(K, V, data)


  // Access operations.
  public:
  BITSET_ENTRIES(K, V, data)
  BITSET_FOREACH(K, V, data)
  BITSET_HAS(K, V)
  BITSET_GET(K, V)
  BITSET_SET(K, V)
  BITSET_SUBSCRIPT(K, V)
  BITSET_AT(K, V)


  // Update operations.
  public:
  BITSET_CORRECT_NONE(K, V)
  BITSET_FILTER_IF(K, V, data)
  inline bool clear() noexcept {
    if (empty()) return false;
    data.clear();
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    auto it = locate_spot(k);
    if (it != end() && (*it).first == k) (*it).second = v;
    else data.insert(it, {k, v});
    return true;
  }
  BITSET_ADD_UNCHECKED_DEFAULT(K, V)

  inline bool remove(const K& k) {
    auto it = locate_match(k);
    if (it == end()) return false;
    data.erase(it);
    return true;
  }
};

template <class K=int, class V=NONE>
inline auto orderedBitset(K _k=K(), V _v=V()) {
  return OrderedBitset<K, V>();
}




// PORDERED-BITSET (PARTIALLY-SORTED)
// ----------------------------------
// An integer set that constantly checks duplicates.
// It maintains a portion of integers in ascending value order.

#define PORDERED_BITSET_LOCATE(K, V, f0, f1) \
  f0 auto locate_match_ordered(const K& k) f1 { \
    auto fl = [](const pair<K, V>& p, const K& k) { return p.first < k; }; \
    auto it = lower_bound(begin(), middle(), k, fl); \
    return it == middle() || (*it).first != k? end() : it; \
  } \
  f0 auto locate_match_unordered(const K& k) f1 { \
    auto fe = [&](const pair<K, V>& p) { return p.first == k; }; \
    return find_if(middle(), end(), fe); \
  } \
  f0 auto locate_match(const K& k) f1 { \
    auto it = locate_match_ordered(k); \
    return it != end()? it : locate_match_unordered(k); \
  }


template <class K=int, class V=NONE, size_t LIMIT=64>
class POrderedBitset {
  // Data.
  protected:
  vector<pair<K, V>> data;
  size_t ordered = 0;

  // Types.
  public:
  BITSET_TYPES(K, V, data)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, data)
  BITSET_CITERATOR(K, V, data)
  protected:
  ITERABLE_NAMES(inline, noexcept, middle, begin() + ordered)


  // Size operations.
  public:
  BITSET_SIZE(K, V, data)
  BITSET_EMPTY(K, V)
  protected:
  inline size_t unordered() const noexcept { return size() - ordered; }


  // Search operations.
  protected:
  PORDERED_BITSET_LOCATE(K, V, inline, noexcept)
  PORDERED_BITSET_LOCATE(K, V, inline, const noexcept)
  public:
  BITSET_FIND(K, V, data)


  // Ordering opertions.
  protected:
  inline void mergePartitions() {
    BITSET_FCOMPARE(K, V, fl, <)
    sort(middle(), end(), fl);
    inplace_merge(begin(), middle(), end(), fl);
    ordered = size();
  }


  // Access operations.
  public:
  BITSET_ENTRIES(K, V, data)
  BITSET_FOREACH(K, V, data)
  BITSET_HAS(K, V)
  BITSET_GET(K, V)
  BITSET_SET(K, V)
  BITSET_SUBSCRIPT(K, V)
  BITSET_AT(K, V)


  // Update operations.
  public:
  BITSET_FILTER_IF_BIPARTITE(K, V, data, ordered)
  inline bool correct(bool unq=false) {
    if (unordered() == 0) return false;
    mergePartitions();
    return true;
  }
  inline bool correct(bool unq, vector<pair<K, V>>& buf) { return correct(); }

  inline bool clear() noexcept {
    if (empty()) return false;
    data.clear();
    ordered = 0;
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    auto it = locate_match(k);
    if (it != end()) (*it).second = v;
    else {
      data.push_back({k, v});
      if (unordered() <= LIMIT) mergePartitions();
    }
    return true;
  }
  inline bool addUnchecked(const K& k, const V& v=V()) {
    data.push_back({k, v});
    if (unordered() <= LIMIT) mergePartitions();
    return true;
  }

  inline bool remove(const K& k) {
    auto it = locate_match(k);
    if (it == end()) return false;
    if (it < middle()) --ordered;
    data.erase(it);
    return true;
  }
};

template <class K=int, class V=NONE>
inline auto porderedBitset(K _k=K(), V _v=V()) {
  return POrderedBitset<K, V, 64>();
}




// RORDERED-BITSET (RISKY!)
// ------------------------
// An integer set that does not check duplicates.
// Removing duplicates can be done manually, with correct().
// It maintains integers in ascending value order (after correct()).

#define RORDERED_BITSET_LOCATE(K, V, f0, f1) \
  PORDERED_BITSET_LOCATE(K, V, f0, f1)


template <class K=int, class V=NONE>
class ROrderedBitset {
  // Data.
  protected:
  vector<pair<K, V>> data;
  size_t ordered = 0;

  // Types.
  public:
  BITSET_TYPES(K, V, data)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, data)
  BITSET_CITERATOR(K, V, data)
  protected:
  ITERABLE_NAMES(inline, noexcept, middle, begin() + ordered)


  // Size operations.
  public:
  BITSET_SIZE(K, V, data)
  BITSET_EMPTY(K, V)


  // Search operations.
  protected:
  RORDERED_BITSET_LOCATE(K, V, inline, noexcept)
  RORDERED_BITSET_LOCATE(K, V, inline, const noexcept)
  public:
  BITSET_FIND(K, V, data)


  // Access operations.
  public:
  BITSET_ENTRIES(K, V, data)
  BITSET_FOREACH(K, V, data)
  BITSET_HAS(K, V)
  BITSET_GET(K, V)
  BITSET_SET(K, V)
  BITSET_SUBSCRIPT(K, V)
  BITSET_AT(K, V)


  // Update operations.
  public:
  BITSET_FILTER_IF_BIPARTITE(K, V, data, ordered)
  inline bool correct(bool unq, vector<pair<K, V>>& buf) {
    BITSET_FCOMPARES(K, V)
    size_t e = size();
    if (ordered == size()) return false;
    sort(middle(), end(), fl);
    if (unq) inplaceMerge(data, ordered, fl);
    else e = inplaceMergeUnique(data, ordered, buf, fl, fe);
    data.resize(e);
    ordered = size();
    return true;
  }
  inline bool correct(bool unq=false) {
    BITSET_FCOMPARES(K, V)
    size_t e = size();
    if (ordered == size()) return false;
    sort(middle(), end(), fl);
    if (!unq) e = unique(middle(), end(), fe) - begin();
    inplaceMerge(data, ordered, fl);
    data.resize(e);
    ordered = size();
    return true;
  }

  inline bool clear() noexcept {
    if (empty()) return false;
    data.clear();
    ordered = 0;
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    data.push_back({k, v});
    return true;
  }
  BITSET_ADD_UNCHECKED_DEFAULT(K, V)

  inline bool remove(const K& k) {
    auto it = locate_match(k);
    if (it == end()) return false;
    if (it < middle()) --ordered;
    data.erase(it);
    return true;
  }
};

template <class K=int, class V=NONE>
inline auto rorderedBitset(K _k=K(), V _v=V()) {
  return ROrderedBitset<K, V>();
}




// RETYPE
// ------

template <class K, class V, class KA=K, class VA=V>
constexpr auto retype(const DenseBitset<K, V>& x, KA _k=KA(), VA _v=VA()) {
  return DenseBitset<KA, VA>();
}
template <class K, class V, class KA=K, class VA=V>
constexpr auto retype(const UnorderedBitset<K, V>& x, KA _k=KA(), VA _v=VA()) {
  return UnorderedBitset<KA, VA>();
}
template <class K, class V, class KA=K, class VA=V>
constexpr auto retype(const OrderedBitset<K, V>& x, KA _k=KA(), VA _v=VA()) {
  return OrderedBitset<KA, VA>();
}
template <class K, class V, size_t N, class KA=K, class VA=V, size_t NA=N>
constexpr auto retype(const POrderedBitset<K, V, N>& x, KA _k=KA(), VA _v=VA()) {
  return POrderedBitset<KA, VA, NA>();
}
template <class K, class V, class KA=K, class VA=V>
constexpr auto retype(const ROrderedBitset<K, V>& x, KA _k=KA(), VA _v=VA()) {
  return ROrderedBitset<KA, VA>();
}




// WRITE
// -----

template <class B>
void writeBitset(ostream& a, const B& x) {
  a << "{\n";
  x.forEach([&](const auto& k, const auto& v) {
    a << "  " << k << ": " << v << "\n";
  });
  a << "}";
}

template <class K, class V>
inline void write(ostream& a, const DenseBitset<K, V>& x)       { writeBitset(a, x); }
template <class K, class V>
inline void write(ostream& a, const SparseBitset<K, V>& x)      { writeBitset(a, x); }
template <class K, class V>
inline void write(ostream& a, const UnorderedBitset<K, V>& x)   { writeBitset(a, x); }
template <class K, class V>
inline void write(ostream& a, const OrderedBitset<K, V>& x)     { writeBitset(a, x); }
template <class K, class V, size_t N>
inline void write(ostream& a, const POrderedBitset<K, V, N>& x) { writeBitset(a, x); }
template <class K, class V>
inline void write(ostream& a, const ROrderedBitset<K, V>& x)    { writeBitset(a, x); }

template <class K, class V>
inline ostream& operator<<(ostream& a, const DenseBitset<K, V>& x)       { write(a, x); return a; }
template <class K, class V>
inline ostream& operator<<(ostream& a, const SparseBitset<K, V>& x)      { write(a, x); return a; }
template <class K, class V>
inline ostream& operator<<(ostream& a, const UnorderedBitset<K, V>& x)   { write(a, x); return a; }
template <class K, class V>
inline ostream& operator<<(ostream& a, const OrderedBitset<K, V>& x)     { write(a, x); return a; }
template <class K, class V, size_t N>
inline ostream& operator<<(ostream& a, const POrderedBitset<K, V, N>& x) { write(a, x); return a; }
template <class K, class V>
inline ostream& operator<<(ostream& a, const ROrderedBitset<K, V>& x)    { write(a, x); return a; }




// COPY
// ----

template <class B, class C>
inline void copyBitsetW(B& a, const C& x) {
  a.clear();
  x.forEach([&](const auto& k, const auto& v) { a.addUnchecked(k, v); });
}

template <class B, class K, class V>
inline void copyW(B& a, const DenseBitset<K, V>& x)       { copyBitsetW(a, x); }
template <class B, class K, class V>
inline void copyW(B& a, const SparseBitset<K, V>& x)      { copyBitsetW(a, x); }
template <class B, class K, class V>
inline void copyW(B& a, const UnorderedBitset<K, V>& x)   { copyBitsetW(a, x); }
template <class B, class K, class V>
inline void copyW(B& a, const OrderedBitset<K, V>& x)     { copyBitsetW(a, x); }
template <class B, class K, class V, size_t N>
inline void copyW(B& a, const POrderedBitset<K, V, N>& x) { copyBitsetW(a, x); }
template <class B, class K, class V>
inline void copyW(B& a, const ROrderedBitset<K, V>& x)    { copyBitsetW(a, x); }
