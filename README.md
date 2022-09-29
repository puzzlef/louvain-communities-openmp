Comparing static vs dynamic approaches of OpenMP-based [Louvain algorithm]
for [community detection].

[Louvain] is an algorithm for **detecting communities in graphs**. *Community*
*detection* helps us understand the *natural divisions in a network* in an
**unsupervised manner**. It is used in **e-commerce** for *customer*
*segmentation* and *advertising*, in **communication networks** for *multicast*
*routing* and setting up of *mobile networks*, and in **healthcare** for
*epidemic causality*, setting up *health programmes*, and *fraud detection* is
hospitals. *Community detection* is an **NP-hard** problem, but heuristics exist
to solve it (such as this). **Louvain algorithm** is an **agglomerative-hierarchical**
community detection method that **greedily optimizes** for **modularity**
(**iteratively**).

**Modularity** is a score that measures *relative density of edges inside* vs
*outside* communities. Its value lies between `−0.5` (*non-modular clustering*)
and `1.0` (*fully modular clustering*). Optimizing for modularity *theoretically*
results in the best possible grouping of nodes in a graph.

Given an *undirected weighted graph*, all vertices are first considered to be
*their own communities*. In the **first phase**, each vertex greedily decides to
move to the community of one of its neighbors which gives greatest increase in
modularity. If moving to no neighbor's community leads to an increase in
modularity, the vertex chooses to stay with its own community. This is done
sequentially for all the vertices. If the total change in modularity is more
than a certain threshold, this phase is repeated. Once this **local-moving**
**phase** is complete, all vertices have formed their first hierarchy of
communities. The **next phase** is called the **aggregation phase**, where all
the *vertices belonging to a community* are *collapsed* into a single
**super-vertex**, such that edges between communities are represented as edges
between respective super-vertices (edge weights are combined), and edges within
each community are represented as self-loops in respective super-vertices
(again, edge weights are combined). Together, the local-moving and the
aggregation phases constitute a **pass**. This super-vertex graph is then used
as input for the next pass. This process continues until the increase in
modularity is below a certain threshold. As a result from each pass, we have a
*hierarchy of community memberships* for each vertex as a **dendrogram**. We
generally consider the *top-level hierarchy* as the *final result* of community
detection process.

*Louvain* algorithm is a hierarchical algorithm, and thus has two different
tolerance parameters: `tolerance` and `passTolerance`. **tolerance** defines the
minimum amount of increase in modularity expected, until the local-moving phase
of the algorithm is considered to have converged. We compare the increase in
modularity in each iteration of the local-moving phase to see if it is below
`tolerance`. **passTolerance** defines the minimum amount of increase in
modularity expected, until the entire algorithm is considered to have converged.
We compare the increase in modularity across all iterations of the local-moving
phase in the current pass to see if it is below `passTolerance`. `passTolerance`
is normally set to `0` (we want to maximize our modularity gain), but the same
thing does not apply for `tolerance`. Adjusting values of `tolerance` between
each pass have been observed to impact the runtime of the algorithm, without
significantly affecting the modularity of obtained communities. In this
experiment, we compare the performance of *three different types* of OpenMP-based
**dynamic Louvain** with respect to the *static* version. We also compare with
sequential static approach.

**Naive dynamic**:
- We start with previous community membership of each vertex (instead of each vertex its own community).

**Delta screening**:
- All edge batches are undirected, and sorted by source vertex-id.
- For edge additions across communities with source vertex `i` and highest modularity changing edge vertex `j*`,
  `i`'s neighbors and `j*`'s community is marked as affected.
- For edge deletions within the same community `i` and `j`,
  `i`'s neighbors and `j`'s community is marked as affected.

**Frontier-based**:
- All source and destination vertices are marked as affected for insertions and deletions.
- For edge additions across communities with source vertex `i` and destination vertex `j`,
  `i` is marked as affected.
- For edge deletions within the same community `i` and `j`,
  `i` is marked as affected.
- Vertices whose communities change in local-moving phase have their neighbors marked as affected.

First, we compute the community membership of each vertex using the static
Louvain algorithm. We then generate *batches* of *insertions (+)* and
*deletions (-)* of edges of sizes 500, 1000, 5000, ... 100000. For each batch
size, we generate *five* different batches for the purpose of *averaging*. Each
batch of edges (insertion / deletion) is generated randomly such that the
selection of each vertex (as endpoint) is *equally probable*. We choose the
Louvain *parameters* as `resolution = 1.0`, `tolerance = 1e-2` (for local-moving
phase) with *tolerance* decreasing after every pass by a factor of
`toleranceDeclineFactor = 10`, and a `passTolerance = 0.0` (when passes stop).
In addition we limit the maximum number of iterations in a single local-moving
phase with `maxIterations = 500`, and limit the maximum number of passes with
`maxPasses = 500`. We run the Louvain algorithm until convergence (or until the
maximum limits are exceeded), and measure the **time taken** for the
*computation* (performed 5 times for averaging), the **modularity score**, the
**total number of iterations** (in the *local-moving phase*), and the number
of **passes**. This is repeated for *seventeen* different graphs.

From the results, we make make the following observations. The frontier-based
dynamic approach converges the fastest, which obtaining communities with only
slightly lower modularity than other approaches. We also observe that
delta-screening based dynamic Louvain algorithm has the same performance as that
of the naive dynamic approach. Therefore, **frontier-based dynamic Louvain**
would be the **best choice**. However, one of the most interesting things we
note is that sequential static approach is only slightly slower than
OpenMP-based approach with 12 threads. This is indeed suprising, and is likely
due to higher pressure on cache coherence system as well as the algorithm
becoming closes to an unordered approach, which is inherently slower than an
ordered approach. Trying to avoid community swaps with OpenMP-based approach
does not seem to improve performance by any significant amount. However, it is
possible that if unordered approach is used with OpenMP, then its performance
may be a bit better.

All outputs are saved in a [gist] and a small part of the output is listed here.
Some [charts] are also included below, generated from [sheets]. The input data
used for this experiment is available from the [SuiteSparse Matrix Collection].
This experiment was done with guidance from [Prof. Kishore Kothapalli] and
[Prof. Dip Sankar Banerjee].


[Louvain algorithm]: https://en.wikipedia.org/wiki/Louvain_method
[community detection]: https://en.wikipedia.org/wiki/Community_search

<br>

```bash
$ g++ -std=c++17 -O3 main.cxx
$ ./a.out ~/data/web-Stanford.mtx
$ ./a.out ~/data/web-BerkStan.mtx
$ ...

# Loading graph /home/subhajit/data/web-Stanford.mtx ...
# order: 281903 size: 2312497 [directed] {}
# order: 281903 size: 3985272 [directed] {} (symmetricize)
# OMP_NUM_THREADS=12
# [-0.000497 modularity] noop
# [0e+00 batch_size; 00440.006 ms; 0025 iters.; 009 passes; 0.923382580 modularity] louvainSeqStatic
# [5e+02 batch_size; 00394.563 ms; 0027 iters.; 009 passes; 0.923351705 modularity] louvainSeqStatic
# [5e+02 batch_size; 00278.041 ms; 0030 iters.; 009 passes; 0.927210510 modularity] louvainOmpStatic
# [5e+02 batch_size; 00101.633 ms; 0003 iters.; 003 passes; 0.914556682 modularity] louvainOmpNaiveDynamic
# [5e+02 batch_size; 00107.587 ms; 0004 iters.; 004 passes; 0.914944172 modularity] louvainOmpDynamicDeltaScreening
# [5e+02 batch_size; 00088.939 ms; 0003 iters.; 003 passes; 0.913411736 modularity] louvainOmpDynamicFrontier
# ...
# [1e+05 batch_size; 00552.708 ms; 0019 iters.; 006 passes; 0.925986648 modularity] louvainSeqStatic
# [1e+05 batch_size; 00321.840 ms; 0026 iters.; 005 passes; 0.925740898 modularity] louvainOmpStatic
# [1e+05 batch_size; 00130.201 ms; 0010 iters.; 004 passes; 0.912258267 modularity] louvainOmpNaiveDynamic
# [1e+05 batch_size; 00115.326 ms; 0002 iters.; 002 passes; 0.912238598 modularity] louvainOmpDynamicDeltaScreening
# [1e+05 batch_size; 00117.582 ms; 0010 iters.; 004 passes; 0.911141872 modularity] louvainOmpDynamicFrontier
# [-5e+02 batch_size; 00389.191 ms; 0024 iters.; 008 passes; 0.923092246 modularity] louvainSeqStatic
# [-5e+02 batch_size; 00286.037 ms; 0026 iters.; 008 passes; 0.926223278 modularity] louvainOmpStatic
# [-5e+02 batch_size; 00106.936 ms; 0004 iters.; 004 passes; 0.914734781 modularity] louvainOmpNaiveDynamic
# [-5e+02 batch_size; 00110.743 ms; 0004 iters.; 004 passes; 0.914734781 modularity] louvainOmpDynamicDeltaScreening
# [-5e+02 batch_size; 00073.786 ms; 0002 iters.; 002 passes; 0.913197339 modularity] louvainOmpDynamicFrontier
# ...
# [-1e+05 batch_size; 00471.004 ms; 0018 iters.; 006 passes; 0.881129861 modularity] louvainSeqStatic
# [-1e+05 batch_size; 00387.020 ms; 0017 iters.; 006 passes; 0.877473772 modularity] louvainOmpStatic
# [-1e+05 batch_size; 00115.070 ms; 0004 iters.; 004 passes; 0.869384587 modularity] louvainOmpNaiveDynamic
# [-1e+05 batch_size; 00106.264 ms; 0004 iters.; 004 passes; 0.869384587 modularity] louvainOmpDynamicDeltaScreening
# [-1e+05 batch_size; 00099.210 ms; 0004 iters.; 004 passes; 0.868384838 modularity] louvainOmpDynamicFrontier
#
# Loading graph /home/subhajit/data/web-BerkStan.mtx ...
# order: 685230 size: 7600595 [directed] {}
# order: 685230 size: 13298940 [directed] {} (symmetricize)
# OMP_NUM_THREADS=12
# [-0.000316 modularity] noop
# [0e+00 batch_size; 00735.782 ms; 0028 iters.; 009 passes; 0.935839474 modularity] louvainSeqStatic
# [5e+02 batch_size; 00741.732 ms; 0027 iters.; 009 passes; 0.937690854 modularity] louvainSeqStatic
# [5e+02 batch_size; 00655.410 ms; 0028 iters.; 009 passes; 0.935854971 modularity] louvainOmpStatic
# [5e+02 batch_size; 00216.361 ms; 0003 iters.; 003 passes; 0.932617486 modularity] louvainOmpNaiveDynamic
# [5e+02 batch_size; 00246.924 ms; 0003 iters.; 003 passes; 0.932617486 modularity] louvainOmpDynamicDeltaScreening
# [5e+02 batch_size; 00186.971 ms; 0004 iters.; 004 passes; 0.932644546 modularity] louvainOmpDynamicFrontier
# ...
```

[![](https://i.imgur.com/p8ZCnh3.png)][sheetp]
[![](https://i.imgur.com/Egy7d0k.png)][sheetp]
[![](https://i.imgur.com/sgqDzGG.png)][sheetp]
[![](https://i.imgur.com/KXMkFpV.png)][sheetp]

<br>
<br>


## References

- [Fast unfolding of communities in large networks; Vincent D. Blondel et al. (2008)](https://arxiv.org/abs/0803.0476)
- [Community Detection on the GPU; Md. Naim et al. (2017)](https://arxiv.org/abs/1305.2006)
- [Scalable Static and Dynamic Community Detection Using Grappolo; Mahantesh Halappanavar et al. (2017)](https://ieeexplore.ieee.org/document/8091047)
- [From Louvain to Leiden: guaranteeing well-connected communities; V.A. Traag et al. (2019)](https://www.nature.com/articles/s41598-019-41695-z)
- [CS224W: Machine Learning with Graphs | Louvain Algorithm; Jure Leskovec (2021)](https://www.youtube.com/watch?v=0zuiLBOIcsw)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)

<br>
<br>

[![](https://i.imgur.com/UGB0g2L.jpg)](https://www.youtube.com/watch?v=pIF3wOet-zw)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/540003159.svg)](https://zenodo.org/badge/latestdoi/540003159)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[Louvain]: https://en.wikipedia.org/wiki/Louvain_method
[gist]: https://gist.github.com/wolfram77/3f9d0452901d3719d0e0baf345615c91
[charts]: https://imgur.com/a/igL8c2j
[sheets]: https://docs.google.com/spreadsheets/d/13GMWmhcw5EbCVgVVtP08yQQrpQmE_EIyhNGb0MzEPX8/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vR_oG_LC7Nuy3B8dlM1SUM4UeCpB946foX7cvBxeYs8YZHS0h76thPjQU5kI_shiSvD7FjbppNT4-G1/pubhtml
