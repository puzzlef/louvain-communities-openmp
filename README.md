Multi-threaded OpenMP-based [Louvain] algorithm for [community detection].

The **Louvain method** is a greedy modularity optimization based agglomerative
algorithm by *Blondel et al.* that efficiently identifies **high-quality**
**communities** in a graph. With `O(n \log n)` average time complexity (`n` being
the number of vertices in the graph), it consists of two phases: local-moving
and aggregation. In the **local-moving phase**, each vertex greedily selects the
neighboring community with the highest modularity increase. The **aggregation**
**phase** merges vertices within a community into a super-vertex. This iterative
process produces a dendrogram hierarchy of community memberships, with the
top-level representing the final output.

[Louvain]: https://en.wikipedia.org/wiki/Louvain_method
[community detection]: https://en.wikipedia.org/wiki/Community_structure

<br>
<br>


### Optimizations

#### OpenMP schedule

It appears `schedule(dynamic, 2048)` is the best choice.

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-schedule),
> [output](https://gist.github.com/wolfram77/092d8d26e7d8c73d334c9d0f3629181d), or
> [sheets][sheets-o1].

[![](https://i.imgur.com/q1Z6ZCz.png)][sheets-o1]
[![](https://i.imgur.com/iHYYLaq.png)][sheets-o1]

[sheets-o1]: https://docs.google.com/spreadsheets/d/1e5VKjsEFXWnPfNofpz2-oYg7ERkCPgADY1LHk0V17z4/edit?usp=sharing


#### Limiting number of iterations

It appears allowing a **maximum** of `20` **iterations** is ok.

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-max-iterations),
> [output](https://gist.github.com/wolfram77/9247cb416ebb0408917dc6ee0e290630), or
> [sheets][sheets-o2].

[![](https://i.imgur.com/113nP30.png)][sheets-o2]
[![](https://i.imgur.com/bY0n2eR.png)][sheets-o2]

[sheets-o2]: https://docs.google.com/spreadsheets/d/1R4G61T1RMPvJilf1eW1AVlZNAnWpTEL1rxuywTnjRyU/edit?usp=sharing


#### Adjusting tolerance drop rate (threshold scaling)

Instead of using a *fixed* tolerance across all passes of the Louvain algorithm,
we can start with a *high* tolerance and then *gradually* reduce it. This is
called *threshold scaling*, and it helps minimize runtime as the first pass is
usually the most expensive. It appears a **drop rate** of `10` is ok.

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-tolerance-drop),
> [output](https://gist.github.com/wolfram77/6f85edcf691aa2a044d65bdfda464eba), or
> [sheets][sheets-o3].

[![](https://i.imgur.com/JXclQqM.png)][sheets-o3]
[![](https://i.imgur.com/Z6XuF6m.png)][sheets-o3]

[sheets-o3]: https://docs.google.com/spreadsheets/d/1JWPDWDyF_5Spa8pkBAzJBYBMWJRe1rVZ1JWBZsQ2r44/edit?usp=sharing


#### Adjusting (initial) tolerance

As mentioned above, we can start with an initial *high* tolerance. It appears a
**tolerance** of `0.01` is suitable.

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-tolerance),
> [output](https://gist.github.com/wolfram77/f293e46c52a8d6baa53b2e78ddc202a1), or
> [sheets][sheets-o4].

[![](https://i.imgur.com/DpEHE5V.png)][sheets-o4]
[![](https://i.imgur.com/j1sQhpd.png)][sheets-o4]

[sheets-o4]: https://docs.google.com/spreadsheets/d/1PhO_jF32Slm7RNF9jUeiNgRRLiXHqBA82iHICmqvDS4/edit?usp=sharing


#### Adjusting aggregation tolerance

This controls when to consider communities to have converged based on the number
of community merges. That is if too few communties merged this pass, we should
stop here. Or, `|Vsuper-vertices|/|V| >= aggegation tolerance` => Converged.

It appears an **aggregation tolerance** of `0.8` might be the best choice.

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-aggregation-tolerance),
> [output](https://gist.github.com/wolfram77/5fd2c13f43ddcb886e7b92bbe8e727b8), or
> [sheets][sheets-o5].

[![](https://i.imgur.com/2VJGnD5.png)][sheets-o5]
[![](https://i.imgur.com/lCsdhp2.png)][sheets-o5]

[sheets-o5]: https://docs.google.com/spreadsheets/d/19YOb5lr18jWbhxm8mZr_ikdeio8PRSrHcR3IsVw2W3Y/edit?usp=sharing


#### Vertex pruning

When a vertex changes its community, its marks its neighbors to be processed.
Once a vertex has been processed, it is marked as not to be processed. It helps
*minimize unnecessary computation*.

It appears with **vertex pruning** we get *better* timings.

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-pruning),
> [output](https://gist.github.com/wolfram77/e9de07d5a989c5ca744310b3abca76c4), or
> [sheets][sheets-o6].

[![](https://i.imgur.com/NHurIeo.png)][sheets-o6]
[![](https://i.imgur.com/H7y7M6G.png)][sheets-o6]

[sheets-o6]: https://docs.google.com/spreadsheets/d/1vwwavpvs58U6_z9rpqiJ6VCYHq-MS6lvPBmDYQlsaIA/edit?usp=sharing


#### Finding community vertices for aggregation phase

I originally used a simple *vector2d* for storing vertices belonging to each
community. But this *requires memory allocation* during the algorithm, which is
expensive. Using a *prefix sum* along with a *preallocated CSR* helps to to
*avoid* repeated *mallocs* and *frees*.

It appears using **prefix sum** is quite *faster* than using vector2d.

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/approach-communities-vector2d),
> [output](https://gist.github.com/wolfram77/0ab99132ad5af767f266df95a00192ec), or
> [sheets][sheets-o7].

[![](https://i.imgur.com/KT1AGmy.png)][sheets-o7]
[![](https://i.imgur.com/kZq5AYu.png)][sheets-o7]

[sheets-o7]: https://docs.google.com/spreadsheets/d/1ItKTvGPtCVhYmb3RmURXa56RHn5c6o12sE3dHISv7Xo/edit?usp=sharing


#### Storing aggregated communities (super-vertex graph)

I also originally used a *vector2d* based dynamic graph data structure for
storing the super-vertex graph. Again, this *requires memory allocation* during
the algorithm. Using two *preallocated CSRs* along with parallel *prefix sum* can
help here.

It appears using **preallocated CSRs** is *faster*.

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/approach-aggregate-csr),
> [output](https://gist.github.com/wolfram77/72bb8d766ad590544e72253874e264f8), or
> [sheets][sheets-o8].

[![](https://i.imgur.com/ZUQVZGX.png)][sheets-o8]
[![](https://i.imgur.com/WLXrFnX.png)][sheets-o8]

[sheets-o8]: https://docs.google.com/spreadsheets/d/1mYgXqGYeU3D4FMnjUitT2RxsXdepYCVc5VTwfK9Oed8/edit?usp=sharing


#### Hashtable design for local-moving/aggregation phases

One can use `std::map`s (C++ inbuilt map) as the hastable for *Louvain*. But
this has poor performance. So i use a **key-list** and a **full-size values**
**vector** (*collision-free*) we can dramatically improve performance. However if
the memory addresses of the hastables are **nearby**, even if each thread uses
its own hash table performance is not so high possibly due to false
cache-sharing (**Close-KV**). However if i ensure memory address are farther
away, the perf. improves (**Far-KV**).

It seems **Far-KV** has the best performance.

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-hashtable),
> [output](https://gist.github.com/wolfram77/5b0addb1a69d5a9212af63cfa45f6fa9), or
> [sheets][sheets-o9].

[![](https://i.imgur.com/ALrS6iN.png)][sheets-o9]
[![](https://i.imgur.com/Jlbqpa9.png)][sheets-o9]

[sheets-o9]: https://docs.google.com/spreadsheets/d/1EeZM4MSfLe9mmL7potgplUxpu5owLCBhlUp1QgD1vLg/edit?usp=sharing

<br>
<br>


### Main results

We combine the above *optimizations* and observe the performance of
**OpenMP-based Louvain** on a number of graphs. Observe that Static
Louvain completes in *6.2 s* on `sk-2005` (a graph with *3.8 billion edges*).

> See
> [code](https://github.com/puzzlef/louvain-communities-openmp/tree/input-large),
> [output](https://gist.github.com/wolfram77/a1159c17bd2e7093e8504bddef472d79), or
> [sheets].

[![](https://i.imgur.com/2vBkuUf.png)][sheets]
[![](https://i.imgur.com/LydDEfC.png)][sheets]

*Larger* number of *iterations/passes* are required for graphs with *lower average*
*degree* (road networks such as `asia_osm` and `europe_osm`, and protein k-mer
graphs such as `kmer_A2a` and `kmer_V1r`).

[![](https://i.imgur.com/S1GI8An.png)][sheets]
[![](https://i.imgur.com/IMRK6nU.png)][sheets]

Also note how graphs with **poor community structure** (such as `com-LiveJournal`,
`com-Orkut`) have a **larger** `time / (n log n)` factor.

[![](https://i.imgur.com/HGzTEEO.png)][sheets]

Finally note how **most** of the **runtime** of the algorithm is spent in the
**local-moving phase**. Further, most of the runtime is spent in the **first**
**pass** of the algorithm, which is the most expensive pass due to the size of the
original graph (later passes work on super-vertex graphs).

[![](https://i.imgur.com/78x6g2d.png)][sheets]
[![](https://i.imgur.com/ngw119X.png)][sheets]

The input data used for this experiment is available from the
[SuiteSparse Matrix Collection]. This experiment was done with guidance
from [Prof. Kishore Kothapalli] and [Prof. Dip Sankar Banerjee].


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[sheets]: https://docs.google.com/spreadsheets/d/1nHwee7pVpqOenTMlnfNfPvtt91-jttPpPOHwLOWoAhA/edit?usp=sharing

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


[![](https://img.youtube.com/vi/M6npDdVGue4/maxresdefault.jpg)](https://www.youtube.com/watch?v=M6npDdVGue4)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
