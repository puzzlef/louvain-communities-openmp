Effect of adjusting threads with OpenMP-based [Louvain algorithm] for
[community detection].

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

In this experiment, we perform OpenMP-based Louvain algorithm with the number of
threads varying from `2` to `48` in steps of `2` (the system has 2 CPUs with 12
cores each and 2 hyper-threads per core). Each thread uses its own accumulation
hashtable in order to find the most suitable community to join to, for each
vertex. Once the best community is found, the community membership of that
vertex is updated, along with the total weight of each community. This update
can affect the community moving decision of other vertices (ordered approach).
We choose the Louvain *parameters* as `resolution = 1.0`, `tolerance = 1e-2`
(for local-moving phase) with *tolerance* decreasing after every pass by a
factor of `toleranceDeclineFactor = 10`, and a `passTolerance = 0.0` (when
passes stop). In addition we limit the maximum number of iterations in a single
local-moving phase with `maxIterations = 500`, and limit the maximum number of
passes with `maxPasses = 500`. We run the Louvain algorithm until convergence
(or until the maximum limits are exceeded), and measure the **time taken** for
the *computation* (performed 5 times for averaging), the **modularity score**,
the **total number of iterations** (in the *local-moving phase*), and the number
of **passes**. This is repeated for *seventeen* different graphs.

From the results, we make make the following observations. **Increasing the number**
of threads **only decreases the runtime** of the Louvain algorithm **by a small**
**amount**. Utilizing all 48 threads for community detection significantly increases
the time required for obtaining the results. The number of iterations required
to converge also increases with the number of threads, indicating that the
behaviour of OpenMP-based Louvain algorithm starts to approach the unordered
variant of the algorithm, which converges mush more slowly than the ordered
variant.

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
# [-0.000497 modularity] noop
# [00448.766 ms; 0025 iters.; 009 passes; 0.923382580 modularity] louvainSeq
# [00523.509 ms; 0025 iters.; 009 passes; 0.923540175 modularity] louvainOmp {threads=02}
# [00424.414 ms; 0025 iters.; 009 passes; 0.923708200 modularity] louvainOmp {threads=04}
# [00410.485 ms; 0026 iters.; 009 passes; 0.923715591 modularity] louvainOmp {threads=06}
# [00422.342 ms; 0034 iters.; 009 passes; 0.923617899 modularity] louvainOmp {threads=08}
# ...
# [00265.765 ms; 0038 iters.; 009 passes; 0.927356064 modularity] louvainOmp {threads=42}
# [00252.321 ms; 0040 iters.; 009 passes; 0.927225232 modularity] louvainOmp {threads=44}
# [00253.382 ms; 0074 iters.; 009 passes; 0.927367568 modularity] louvainOmp {threads=46}
# [00276.118 ms; 0032 iters.; 009 passes; 0.927250326 modularity] louvainOmp {threads=48}
#
# Loading graph /home/subhajit/data/web-BerkStan.mtx ...
# order: 685230 size: 7600595 [directed] {}
# order: 685230 size: 13298940 [directed] {} (symmetricize)
# [-0.000316 modularity] noop
# [00743.776 ms; 0028 iters.; 009 passes; 0.935839474 modularity] louvainSeq
# [00964.372 ms; 0028 iters.; 009 passes; 0.935696840 modularity] louvainOmp {threads=02}
# [00798.594 ms; 0029 iters.; 009 passes; 0.935616016 modularity] louvainOmp {threads=04}
# [01217.822 ms; 0029 iters.; 009 passes; 0.935623407 modularity] louvainOmp {threads=06}
# [00981.672 ms; 0029 iters.; 009 passes; 0.934371948 modularity] louvainOmp {threads=08}
# ...
```

[![](https://i.imgur.com/SxNaJki.png)][sheetp]
[![](https://i.imgur.com/jhmxA9c.png)][sheetp]
[![](https://i.imgur.com/Ys0tife.png)][sheetp]
[![](https://i.imgur.com/MqAypz2.png)][sheetp]

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

[![](https://i.imgur.com/7LbyT2c.jpg)](https://www.youtube.com/watch?v=kQZ2nA2GpXw)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/542910968.svg)](https://zenodo.org/badge/latestdoi/542910968)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[Louvain]: https://en.wikipedia.org/wiki/Louvain_method
[gist]: https://gist.github.com/wolfram77/52e3228bf8aaf0342681cc821eb1e13d
[charts]: https://imgur.com/a/0Urw7Tj
[sheets]: https://docs.google.com/spreadsheets/d/1Ghp4B9I121mjWtAES9jK6C3WoBJQ2b3l6xeLsr5HMh0/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vR7szglMFn_31IVDz5LrlcUg_9TgvIhtJAW-XgLdbRcM9mXPUE3IdEr8rSd-DbsPTMYIQ_i5iNint7D/pubhtml
