Effect of adjusting schedule in OpenMP-based Louvain algorithm for community
detection.

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

There exist two possible approaches of vertex processing with the Louvain
algorithm: *ordered* and *unordered*. With the **ordered approach** (original
paper's approach), the *local-moving phase* is **performed sequentially** upon
each vertex such that the moving of a *previous vertex* in the graph *affects*
the decision of the *current vertex* being processed. On the other hand, with
the **unordered approach** the moving of a *previous vertex* in the graph *does*
*not affect* the decision of movement for the *current vertex*. This *unordered*
*approach* (aka *relaxed approach*) is made possible by maintaining the
*previous* and the *current community membership status* of each vertex (along
with associated community information), and is the approach **followed by**
**parallel Louvain implementation** on the CPU as well as the GPU. We are
interested in looking at *performance/modularity penalty* (if any) associated
with the *unordered approach*.

In this experiment we attempt performing the **OpenMP-based Louvain algorithm**
using the **ordered approach** with all different **schedule kinds** (`static`,
`dynamic`, `guided`, and `auto`), which adjusting the **chunk size** from `1` to
`65536` in multiples of 2. In all cases, we use a total of `12 threads`. We
choose the Louvain *parameters* as `resolution = 1.0`, `tolerance = 1e-2` (for
local-moving phase) with *tolerance* decreasing after every pass by a factor of
`toleranceDeclineFactor = 10`, and a `passTolerance = 0.0` (when passes stop).
In addition we limit the maximum number of iterations in a single local-moving
phase with `maxIterations = 500`, and limit the maximum number of passes with
`maxPasses = 500`. We run the Louvain algorithm until convergence (or until the
maximum limits are exceeded), and measure the **time** **taken** for the
*computation* (performed 5 times for averaging), the **modularity score**, the
**total number of iterations** (in the *local-moving* *phase*), and the number
of **passes**. This is repeated for *seventeen* different graphs.

From the results, we observe that little to **no benefit** is provided with a
multi-threaded **OpenMP-based ordered** *Louvain algorithm*, compared to the
sequential ordered approach. This is somewhat surprising (not surprising to me)
and indicates that when *multiple reader threads* and *a writer thread* access
*common memory locations* (here it is the community membership of each vertex)
perfomance is degraded and would tend to approach the performance of a
sequential algorithm if there is just too much overlap. What should you do in
that case? A simple approach would be to just **go ahead with the sequential**
**(ordered) Louvain algorithm**, **or partition the graph** is such a way that
each partition can be run independently, **and then combined back together**. We
could also try this same experiment with the *unordered approach*, and see if it
performs better than the ordered sequential approach (the ordered approach is an
order of magnitude slower than the unordered approach, but is easily
parallelizable). **Partially ordered approaches** for vertex processing may also
be explored. *Vertex ordering* via *graph coloring* has been explored by
Halappanavar et al.

All outputs are saved in a [gist] and a small part of the output is listed here.
Some [charts] are also included below, generated from [sheets]. The input data
used for this experiment is available from the [SuiteSparse Matrix Collection].
This experiment was done with guidance from [Prof. Kishore Kothapalli] and
[Prof. Dip Sankar Banerjee].

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
# [00636.194 ms; 0025 iterations; 009 passes; 0.923383 modularity] louvainSeq
# [00698.879 ms; 0026 iterations; 009 passes; 0.923119 modularity] louvainOmp {sch_kind: static, chunk_size: 1}
# [00655.290 ms; 0025 iterations; 009 passes; 0.922996 modularity] louvainOmp {sch_kind: static, chunk_size: 2}
# [00661.412 ms; 0026 iterations; 009 passes; 0.923447 modularity] louvainOmp {sch_kind: static, chunk_size: 4}
# ...
# [00617.610 ms; 0025 iterations; 009 passes; 0.927243 modularity] louvainOmp {sch_kind: auto, chunk_size: 16384}
# [00620.235 ms; 0025 iterations; 009 passes; 0.923699 modularity] louvainOmp {sch_kind: auto, chunk_size: 32768}
# [00614.724 ms; 0023 iterations; 009 passes; 0.923273 modularity] louvainOmp {sch_kind: auto, chunk_size: 65536}
#
# Loading graph /home/subhajit/data/web-BerkStan.mtx ...
# order: 685230 size: 7600595 [directed] {}
# order: 685230 size: 13298940 [directed] {} (symmetricize)
# OMP_NUM_THREADS=12
# [-0.000316 modularity] noop
# [01208.700 ms; 0028 iterations; 009 passes; 0.935839 modularity] louvainSeq
# [01747.514 ms; 0027 iterations; 009 passes; 0.938295 modularity] louvainOmp {sch_kind: static, chunk_size: 1}
# [01569.095 ms; 0025 iterations; 009 passes; 0.933566 modularity] louvainOmp {sch_kind: static, chunk_size: 2}
# [01541.691 ms; 0025 iterations; 009 passes; 0.938013 modularity] louvainOmp {sch_kind: static, chunk_size: 4}
# ...
```

[![](https://i.imgur.com/9JVW1Au.png)][sheetp]
[![](https://i.imgur.com/sY4bEZz.png)][sheetp]
[![](https://i.imgur.com/BbpUupy.png)][sheetp]
[![](https://i.imgur.com/JIqnZjh.png)][sheetp]

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

[![](https://i.imgur.com/b4TCcWX.jpg)](https://www.youtube.com/watch?v=M6npDdVGue4)<br>
[![DOI](https://zenodo.org/badge/519156419.svg)](https://zenodo.org/badge/latestdoi/519156419)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[Louvain]: https://en.wikipedia.org/wiki/Louvain_method
[gist]: https://gist.github.com/wolfram77/07d31e40dee392a1860edfb24d35943b
[charts]: https://imgur.com/a/FntD3KO
[sheets]: https://docs.google.com/spreadsheets/d/1aMGHE5KtHl30qvDH0Sq1W46ZuNssbs0FKiQyaUFtkMg/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3DMyRo-a7EA3XrY4mR1ABCo5kscSnOG9M6UCD_7MIlr2UltaSrAJ6eTMqNEL-BZjP5t8BbthQYzb9/pubhtml
