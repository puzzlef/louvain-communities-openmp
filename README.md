Comparing approaches for *community detection* using *OpenMP-based* **Louvain algorithm**.

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

The input data used for the experiments given below is available from the
[SuiteSparse Matrix Collection]. These experiments are performed with guidance
from [Prof. Kishore Kothapalli] and [Prof. Dip Sankar Banerjee].

<br>


### Adjusting OpenMP schedule (Ordered approach)

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

In this experiment ([adjust-schedule]), we attempt performing the **OpenMP-based**
**Louvain algorithm** using the **ordered approach** with all different **schedule**
**kinds** (`static`, `dynamic`, `guided`, and `auto`), which adjusting the **chunk**
**size** from `1` to `65536` in multiples of 2. In all cases, we use a total of
`12 threads`. We choose the Louvain *parameters* as `resolution = 1.0`,
`tolerance = 1e-2` (for local-moving phase) with *tolerance* decreasing after
every pass by a factor of `toleranceDeclineFactor = 10`, and a `passTolerance = 0.0`
(when passes stop). In addition we limit the maximum number of iterations
in a single local-moving phase with `maxIterations = 500`, and limit the maximum
number of passes with `maxPasses = 500`. We run the Louvain algorithm until
convergence (or until the maximum limits are exceeded), and measure the **time**
**taken** for the *computation* (performed 5 times for averaging), the
**modularity score**, the **total number of iterations** (in the *local-moving*
*phase*), and the number of **passes**. This is repeated for *seventeen*
different graphs.

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

[adjust-schedule]: https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-schedule

<br>


### Adjusting number of threads (Strong scaling)

In this experiment ([adjust-threads]), we perform OpenMP-based Louvain algorithm
with the number of threads varying from `2` to `48` in steps of `2` (the system
has 2 CPUs with 12 cores each and 2 hyper-threads per core). Each thread uses
its own accumulation hashtable in order to find the most suitable community to
join to, for each vertex. Once the best community is found, the community
membership of that vertex is updated, along with the total weight of each
community. This update can affect the community moving decision of other
vertices (ordered approach). We choose the Louvain *parameters* as `resolution = 1.0`,
`tolerance = 1e-2` (for local-moving phase) with *tolerance* decreasing
after every pass by a factor of `toleranceDeclineFactor = 10`, and a
`passTolerance = 0.0` (when passes stop). In addition we limit the maximum
number of iterations in a single local-moving phase with `maxIterations = 500`,
and limit the maximum number of passes with `maxPasses = 500`. We run the
Louvain algorithm until convergence (or until the maximum limits are exceeded),
and measure the **time taken** for the *computation* (performed 5 times for
averaging), the **modularity score**, the **total number of iterations** (in the
*local-moving phase*), and the number of **passes**. This is repeated for
*seventeen* different graphs.

From the results, we make make the following observations. **Increasing the number**
of threads **only decreases the runtime** of the Louvain algorithm **by a small**
**amount**. Utilizing all 48 threads for community detection significantly increases
the time required for obtaining the results. The number of iterations required
to converge also increases with the number of threads, indicating that the
behaviour of OpenMP-based Louvain algorithm starts to approach the unordered
variant of the algorithm, which converges mush more slowly than the ordered
variant.

[adjust-threads]: https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-threads

<br>


### Other experiments

- [adjust-schedule-unordered](https://github.com/puzzlef/louvain-communities-openmp/tree/adjust-schedule-unordered)

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
[![DOI](https://zenodo.org/badge/519156419.svg)](https://zenodo.org/badge/latestdoi/519156419)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[Louvain]: https://en.wikipedia.org/wiki/Louvain_method
