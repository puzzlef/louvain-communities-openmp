Multi-threaded OpenMP-based [Louvain] algorithm for [community detection].

Recent advancements in data collection and graph representations have led to unprecedented levels of complexity, demanding efficient parallel algorithms for community detection on large networks. The use of multicore/shared memory setups is crucial for energy efficiency and compatibility with extensive DRAM sizes. However, existing community detection algorithms face challenges in parallelization due to their irregular and inherently sequential nature. While studies on the Louvain algorithm propose optimizations and parallelization techniques, they often neglect the aggregation phase, creating a bottleneck even after optimizing the local-moving phase. Additionally, these optimization techniques are scattered across multiple papers, making it challenging for readers to grasp and implement them effectively. To address this, we introduce **GVE-Louvain**, an optimized *parallel implementation of Louvain* for shared memory multicores.

Below we plot the time taken by [Vite] (Louvain), [Grappolo] (Louvain), [NetworKit] Louvain, and GVE-Louvain on 13 different graphs. GVE-Louvain surpasses Vite, Grappolo, and NetworKit by `50×`, `22×`, and `20×` respectively, achieving a processing rate of `560𝑀` edges/s on a `3.8𝐵` edge graph.

[![](https://i.imgur.com/nVvNACt.png)][sheets-o1]

Below we plot the speedup of GVE-Louvain wrt Vite, Grappolo, and NetworKit.

[![](https://i.imgur.com/6KzkyY8.png)][sheets-o1]

Next, we plot the modularity of communities identified by Vite, Grappolo, NetworKit, and GVE-Louvain. GVE-Louvain on average obtains `3.1%` higher modularity than Vite (especially on web graphs), and `0.6%` lower modularity than Grappolo and NetworKit (especially on social networks with poor clustering).

[![](https://i.imgur.com/8KqgWBi.png)][sheets-o1]

Finally, we plot the strong scaling behaviour of GVE-Louvain. With doubling of threads, GVE-Louvain exhibits an average performance scaling of `1.6×`.

[![](https://i.imgur.com/GjciJ9V.png)][sheets-o2]

Refer to our technical report for more details: \
[GVE-Louvain: Fast Louvain Algorithm for Community Detection in Shared Memory Setting][report].

<br>

> [!NOTE]
> You can just copy `main.sh` to your system and run it. \
> For the code, refer to `main.cxx`.


[Louvain]: https://en.wikipedia.org/wiki/Louvain_method
[community detection]: https://en.wikipedia.org/wiki/Community_structure
[sheets-o1]: https://docs.google.com/spreadsheets/d/1aJI2Us60KXbSx9LeGyHdnuYfEg4d_5bFhiLXc9eaUjM/edit?usp=sharing
[sheets-o2]: https://docs.google.com/spreadsheets/d/1eR0jkbjoskL9K2HNVy-irnHERMV770egljh94alRT0U/edit?usp=sharing
[report]: https://arxiv.org/abs/2312.04876
[Vite]: https://github.com/ECP-ExaGraph/vite
[Grappolo]: https://github.com/ECP-ExaGraph/grappolo
[NetworKit]: https://github.com/networkit/networkit

<br>
<br>


### Code structure

The code structure of GVE-Louvain is as follows:

```bash
- inc/_algorithm.hxx: Algorithm utility functions
- inc/_bitset.hxx: Bitset manipulation functions
- inc/_cmath.hxx: Math functions
- inc/_ctypes.hxx: Data type utility functions
- inc/_cuda.hxx: CUDA utility functions
- inc/_debug.hxx: Debugging macros (LOG, ASSERT, ...)
- inc/_iostream.hxx: Input/output stream functions
- inc/_iterator.hxx: Iterator utility functions
- inc/_main.hxx: Main program header
- inc/_mpi.hxx: MPI (Message Passing Interface) utility functions
- inc/_openmp.hxx: OpenMP utility functions
- inc/_queue.hxx: Queue utility functions
- inc/_random.hxx: Random number generation functions
- inc/_string.hxx: String utility functions
- inc/_utility.hxx: Runtime measurement functions
- inc/_vector.hxx: Vector utility functions
- inc/batch.hxx: Batch update generation functions
- inc/bfs.hxx: Breadth-first search algorithms
- inc/csr.hxx: Compressed Sparse Row (CSR) data structure functions
- inc/dfs.hxx: Depth-first search algorithms
- inc/duplicate.hxx: Graph duplicating functions
- inc/Graph.hxx: Graph data structure functions
- inc/louvain.hxx: Louvain community detection algorithm functions
- inc/main.hxx: Main header
- inc/mtx.hxx: Graph file reading functions
- inc/properties.hxx: Graph Property functions
- inc/selfLoop.hxx: Graph Self-looping functions
- inc/symmetricize.hxx: Graph Symmetricization functions
- inc/transpose.hxx: Graph transpose functions
- inc/update.hxx: Update functions
- main.cxx: Experimentation code
- process.js: Node.js script for processing output logs
```

Note that each branch in this repository contains code for a specific experiment. The `main` branch contains code for the final experiment. If the intention of a branch in unclear, or if you have comments on our technical report, feel free to open an issue.

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
