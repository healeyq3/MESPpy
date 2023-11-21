# MESPpy
This repository provides a declarative language for formulating, approximating, solving, and experimenting with the **maximum entropy sampling problem** (MESP).

**Contents**
- [The Maximum Entropy Sampling Problem](#the-maximum-entropy-sampling-problem)
- [Associated Research](#associated-research)
- [Using this repository](#using-mesppy)
- [More on Maximum Entropy Sampling](#maximum-entropy-sampling-more-broadly)

## The Maximum Entropy Sampling Problem
We'll take four passes at explaining this problem.

**Intuitively,** the MESP aims to select some (prespecified) subset of random observations from a set of candidate observations to maximize the information obtained. *Usually,* in practice, we are looking to select a small subset from a large candidate set.

**More mathematically,** "in the maximum entropy sampling problem we are looking to find a maximum (log-)determinant principal submatrix $C_{S, S}$ of a given order $s$ from an input positive-semidefinite matrix $C$ of order $n > s$." [[1]](#1)

**Most thoroughly,** suppose we have $n$ random variables generated from a multivariate elliptical (but for our purposes, assume Gaussian) distribution. Given the covariance matrix, $C \in \mathbb{S}_{+}^{n}$, associated with these random variables, the maximum entropy sampling problem seeks to find, for some integer $0 < s \le \min \{ \textbf{rank}\, C, n - 1 \}$, the subset of random variables, $S$ such that $\left| S  \right| = s$, which maximizes the log determinant of $C$ (this is equivalent to selecting the subset of variables with the greatest differential entropy).

**Most mathematically,** the MESP is the following combinatorial optimization problem

$$
\begin{align*}
\text{maximize} \; \{ \textbf{ldet} \, C_{S, S} : S \subseteq [n], \left| S \right| = s \}.
\end{align*}
$$


### What MESP is NOT
We want to emphasize the distinction between the **principle of maximum entropy** (equivalently, the *entropy maximization problem*) and the **maximum entropy sampling problem**. Mathematically, the former can be expressed as 
$$
\begin{equation*}
\begin{array}{lll}
\text{maximize} & \sum_{i =1}^{n}x_i \log x_i & \\
\text{subject to} & Ax \preceq b \\
& \mathbf{1}^Tx = 1,
\end{array}
\end{equation*}
$$

while we already stated the MESP formulation above. We felt the need to explicitly state the difference between the two problems as clearly the entropy maximization problem is convex (this particular example is found in [[2]](#2)), and thus can already be formulated and solved using CVXPY and some backend solver.

## Associated Research
MESPpy aims to push the boundaries of the maximum entropy sampling problem in the following three ways.
1. **(Performance)** Using the bounds proposed in [[3]](#3) with novel branching strategies and numerical implementations, MESPpy aims to solve (and does solve) to optimality instances of the MESP faster than the best known runtimes shown in [[4]](#4). See [current results](#current-results) for more details regarding the state of our exact solver on common MESP datasets.
2. **(Accessibility)** As far as we are aware, there is no existing (open source or commercial) package which easily allows practitioners to approximately or globally solve their own maximum entropy sampling problems. MESPpy provides a declarative language which gives easy access to the accurate and scalable approximation algorithms found in [[3]](#3). Of course, it also allows access to our exact solver, however, a potential user is recommended to see our [current results](#current-results) for guidance on whether or not solving their problem to optimality is tractable.
3. **(Experimentation)** Finally, MESPpy hopes to encourage further research on the maximum entropy sampling problem by 
    1. Allowing more advanced practitioners to incorporate *their own bounding algorithms* and even *combination of bounding algorithms* (configured based on problem size and chosen submatrix order) by simply interfacing with our ```BoundChooser``` class. In other words, researchers can leverage our branch-and-bound framework to test how their bounding algorithms perform attempting to solve a MESP to optimality.
    2. Providing code templates to generate results for a user specified collection of $s$ values on any valid MESP.
    3. Making our code open source, providing a building block for others to attempt their own branching (and other performance related) strategies.
    

**Contributors.**
This work was conducted under [Dr. Weijun Xie's](https://sites.google.com/site/weijunxieor/home) supervision and was contributed to by PhD candidate Yongchun Li and undergraduate student Quill Healey. Specifically, Yongchun is responsible for the approximation and bounding algorithms, which she and Dr. Xie developed in ["Best Best Principal Submatrix Selection for the Maximum Entropy Sampling Problem:
Scalable Algorithms and Performance Guarantees"](https://arxiv.org/pdf/2001.08537.pdf) with the associated [codebase](https://github.com/yongchunli-13/Approximation-Algorithms-for-MESP). Quill is responsible for the creation of this declarative language driven repository, the branch and bound framework, and the related experimentation features (of course, with the guidance and help of both Dr. Xie and Yongchun). 

## Using MESPpy

> Mention the ```MespData``` object

### Basic

### Advanced

## Maximum Entropy Sampling (more broadly)

### Motivating Examples

## References
<a id="1">[1]</a>: Maximum-Entropy Sampling Algorithms and Application, Fampa and Lee.

<a id="2">[2]</a> : Convex Optimization, Vandenberghe and Boyd

<a id="3">[3]</a> : Best Best Principal Submatrix Selection for the Maximum Entropy Sampling Problem: Scalable Algorithms and Performance Guarantees, Li and Xie

<a id="4">[4]</a> : Efficient Solution of Maximum-Entropy Sampling Problems, Anstreicher

## Current Results
>Expand on how we've improved the runtime performance so far.

## Repository Explanations

> Add explanations for certain branching strategies and numerical approaches we use.

## Repository Status

Updated 11/20/23

Currently in the progress of a MAJOR OVERHAUL. I do not recommend experimenting with this repository right now.

## Todos
> Not relevant to anyone but me (Quill) - this is just a convenient place to house different ideas and next steps for this research.

### Performance (Research) Related
- **IMPORTANT** Switch to the global UB strategy proposed in MESP book (high hopes for this)
- Finish creating the ```BoundChooser``` class, including updating where the node class calls the solver
- Finish creating the three types of nodes (NonShrinkingNode, IterativeNode, BatchNode), but focus on the first two
- More solver attribute/statistic tracking
    - "time to opt" statistic (once the worst upper bound is within $\varepsilon$)
    - if termination early, be able to enumerate remainder of open nodes and determine how close your upper bound is
    - count the enumeration of nodes performed, but be sure to differentiate between that number and number of nodes actually solved
    - break up what solve is returning and instead have getters (see dataclasses below)

### Solver Build-out Related
- Use dataclasses for mesp file to only allow future users to interface with certain behavior 
- Tree and Mesp statistics getters for experimentation
- Proper experimentation folder
    - built-in excel file generation, so can run tests without fear of overwriting results. Also good to move experiments out of top-level directory.
- More type checks to help with the ease of using MESPpy as a declarative language
- Good code hygiene
    - type hints
    - refactor functions - break up more?
    - docstrings with brief mathematical description
    - Tests 