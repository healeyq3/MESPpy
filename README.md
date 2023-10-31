# Maximum-Entropy Sampling

## Overview

Updated 10/31/23

See test2.py - framework is solving instance s=10 with (from as far as I'm aware) fastest runtime seen. The instance s=20 is also only solved a bit slower than Kurt's runtime.

The instance s=30 was not solved in under an hour.

## Todos
> Not relevant to anyone but me (Quill) - this is just a convenient place to house different ideas and next steps for this research.

### Performance (Research) Related
- Implement dual branching strategy found in Kurt (This might be working now...I believe it was not working before due to improper parameter passing).
- Look through Kurt's paper again and see how often they use variable fixing. For the larger problem instances would it make since to fix more variables once the problem is reduced enough. Also, how would this work?  => If branched then we've reduced $C$ to some $\hat{C}$. Could then pass the associated matrices with $n'$ and $s'$ to localsearch.
- Look at connection between dual branching strategy and variable fixing.
- Is there a way to salvage $V, V^{2}, E$ from $\hat{C}$ when fixing variables out of $C$.
- Think about and discuss with Dr. Xie interesting result - s=20 was solved to optimality with $s=10$ fixed out variables and $s=20$ fixed in variable. Runtime was under 8 minutes.
- Ensure variable fixing working for $s=50$.
- More solver attribute/statistic tracking
    - "time to opt" statistic (once the worst upper bound is within $\varepsilon$)
    - if termination early, be able to enumerate remainder of open nodes and determine how close your upper bound is
    - count the enumeration of nodes performed, but be sure to differentiate between that number and number of nodes actually solved
    - break up what solve is returning and instead have getters (see dataclasses below)

### Solver Build-out Related
- Use dataclasses for mesp file to only allow future users to interface with certain behavior
- Proper experimentation folder
    - built-in excel file generation, so can run tests without fear of overwriting results. Also good to move experiments out of top-level directory.
- Good code hygiene
    - type hints
    - refactor functions - break up more?
    - docstrings with brief mathematical description
- Build some tests for functions
    - potentially help understand the bounding and approximation Yongchun contributed

### Further Off
- Replicate Kurt's node count prediction  