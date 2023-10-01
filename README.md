# Maximum-Entropy Sampling

Updated 9/30/23

To be filled in.

Note that I have not yet refactored/reincorporated the Branch-and-Bound framework - I will start incrementally adding that, but most likely will not finish it until this Wednesday 10/04. Along with that I'll work on the problem size shrinking code, which I do think I should handle since it is intrisically tied to the tree behavior. I'm hoping that between the repo refactoring, code streamlining, and addition of the subproblem reduction we will see a good performance jump. From there it will be easy to start playing with branching rules and building out the problem size estimation that Kurt uses.

Please see test.py for an example of the current functionality.