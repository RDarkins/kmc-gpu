# kmc-gpu

GPU-based algorithm for on-lattice kinetic Monte Carlo without approximations, as described here:

[Exact kinetic Monte Carlo in two dimensions on a GPU](preprint.pdf) <br />
Robert Darkins, Dorothy M. Duffy, Ian J. Ford <br />
University College London

The algorithm uses the waiting time method to perform rejection-free KMC over dynamically chosen time steps. It is parallelised using domain decomposition, and exactness is achieved by rejecting time steps that introduce errors detected through a consistency check. The code supports one, two and three dimensions, but the acceleration is most pronounced for two dimensions.
