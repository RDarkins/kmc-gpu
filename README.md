# kmc-gpu

This C++/CUDA code implements an exact GPU-based algorithm for performing on-lattice kinetic Monte Carlo (KMC). One, two and three spatial dimensions are supported, although the acceleration is most pronounced for two dimensions.

The algorithm uses the waiting time method to perform rejection-free KMC over dynamically chosen time steps. It is parallelised using domain decomposition, and exactness is achieved by rejecting time steps that introduce errors detected through a consistency check.

A CPU-based implementation of the BKL algorithm is included for benchmarking purposes.

