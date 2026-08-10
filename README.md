This repository holds a Julia conversion of the geodesicLM algorithm for nonlinear least squares optimization, originally implemented in Fortran by Mark Transtrum.

The initial version of the Julia code was created automatically by github copilot, followed by extensive manual refinement to ensure accuracy, performance, and idiomatic Julia style.

## Testing

Run the unit test suite with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

or directly:

```bash
julia --project=. test/runtests.jl
```

The tests cover the main optimizer and its helper routines (finite-difference
Jacobian/acceleration, acceptance criterion, lambda/delta updates, trust-region
solver, smallest-singular-value estimator, machine parameters, convergence
codes), across different solver configurations (analytic vs. finite-difference
derivatives, all update methods, damping modes, bold acceptance criteria).

## Benchmarking

Performance across git revisions is tracked with
[AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl).
The benchmark suite lives in [`benchmark/`](benchmark/); see
[`benchmark/README.md`](benchmark/README.md) for how to run it.

Relevant documents:
- [QUICK_START.md](src/QUICK_START.md): A quick start guide for using the Julia module
- [README_JULIA.md](src/README_JULIA.md): Overview of the Julia conversion
- [CONVERSION_SUMMARY.md](src/CONVERSION_SUMMARY.md): Summary of the conversion process
- [CONVERSION_MAPPING.md](src/CONVERSION_MAPPING.md): Technical details of the conversion process
- [geodesiclm_alg.jl](src/geodesiclm_alg.jl): Main optimization algorithm in Julia
- [GeodesicLM.jl](src/GeodesicLM.jl): Module wrapper for easy usage