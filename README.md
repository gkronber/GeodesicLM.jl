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

## Inner-loop use / allocations

`geodesiclm` reuses a pre-allocated `GLMWorkspace` (all internal buffers) via a
OncePerTask lazy cache, so repeated calls on the same task allocate almost
nothing (the residual is mostly the Cholesky factor). This makes it suitable for
calling from a hot loop; different Julia tasks each get their own workspace, so
it is thread-safe.

```julia
x   = [0.0, 0.0]
fvec = zeros(2)
for _ in 1:10_000
    fill!(x, 0); fill!(fvec, 0)
    geodesiclm(my_residual!, nothing, nothing; x=x, fvec=fvec, n=2, m=2)
end
```

You may also construct and pass a `GLMWorkspace(n, m)` explicitly via the `ws`
keyword to share buffers with other code (e.g. an analytic Jacobian scratch):

```julia
ws = GLMWorkspace(n, m)
geodesiclm(..., n=n, m=m, ws=ws)
```

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