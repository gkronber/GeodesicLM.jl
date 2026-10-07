# GeodesicLM.jl

A Julia implementation of the geodesic Levenberg-Marquardt algorithm for
nonlinear least squares, converted from the Fortran package
[geodesicLM](https://sourceforge.net/projects/geodesiclm/) by Mark K. Transtrum.
It adds geodesic acceleration, a bold acceptance criterion and Broyden updates
of the Jacobian to Levenberg-Marquardt, works in any floating-point precision,
and is allocation-free when a workspace is reused across fits.

The initial version of the Julia code was created automatically by github
copilot, followed by extensive manual refinement for accuracy, performance and
idiomatic Julia style.  The original Fortran sources and their license are kept
in [`original/`](original/) for reference.

## Installation

GeodesicLM is not registered; with Julia 1.13 or newer install it from its git
repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/gkronber/GeodesicLM.jl")
```

## Usage

`geodesiclm` minimizes `½ Σᵢ fᵢ(x)²`.  The residual function writes the `m`
residuals for the `n` parameters `x` into `fvec`:

```julia
using GeodesicLM

# Fit y = a * exp(-b * t) to data
t = collect(0.0:0.5:5.0)
y = 2.0 .* exp.(-0.7 .* t)
residuals!(x, fvec) = (fvec .= x[1] .* exp.(-x[2] .* t) .- y)

x = [1.0, 1.0]                 # initial guess, overwritten with the solution
fvec = zeros(length(t))
x, fvec, niters, nfev, njev, naev, converged =
    geodesiclm(residuals!, nothing, nothing; x, fvec, n = 2, m = length(t))
# x ≈ [2.0, 0.7], converged == 2 (the cost goal was reached)
```

The second and third arguments are an optional analytic Jacobian
`jacobian!(x, fjac)` (with `analytic_jac = true`) and second directional
derivative `Avv!(x, v, acc)` (with `analytic_Avv = true`); without them both are
computed by finite differences.  Pass `workspace = GeodesicLMWorkspace{T}(m, n)`
to reuse the memory across fits, and `callback` to monitor the iterations; a
callback that returns a nonzero value stops the fit.

The keyword options (convergence tolerances, damping, update method, acceptance
criterion, Broyden updates, printing) and their defaults are documented in the
docstring, `?geodesiclm`.  The tolerances default to precision-dependent values
(see `GeodesicLM.default_tolerance`) that equal those of the reference
implementation in `Float64`.

### Convergence codes

| `converged` | Reason |
|---|---|
| 1 | angle between residuals and tangent plane below `artol` |
| 2 | cost below `Cgoal` |
| 3 | gradient norm below `gtol` |
| 4 | step size below `xtol` |
| 5 | relative parameter change below `xrtol` |
| 6 | cost decrease below `ftol` three times in a row |
| 7 | relative cost decrease below `frtol` three times in a row |
| -1 | `maxiter` iterations reached |
| -2, -3, -4 | `maxfev`, `maxjev` or `maxaev` evaluations reached |
| -5, -6 | damping parameter reached `maxlam` or `minlam` |
| -10 | stopped by the callback |
| -11 | NaN in the initial residuals or in the Jacobian |

## Testing

Run the unit test suite with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
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

## References

If you use this code, please cite

- Transtrum M.K., Machta B.B., and Sethna J.P., "Why are nonlinear fits to data
  so challenging?" Phys. Rev. Lett. 104, 060201 (2010)
- Transtrum M.K., Machta B.B., and Sethna J.P., "The geometry of nonlinear least
  squares with applications to sloppy models and optimization," Phys. Rev. E 83,
  036701 (2011)
