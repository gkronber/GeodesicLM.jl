# GeodesicLM.jl - Julia Translation of Geodesic Levenberg-Marquardt Algorithm

## Overview

This directory contains a complete Julia translation of the geodesic Levenberg-Marquardt optimization algorithm. The original Fortran source files from Mark Transtrum (https://sourceforge.net/projects/geodesiclm/) have been converted to Julia using github copilot.

## What's Included

### Original Fortran Files (unchanged)
- `accept.f90` - Acceptance criterion
- `converge.f90` - Convergence checking
- `destsv.f` - Singular value estimation
- `dgqt.f` - Trust region solver
- `dpmpar.f` - Machine parameters
- `fdavv.f90` - Finite-difference second derivatives
- `fdjac.f90` - Finite-difference Jacobian
- `geodesiclm.f90` - Main algorithm
- `lambda.f90` - Lambda/delta update methods
- `updatejac.f90` - Jacobian updates

### New Julia Files
- **GeodesicLM.jl** - Module wrapper (main entry point)
- **geodesiclm_alg.jl** - Main optimization algorithm
- **accept.jl** - Acceptance criterion
- **converge.jl** - Convergence checking
- **destsv.jl** - Singular value estimation
- **dgqt.jl** - Trust region solver
- **dpmpar.jl** - Machine parameters
- **fdavv.jl** - Finite-difference second derivatives
- **fdjac.jl** - Finite-difference Jacobian
- **lambda.jl** - Lambda/delta update methods
- **updatejac.jl** - Jacobian updates

### Documentation Files
- **CONVERSION_SUMMARY.md** - Detailed conversion notes and technical details
- **QUICK_START.md** - Quick start guide with examples
- **README.jl** - This file

## Quick Start

```julia
# Load the module
include("GeodesicLM.jl")
using .GeodesicLM

# Define your objective function
function my_func(x, fvec)
    fvec[1] = (x[1] - 1)^2
    fvec[2] = (x[2] - 2)^2
end

# Initial guess and setup
x = [0.0, 0.0]
fvec = zeros(2)

# Run optimization
(x_opt, fvec_final, niters, nfev, njev, naev, converged) = geodesiclm(
    my_func, nothing, nothing,
    x=x, fvec=fvec, n=2, m=2,
    print_level=1
)

println("Solution: ", x_opt)  # Should be close to [1.0, 2.0]
```

## Key Features

✓ **Complete Conversion** - All 10 Fortran subroutines converted to Julia functions
✓ **LinearAlgebra Integration** - Uses Julia's built-in linear algebra instead of BLAS/LAPACK
✓ **Algorithm Preservation** - Core algorithm structure maintained from Fortran
✓ **Comment Preservation** - All original comments carried over
✓ **Type Safety** - Explicit type annotations throughout
✓ **Well Documented** - Comprehensive docstrings and guide files
✓ **Pure Julia** - No external Fortran dependencies

## Algorithm Overview

The Geodesic Levenberg-Marquardt algorithm minimizes the sum of squares of nonlinear functions:

```
minimize: f(x) = (1/2) * Σ fᵢ(x)²
```

Key features:
- **Geodesic Acceleration**: Second-order correction for faster convergence
- **Bold Acceptance Criterion**: More efficient step acceptance
- **Broyden Updates**: Rank-deficient Jacobian updates
- **Adaptive Damping**: Automatic parameter scaling
- **Multiple Update Methods**: Various lambda/delta update strategies

## Main Function API

```julia
(x, fvec, niters, nfev, njev, naev, converged) = geodesiclm(
    func::Function,                      # Residual function
    jacobian::Union{Function, Nothing},  # Jacobian (optional)
    Avv::Union{Function, Nothing};       # Second derivatives (optional)
    
    # Required keywords
    x::Vector{Float64},                  # Initial parameters
    fvec::Vector{Float64},               # Residuals output
    n::Int,                              # Number of parameters
    m::Int,                              # Number of residuals
    
    # Optional keywords with defaults
    callback::Union{Function, Nothing}=nothing,
    info::Int=0,
    analytic_jac::Bool=false,
    analytic_Avv::Bool=false,
    center_diff::Bool=true,
    h1::Float64=1.0e-5,
    h2::Float64=1.0e-5,
    dtd::Union{Matrix{Float64}, Nothing}=nothing,
    damp_mode::Int=0,
    maxiter::Int=100,
    maxfev::Int=0,
    maxjev::Int=0,
    maxaev::Int=0,
    maxlam::Float64=-1.0,
    minlam::Float64=-1.0,
    artol::Float64=0.0,
    Cgoal::Float64=0.0,
    gtol::Float64=1.0e-8,
    xtol::Float64=1.0e-8,
    xrtol::Float64=1.0e-8,
    ftol::Float64=1.0e-8,
    frtol::Float64=1.0e-8,
    converged::Int=0,
    print_level::Int=0,
    print_unit::IO=stdout,
    imethod::Int=0,
    iaccel::Int=1,
    ibold::Int=1,
    ibroyden::Int=1,
    initialfactor::Float64=100.0,
    factoraccept::Float64=2.0,
    factorreject::Float64=2.0,
    avmax::Float64=10.0
)
```

### Return Values
- `x`: Optimized parameters
- `fvec`: Function values at solution
- `niters`: Number of iterations performed
- `nfev`: Number of function evaluations
- `njev`: Number of Jacobian evaluations
- `naev`: Number of second derivative evaluations
- `converged`: Convergence code (see documentation)

## Conversion Highlights

### BLAS/LAPACK → Julia LinearAlgebra

| Operation | BLAS/LAPACK | Julia |
|-----------|-------------|-------|
| Matrix multiply | DGEMM | `*` or `mul!()` |
| Dot product | DDOT | `dot()` |
| Norm | DNRM2 | `norm()` |
| Scaled sum | DAXPY | `+` with `.*` |
| Vector scale | DSCAL | `.*` operator |
| Cholesky factor | DPOTRF | `cholesky()` |
| Cholesky solve | DPOTRS | `\` operator |
| Triangular solve | DTRSV | `\` operator |

### Language Features

- ✓ Preserved 1-based indexing (Julia default)
- ✓ Replaced DO loops with Julia `for` loops
- ✓ Used Julia's native matrix operations
- ✓ Maintained algorithm structure exactly
- ✓ Added comprehensive docstrings

## File-by-File Conversion Details

See [CONVERSION_SUMMARY.md](CONVERSION_SUMMARY.md) for detailed information about:
- How each Fortran subroutine was converted
- BLAS/LAPACK to LinearAlgebra mappings
- Data type conversions
- Comment preservation notes

## Usage Examples

### Example 1: Fit a Line to Data

```julia
# Data
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [1.9, 3.8, 6.1, 7.9, 10.0]

function fit_line(params, residuals)
    a, b = params
    for i in 1:length(x_data)
        residuals[i] = y_data[i] - (a * x_data[i] + b)
    end
end

x = [1.0, 0.0]
fvec = zeros(5)

(x_opt, _, _, _, _, _, converged) = geodesiclm(
    fit_line, nothing, nothing,
    x=x, fvec=fvec, n=2, m=5,
    print_level=1
)

println("Slope: $(x_opt[1]), Intercept: $(x_opt[2])")
```

### Example 2: With Analytical Jacobian

```julia
function func(x, fvec)
    # Residuals
    fvec[1] = x[1] - 1
    fvec[2] = x[2] - 2
    fvec[3] = x[1]^2 + x[2]^2 - 5
end

function jac(x, fjac)
    # Jacobian matrix
    fjac[1, 1] = 1.0
    fjac[1, 2] = 0.0
    fjac[2, 1] = 0.0
    fjac[2, 2] = 1.0
    fjac[3, 1] = 2 * x[1]
    fjac[3, 2] = 2 * x[2]
end

x = [0.5, 0.5]
fvec = zeros(3)

(x_opt, _, _, _, _, _, converged) = geodesiclm(
    func, jac, nothing,
    x=x, fvec=fvec, n=2, m=3,
    analytic_jac=true,
    print_level=1
)
```

## References

Original algorithm papers:

- Transtrum, M. K., Machta, B. B., & Sethna, J. P. (2010). "Why are nonlinear 
  fits to data so challenging?" *Physical Review Letters*, 104(6), 060201.
  [doi:10.1103/PhysRevLett.104.060201](https://doi.org/10.1103/PhysRevLett.104.060201)

- Transtrum, M. K., Machta, B. B., & Sethna, J. P. (2011). "The geometry of 
  nonlinear least squares with applications to sloppy models and optimization." 
  *Physical Review E*, 80(3), 036701.
  [doi:10.1103/PhysRevE.80.036701](https://doi.org/10.1103/PhysRevE.80.036701)

## Integration with Other Julia Packages

The module is compatible with packages like:
- **Optim.jl** - Can be called as a custom optimizer
- **LsqFit.jl** - Can use for curve fitting workflows
- **DifferentialEquations.jl** - Can optimize ODE model parameters
- **NLSolve.jl** - Alternative nonlinear solver

## Troubleshooting

### NaN Production
If `converged = -11`, check for:
- Division by zero in your function
- Invalid operations (sqrt of negative)
- Overflow in computations

### Slow Convergence
Try:
- Better initial guess
- Adjust `h1`, `h2` step sizes
- Use analytical Jacobian if available
- Try different `imethod` values

### Memory Issues
For large-scale problems, consider:
- Reducing `maxiter`
- Using limited-memory variants

## License

This Julia conversion maintains the same license as the original Fortran code.
See [license.txt](license.txt) for details.

## Citation

If you use this code in your research, please cite:

1. The original papers listed above
2. This Julia implementation (see original Fortran repository)

## Support

For more information:
- See [QUICK_START.md](QUICK_START.md) for usage examples
- See [CONVERSION_SUMMARY.md](CONVERSION_SUMMARY.md) for technical details
- Check function docstrings: `?geodesiclm` in Julia REPL

---

**Conversion Date**: January 5, 2026  
**Version**: 1.0.2 (matching Fortran original)  
**Target**: Julia 1.6+  
**Dependencies**: LinearAlgebra (included with Julia)
