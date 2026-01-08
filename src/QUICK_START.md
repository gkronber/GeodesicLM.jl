# -*- julia -*-
# QUICK_START.md - Quick start guide for using the Julia version of GeodesicLM

# Quick Start Guide for GeodesicLM.jl

## Loading the Module

```julia
# Add the geodesicLM folder to your Julia load path, then:
include("GeodesicLM.jl")
using .GeodesicLM

# Or load individual components:
include("geodesiclm.jl")
include("accept.jl")
# ... etc
```

## Basic Usage

### Step 1: Define your objective function

Your function should compute residuals and modify a vector in place:

```julia
function my_residuals(x, fvec)
    # Example: fit y = a*x + b to data
    # Assuming you have data points and parameters in scope
    for i in 1:length(fvec)
        fvec[i] = y_data[i] - (x[1] * x_data[i] + x[2])
    end
end
```

### Step 2: (Optional) Define Jacobian

If you want to provide analytical Jacobian:

```julia
function my_jacobian(x, fjac)
    # fjac is m×n where m=number of residuals, n=number of parameters
    for i in 1:m
        fjac[i, 1] = -x_data[i]  # derivative w.r.t. parameter 1
        fjac[i, 2] = -1.0        # derivative w.r.t. parameter 2
    end
end
```

### Step 3: (Optional) Define second derivatives

For geodesic acceleration:

```julia
function my_avv(x, v, acc)
    # acc is the second directional derivative
    # For linear model, this is zero
    acc .= 0.0
end
```

### Step 4: Call the optimizer

```julia
using GeodesicLM

# Set up initial guess
x = [1.0, 0.0]  # Initial guess for [a, b]
m = length(y_data)  # Number of residuals
n = length(x)       # Number of parameters

# Allocate output
fvec = zeros(m)

# Run optimization
(x_opt, fvec_final, niters, nfev, njev, naev, converged) = geodesiclm(
    my_residuals, 
    nothing,  # No Jacobian provided (will use finite differences)
    nothing,  # No second derivative provided
    x=x,
    fvec=fvec,
    n=n,
    m=m,
    # Optional parameters:
    analytic_jac=false,  # Use finite differences for Jacobian
    center_diff=true,    # Use central differences
    h1=1.0e-5,          # Step size for Jacobian
    h2=1.0e-5,          # Step size for second derivatives
    maxiter=100,         # Maximum iterations
    print_level=1        # Print progress information
)

println("Solution found:")
println("  Parameters: ", x_opt)
println("  Final cost: ", 0.5 * dot(fvec_final, fvec_final))
println("  Iterations: ", niters)
println("  Convergence: ", converged)
```

## Common Options

### Print Levels

- `print_level=0`: No output
- `print_level=1`: Summary output
- `print_level=2`: Progress after accepted steps
- `print_level=3`: Progress every iteration
- `print_level=4`: Detailed with vectors (accepted steps only)
- `print_level=5`: Detailed with vectors (every iteration)

### Convergence Criteria

Set to 0 to disable:

- `gtol`: Gradient norm tolerance (default: 1.0e-8)
- `xtol`: Step size tolerance (default: 1.0e-8)
- `xrtol`: Relative parameter change tolerance (default: 1.0e-8)
- `ftol`: Absolute cost stagnation tolerance (default: 1.0e-8)
- `frtol`: Relative cost stagnation tolerance (default: 1.0e-8)
- `Cgoal`: Target cost value (default: 0.0)
- `artol`: Angle criterion tolerance (default: 0.0)

### Method Selection

- `imethod=0`: Fixed factor lambda update (default)
- `imethod=1`: Nelson's lambda update
- `imethod=2`: Umrigar & Nightingale's method
- `imethod=10`: Fixed factor delta (trust region) update
- `imethod=11`: Moré's delta update

### Acceleration

- `iaccel=0`: No geodesic acceleration
- `iaccel=1`: Include geodesic acceleration (default)

### Acceptance Criterion

- `ibold=0`: Only accept downhill steps
- `ibold=1`: Bold criterion with beta*C
- `ibold=2`: Bold criterion with beta²*C
- `ibold=3`: Bold criterion with beta*C (vs current C)
- `ibold=4`: Bold criterion with beta²*C (vs current C)

### Jacobian Updates

- `ibroyden≤0`: Full Jacobian update after acceptance
- `ibroyden>0`: Rank-deficient Broyden updates

## Convergence Return Codes

- `converged = 1`: Angle criterion (artol) reached
- `converged = 2`: Cost goal (Cgoal) reached
- `converged = 3`: Gradient tolerance (gtol) reached
- `converged = 4`: Step size tolerance (xtol) reached
- `converged = 5`: Relative parameter change (xrtol) reached
- `converged = 6`: Cost stagnation (ftol) reached
- `converged = 7`: Relative cost stagnation (frtol) reached
- `converged = -1`: Maximum iterations exceeded
- `converged = -2`: Maximum function evaluations exceeded
- `converged = -3`: Maximum Jacobian evaluations exceeded
- `converged = -4`: Maximum second derivative evaluations exceeded
- `converged = -5`: Maximum lambda exceeded
- `converged = -6`: Minimum lambda exceeded
- `converged = -10`: User requested termination (via callback)
- `converged = -11`: NaN produced during optimization

## Advanced: Damping Modes

- `damp_mode=0`: Identity damping (standard Levenberg-Marquardt)
- `damp_mode=1`: Diagonal scaling based on Jacobian diagonal

## Advanced: Callback Function

Provide a callback to monitor progress or terminate early:

```julia
function my_callback(x, v, a, fvec, fjac, acc, lam, dtd, fvec_new, accepted, info)
    # Custom monitoring
    current_cost = 0.5 * dot(fvec, fvec)
    println("Cost: ", current_cost, " Lambda: ", lam)
    
    # Set info to non-zero to terminate
    # info = 1  # Uncomment to terminate after this iteration
end

# Then pass to geodesiclm:
# callback=my_callback
```

## File Structure

```
src/
├── GeodesicLM.jl          # Main module file
├── geodesiclm.jl          # Main algorithm
├── accept.jl              # Acceptance criterion
├── converge.jl            # Convergence checking
├── destsv.jl              # Singular value estimation
├── dgqt.jl                # Trust region subproblem
├── dpmpar.jl              # Machine parameters
├── fdavv.jl               # Finite-difference second derivatives
├── fdjac.jl               # Finite-difference Jacobian
├── lambda.jl              # Lambda/delta update methods
├── updatejac.jl           # Jacobian update
├── CONVERSION_SUMMARY.md  # Detailed conversion notes
└── QUICK_START.md         # This file
```

## Performance Tips

1. **Analytical derivatives**: Provide `jacobian` for better accuracy and speed
2. **Step sizes**: Adjust `h1` and `h2` for finite differences
3. **Trust region**: Try `imethod=10` or `imethod=11` if direct methods fail
4. **Acceleration**: Enable `iaccel=1` for faster convergence on well-scaled problems
5. **Broyden updates**: Use `ibroyden>0` to reduce Jacobian evaluations

## References

For algorithm details, see:

- Transtrum, M. K., Machta, B. B., & Sethna, J. P. (2010). "Why are nonlinear 
  fits to data so challenging?" Physical Review Letters, 104(6), 060201.

- Transtrum, M. K., Machta, B. B., & Sethna, J. P. (2011). "The geometry of 
  nonlinear least squares with applications to sloppy models and optimization." 
  Physical Review E, 80(3), 036701.
