# -*- julia -*-
# GeodesicLM.jl - Module wrapping the Geodesic Levenberg-Marquardt algorithm

module GeodesicLM

using LinearAlgebra

# Import all component functions
include("smallblas.jl")
include("workspace.jl")
include("accept.jl")
include("converge.jl")
include("destsv.jl")
include("dgqt.jl")
include("dpmpar.jl")
include("fdavv.jl")
include("fdjac.jl")
include("lambda.jl")
include("updatejac.jl")
include("geodesiclm_alg.jl")

# Export the main API
export geodesiclm, GeodesicLMWorkspace

public default_fd_step, default_avv_step, default_tolerance, default_initialfactor

"""
    GeodesicLM

A Julia module implementing the Geodesic Levenberg-Marquardt algorithm 
for nonlinear least squares optimization.

The main entry point is the `geodesiclm()` function.

# Example

```julia
using GeodesicLM

# Define your objective function
function my_func(x, fvec)
    # Compute residuals and store in fvec
    fvec[1] = x[1]^2 + x[2]^2 - 1
    fvec[2] = x[1] - x[2]
end

# Initial guess
x = [0.5, 0.5]
fvec = zeros(2)

# Run optimization
(x_opt, fvec_final, niters, nfev, njev, naev, converged) = geodesiclm(
    my_func, nothing, nothing,
    x=x, fvec=fvec, n=2, m=2,
    print_level=1
)

println("Solution: ", x_opt)
println("Convergence code: ", converged)
```

# References

- Transtrum M.K., Machta B.B., and Sethna J.P., "Why are nonlinear fits to data 
  so challenging?" Phys. Rev. Lett. 104, 060201 (2010)
- Transtrum M.K., Machta B.B., and Sethna J.P., "The geometry of nonlinear 
  least squares with applications to sloppy models and optimization," 
  Phys. Rev. E 83, 036701 (2011)
"""
GeodesicLM

end  # module GeodesicLM
