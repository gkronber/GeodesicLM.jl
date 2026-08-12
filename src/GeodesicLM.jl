# -*- julia -*-
# GeodesicLM.jl - Module wrapping the Geodesic Levenberg-Marquardt algorithm

module GeodesicLM

using LinearAlgebra

# Import all component functions
include("accept.jl")
include("converge.jl")
include("destsv.jl")
include("dgqt.jl")
include("dpmpar.jl")
include("fdavv.jl")
include("fdjac.jl")
include("lambda.jl")
include("updatejac.jl")
include("workspace.jl")
include("geodesiclm_alg.jl")

# Backend-agnostic GPU kernel building blocks (see PLAN.md / test/gpu_kernels.jl)
include("gpu/KAOps.jl")
# M4/M5: device workspace and GPU objective (+ GPU finite differences)
include("gpu/GPUWorkspace.jl")
include("gpu/GPUObjective.jl")
# M6: one on-device LM step (jitj/g/cholesky/solves/pred_red/cos_alpha/acc)
include("gpu/GPUStep.jl")

# Export the main API
export geodesiclm
export GLMWorkspace
export GPUWorkspace
export GPUObjective
export acceptance
export convergence_check
export destsv
export dgqt
export dpmpar
export fd_avv
export fdjac
export trust_region
export update_lam_factor
export update_lam_nelson
export update_lam_umrigar
export update_delta_factor
export update_delta_more
export update_jac!

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
  Phys. Rev. E 80, 036701 (2011)
"""
GeodesicLM

end  # module GeodesicLM
