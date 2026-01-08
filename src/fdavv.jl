# -*- julia -*-
# ****************************************
# Routine for calculating finite-difference second directional derivative

"""
    fd_avv(m::Int, n::Int, x::Vector{Float64}, v::Vector{Float64}, 
           fvec::Vector{Float64}, fjac::Matrix{Float64}, func::Function, 
           jac_uptodate::Bool, h2::Float64)

Calculate the finite-difference second directional derivative of the 
objective functions in the direction v.

# Arguments
- `m`: number of functions
- `n`: number of parameters
- `x`: current point
- `v`: direction vector
- `fvec`: function values at x
- `fjac`: Jacobian matrix at x
- `func`: user-supplied function computing fvec (of form func(x, fvec))
- `jac_uptodate`: whether the Jacobian is current
- `h2`: finite-difference step size

# Returns
- `acc`: estimated second directional derivative (m-vector)
"""
function fd_avv(m::Int, n::Int, x::Vector{Float64}, v::Vector{Float64}, 
               fvec::Vector{Float64}, fjac::Matrix{Float64}, func::Function, 
               jac_uptodate::Bool, h2::Float64)
    
    if jac_uptodate
        # If jacobian is up to date, use it to reduce function evaluations
        xtmp = x + h2 * v
        ftmp = similar(fvec)
        func(xtmp, ftmp)
        acc = (2.0 / h2) * ((ftmp - fvec) / h2 - fjac * v)
    else
        # If jacobian not up to date, do not use jacobian in finite difference
        # This requires one more function call
        xtmp = x + h2 * v
        ftmp = similar(fvec)
        func(xtmp, ftmp)
        
        xtmp = x - h2 * v
        acc_tmp = similar(fvec)
        func(xtmp, acc_tmp)
        
        acc = (ftmp - 2 * fvec + acc_tmp) / (h2 * h2)
    end
    
    return acc
end
