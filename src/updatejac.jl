# -*- julia -*-
#****************************************
# Routine for rank-deficient jacobian update

"""
    update_jac(m::Int, n::Int, fjac::Matrix{Float64}, fvec::Vector{Float64}, 
               fvec_new::Vector{Float64}, acc::Vector{Float64}, v::Vector{Float64}, 
               a::Vector{Float64})

Update the Jacobian matrix using a rank-deficient Broyden update formula.

This routine performs a rank-deficient update of the Jacobian matrix based on 
the change in function values and the current step and acceleration vectors.

# Arguments
- `m`: number of functions
- `n`: number of parameters
- `fjac`: Jacobian matrix (modified in place)
- `fvec`: function values at current point
- `fvec_new`: function values at new point
- `acc`: acceleration term (second directional derivative)
- `v`: velocity step
- `a`: acceleration step

# Returns
- Updated `fjac` matrix (modified in place)
"""
function update_jac!(m::Int, n::Int, fjac::Matrix{Float64}, fvec::Vector{Float64}, 
                    fvec_new::Vector{Float64}, acc::Vector{Float64}, v::Vector{Float64}, 
                    a::Vector{Float64})
    
    # First-order update
    r1 = fvec + 0.5 * (fjac * v) + 0.125 * acc
    djac = 2.0 * (r1 - fvec - 0.5 * (fjac * v)) / dot(v, v)
    
    for i in 1:m
        for j in 1:n
            fjac[i, j] = fjac[i, j] + djac[i] * 0.5 * v[j]
        end
    end
    
    # Second-order update
    v2 = 0.5 * (v + a)
    djac = 0.5 * (fvec_new - r1 - fjac * v2) / dot(v2, v2)
    
    for i in 1:m
        for j in 1:n
            fjac[i, j] = fjac[i, j] + djac[i] * v2[j]
        end
    end
end
