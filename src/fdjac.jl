# -*- julia -*-
# ****************************************
# Routine for calculating finite-difference jacobian

"""
    fdjac(m::Int, n::Int, x::Vector{Float64}, fvec::Vector{Float64}, 
          func::Function, eps::Float64, center_diff::Bool)

Calculate the finite-difference Jacobian matrix of a system of m functions 
in n variables.

# Arguments
- `m`: number of functions
- `n`: number of parameters
- `x`: current point
- `fvec`: function values at x
- `func`: user-supplied function computing fvec (of form func(x, fvec))
- `eps`: finite-difference step size parameter
- `center_diff`: if true, use central differences; if false, use forward differences

# Returns
- `fjac`: m by n Jacobian matrix
"""
function fdjac(m::Int, n::Int, x::Vector{Float64}, fvec::Vector{Float64}, 
              func::Function, eps::Float64, center_diff::Bool)
    
    fjac = zeros(Float64, m, n)
    epsmach = dpmpar(1)
    
    if center_diff
        # Use central differences
        for i in 1:n
            h = eps * abs(x[i])
            if h < epsmach
                h = eps
            end
            
            # Evaluate at x + h*e_i
            x_plus = copy(x)
            x_plus[i] += 0.5 * h
            fvec_plus = similar(fvec)
            func(x_plus, fvec_plus)
            
            # Evaluate at x - h*e_i
            x_minus = copy(x)
            x_minus[i] -= 0.5 * h
            fvec_minus = similar(fvec)
            func(x_minus, fvec_minus)
            
            # Compute finite-difference derivative
            fjac[:, i] = (fvec_plus - fvec_minus) / h
        end
    else
        # Use forward differences
        for i in 1:n
            h = eps * abs(x[i])
            if h < epsmach
                h = eps
            end
            
            # Evaluate at x + h*e_i
            x_plus = copy(x)
            x_plus[i] += h
            fvec_plus = similar(fvec)
            func(x_plus, fvec_plus)
            
            # Compute finite-difference derivative
            fjac[:, i] = (fvec_plus - fvec) / h
        end
    end
    
    return fjac
end
