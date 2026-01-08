# -*- julia -*-
# ****************************************
# Routine to Check for Convergence

"""
    convergence_check(m::Int, n::Int, accepted::Int, counter::Int, C::Float64, 
                     Cnew::Float64, x::Vector{Float64}, fvec::Vector{Float64}, 
                     fjac::Matrix{Float64}, lam::Float64, xnew::Vector{Float64},
                     nfev::Int, maxfev::Int, njev::Int, maxjev::Int, naev::Int, 
                     maxaev::Int, maxlam::Float64, minlam::Float64, artol::Float64,
                     Cgoal::Float64, gtol::Float64, xtol::Float64, xrtol::Float64, 
                     ftol::Float64, frtol::Float64, cos_alpha::Float64)

Check for convergence of the Levenberg-Marquardt algorithm.

# Returns
- `(converged, counter)`: tuple with convergence status and counter value
"""
function convergence_check(m::Int, n::Int, accepted::Int, counter::Int, C::Float64, 
                          Cnew::Float64, x::Vector{Float64}, fvec::Vector{Float64}, 
                          fjac::Matrix{Float64}, lam::Float64, xnew::Vector{Float64},
                          nfev::Int, maxfev::Int, njev::Int, maxjev::Int, naev::Int, 
                          maxaev::Int, maxlam::Float64, minlam::Float64, artol::Float64,
                          Cgoal::Float64, gtol::Float64, xtol::Float64, xrtol::Float64, 
                          ftol::Float64, frtol::Float64, cos_alpha::Float64)
    
    converged = 0
    
    # The first few criteria should be checked every iteration, since
    # they depend on counts and the Jacobian but not the proposed step.
    
    # Check nfev
    if maxfev > 0
        if nfev >= maxfev
            converged = -2
            counter = 0
            return (converged, counter)
        end
    end
    
    # Check njev
    if maxjev > 0
        if njev >= maxjev
            converged = -3
            return (converged, counter)
        end
    end
    
    # Check naev
    if maxaev > 0
        if naev >= maxaev
            converged = -4
            return (converged, counter)
        end
    end
    
    # Check maxlam
    if maxlam > 0.0
        if lam >= maxlam
            converged = -5
            return (converged, counter)
        end
    end
    
    # Check minlam
    if minlam > 0.0 && lam > 0.0
        if lam <= minlam
            counter = counter + 1
            if counter >= 3
                converged = -6
                return (converged, counter)
            end
            return (converged, counter)
        end
    end
    
    # Check artol -- angle between residual vector and tangent plane
    if artol > 0.0
        if cos_alpha <= artol
            converged = 1
            return (converged, counter)
        end
    end
    
    # If gradient is small
    grad = -1.0 * (fvec' * fjac)
    if sqrt(dot(grad, grad)) <= gtol
        converged = 3
        return (converged, counter)
    end
    
    # If cost is sufficiently small
    if C < Cgoal
        converged = 2
        return (converged, counter)
    end
    
    # If step is not accepted, then don't check remaining criteria
    if accepted < 0
        counter = 0
        converged = 0
        return (converged, counter)
    end
    
    # If step size is small
    if sqrt(dot(x - xnew, x - xnew)) < xtol
        converged = 4
        return (converged, counter)
    end
    
    # If each parameter is moving relatively small
    converged = 5
    for i in 1:n
        if abs(x[i] - xnew[i]) > xrtol * abs(x[i]) || (xnew[i] != xnew[i])
            converged = 0
            break
        end
    end
    if converged == 5
        return (converged, counter)
    end
    
    # If cost is not decreasing -- this can happen by accident, so we require that it occur three times in a row
    if (C - Cnew) <= ftol && (C - Cnew) >= 0.0
        counter = counter + 1
        if counter >= 3
            converged = 6
            return (converged, counter)
        end
        return (converged, counter)
    end
    
    # If cost is not decreasing relatively -- again can happen by accident so require three times in a row
    if (C - Cnew) <= (frtol * C) && (C - Cnew) >= 0.0
        counter = counter + 1
        if counter >= 3
            converged = 7
            return (converged, counter)
        end
        return (converged, counter)
    end
    
    # If none of the above: continue
    counter = 0
    converged = 0
    
    return (converged, counter)
end
