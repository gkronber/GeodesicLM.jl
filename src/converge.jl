# -*- julia -*-
# ****************************************
# Routine to Check for Convergence

"""
    convergence_check(m, n, accepted, counter, C, Cnew, x, fvec, fjac, lam, xnew,
                      nfev, maxfev, njev, maxjev, naev, maxaev, maxlam, minlam,
                      artol, Cgoal, gtol, xtol, xrtol, ftol, frtol, cos_alpha,
                      grad = similar(x); gradnorm = nothing)

Check for convergence of the Levenberg-Marquardt algorithm.

`grad` is a length-`n` scratch vector; supply one to keep the call
allocation-free.  `gradnorm`, if given, is the norm of the gradient `fjac'fvec`
for the gtol criterion, which is then not computed again.

# Returns
- `(converged, counter)`: tuple with convergence status and counter value
"""
function convergence_check(m::Int, n::Int, accepted::Int, counter::Int, C::T,
                           Cnew::T, x::AbstractVector{T}, fvec::AbstractVector{T},
                           fjac::AbstractMatrix{T}, lam::T, xnew::AbstractVector{T},
                           nfev::Int, maxfev::Int, njev::Int, maxjev::Int, naev::Int,
                           maxaev::Int, maxlam::T, minlam::T, artol::T,
                           Cgoal::T, gtol::T, xtol::T, xrtol::T,
                           ftol::T, frtol::T, cos_alpha::T,
                           grad::AbstractVector{T} = similar(x);
                           gradnorm::Union{Nothing,T} = nothing) where {T<:AbstractFloat}

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
    if maxlam > zero(T)
        if lam >= maxlam
            converged = -5
            return (converged, counter)
        end
    end

    # Check minlam
    if minlam > zero(T) && lam > zero(T)
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
    if artol > zero(T)
        if cos_alpha <= artol
            converged = 1
            return (converged, counter)
        end
    end

    # If gradient is small
    if gradnorm === nothing
        _gemv_t!(grad, fjac, fvec)
        gradnorm = sqrt(_dot(grad, grad))
    end
    if gradnorm <= gtol
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
    stepnorm2 = zero(T)
    @inbounds for i in 1:n
        d = x[i] - xnew[i]
        stepnorm2 += d * d
    end
    if sqrt(stepnorm2) < xtol
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
    if (C - Cnew) <= ftol && (C - Cnew) >= zero(T)
        counter = counter + 1
        if counter >= 3
            converged = 6
            return (converged, counter)
        end
        return (converged, counter)
    end

    # If cost is not decreasing relatively -- again can happen by accident so require three times in a row
    if (C - Cnew) <= (frtol * C) && (C - Cnew) >= zero(T)
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
