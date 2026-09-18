# -*- julia -*-
#****************************************
# Routines for updating lam

"""
    trust_region(n::Int, m::Int, fvec::Vector{Float64}, fjac::Matrix{Float64}, 
                 dtd::Matrix{Float64}, delta::Float64)

Calculate the Levenberg-Marquardt step using the trust region method via dgqt.

# Arguments
- `n`: number of parameters
- `m`: number of functions
- `fvec`: function values
- `fjac`: Jacobian matrix
- `dtd`: damping matrix (assumed diagonal)
- `delta`: trust region radius

# Returns
- `(v, lam)`: step vector and Lagrange multiplier
"""
function trust_region(n::Int, m::Int, fvec::AbstractVector{T}, fjac::AbstractMatrix{T},
                     dtd::AbstractMatrix{T}, delta::T) where {T<:AbstractFloat}

    # Parameters for dgqt
    rtol = T(1.0e-3)
    atol = T(1.0e-3)
    itmax = 10
    lam = one(T)  # Initial estimate

    # Transform to diagonal scaling
    # Compute J-tilde and gradient
    jtilde = similar(fjac)
    for i in 1:n
        s = sqrt(dtd[i, i])
        @inbounds for k in axes(fjac, 1)
            jtilde[k, i] = fjac[k, i] / s
        end
    end

    gradCtilde = similar(fvec, n)
    mul!(gradCtilde, transpose(jtilde), fvec)
    g = similar(fjac, n, n)
    mul!(g, transpose(jtilde), jtilde)  # Compute J'*J

    # Solve the trust region problem
    (v, lam, info, f) = dgqt(n, g, gradCtilde, delta, rtol, atol, itmax, lam)

    return (v, lam)
end

# ============================================================================
# Traditional update methods
# ============================================================================

"""
    update_lam_factor(lam::Float64, accepted::Int, factoraccept::Float64, factorreject::Float64)

Update lam based on accepted/rejected step using fixed factors.

# Arguments
- `lam`: current Levenberg-Marquardt parameter
- `accepted`: acceptance status
- `factoraccept`: factor to divide lam by on acceptance
- `factorreject`: factor to multiply lam by on rejection

# Returns
- `lam`: updated Levenberg-Marquardt parameter
"""
function update_lam_factor(lam::T, accepted::Int, factoraccept::T, factorreject::T) where {T<:AbstractFloat}
    if accepted >= 0
        lam = lam / factoraccept
    else
        lam = lam * factorreject
    end
    return lam
end

"""
    update_lam_nelson(lam::Float64, accepted::Int, factoraccept::Float64, factorreject::Float64, rho::Float64)

Update lam using the Nelson method.

# Arguments
- `lam`: current Levenberg-Marquardt parameter
- `accepted`: acceptance status
- `factoraccept`: acceptance factor
- `factorreject`: rejection factor
- `rho`: gain ratio

# Returns
- `lam`: updated Levenberg-Marquardt parameter
"""
function update_lam_nelson(lam::T, accepted::Int, factoraccept::T, factorreject::T, rho::T) where {T<:AbstractFloat}
    if accepted >= 0
        lam = lam * max(one(T) / factoraccept, one(T) - (factorreject - one(T)) * (2 * rho - one(T))^3)
    else
        nu = factorreject
        for i in 1:(-accepted)
            # Double nu for each rejection
            nu = nu * 2
        end
        lam = lam * nu
    end
    return lam
end

"""
    update_lam_umrigar(m::Int, n::Int, lam::Float64, accepted::Int, v::Vector{Float64}, 
                       vold::Vector{Float64}, fvec::Vector{Float64}, fjac::Matrix{Float64}, 
                       dtd::Matrix{Float64}, a_param::Float64, C::Float64, Cnew::Float64)

Update lam using the Umrigar and Nightingale method.

# Arguments
- `m`: number of functions
- `n`: number of parameters
- `lam`: current Levenberg-Marquardt parameter
- `accepted`: acceptance status
- `v`: current step
- `vold`: previous step
- `fvec`: function values
- `fjac`: Jacobian matrix
- `dtd`: damping matrix
- `a_param`: adaptation parameter
- `C`: current cost
- `Cnew`: new cost

# Returns
- `(lam, a_param)`: updated Levenberg-Marquardt parameter and adaptation parameter
"""
function update_lam_umrigar(m::Int, n::Int, lam::T, accepted::Int, v::AbstractVector{T},
                           vold::AbstractVector{T}, fvec::AbstractVector{T}, fjac::AbstractMatrix{T},
                           dtd::AbstractMatrix{T}, a_param::T, C::T, Cnew::T,
                           tmp::AbstractVector{T} = similar(v)) where {T<:AbstractFloat}

    amemory = exp(-one(T) / 5)

    # Compute cosine
    mul!(tmp, dtd, vold)
    cos_on = dot(v, tmp)
    vold_dtd_vold = dot(vold, tmp)
    mul!(tmp, dtd, v)
    cos_on = cos_on / sqrt(dot(v, tmp) * vold_dtd_vold)
    
    if accepted >= 0
        if Cnew <= C
            if cos_on > zero(T)
                a_param = amemory * a_param + (one(T) - amemory)
            else
                a_param = amemory * a_param + T(0.5) * (one(T) - amemory)
            end
        else
            a_param = amemory * a_param + T(0.5) * (one(T) - amemory)
        end
        
        factor = min(T(100), max(T(1.1), one(T) / (eps(T) + one(T) - abs(2 * a_param - one(T)))^2))

        if Cnew <= C && cos_on >= zero(T)
            lam = lam / factor
        elseif Cnew > C
            lam = lam * sqrt(factor)
        end
    else
        a_param = amemory * a_param
        factor = min(T(100), max(T(1.1), one(T) / (eps(T) + one(T) - abs(2 * a_param - one(T)))^2))

        if cos_on > zero(T)
            lam = lam * sqrt(factor)
        else
            lam = lam * factor
        end
    end
    
    return (lam, a_param)
end

# ============================================================================
# Trust region update methods
# ============================================================================

"""
    update_delta_factor(delta::Float64, accepted::Int, factoraccept::Float64, factorreject::Float64)

Update delta (trust region radius) based on accepted/rejected step using fixed factors.

# Arguments
- `delta`: current trust region radius
- `accepted`: acceptance status
- `factoraccept`: factor to multiply delta by on acceptance
- `factorreject`: factor to divide delta by on rejection

# Returns
- `delta`: updated trust region radius
"""
function update_delta_factor(delta::T, accepted::Int, factoraccept::T, factorreject::T) where {T<:AbstractFloat}
    if accepted >= 0
        delta = delta * factoraccept
    else
        delta = delta / factorreject
    end
    return delta
end

"""
    update_delta_more(delta::Float64, lam::Float64, n::Int, v::Vector{Float64}, 
                      dtd::Matrix{Float64}, rho::Float64, C::Float64, Cnew::Float64, 
                      dirder::Float64, actred::Float64, av::Float64, avmax::Float64)

Update delta (trust region radius) as described in Moré reference.

# Arguments
- `delta`: current trust region radius
- `lam`: Lagrange multiplier
- `n`: number of parameters
- `v`: step vector
- `dtd`: damping matrix
- `rho`: gain ratio
- `C`: current cost
- `Cnew`: new cost
- `dirder`: directional derivative
- `actred`: actual reduction
- `av`: acceleration vector norm
- `avmax`: maximum allowed acceleration norm

# Returns
- `(delta, lam)`: updated trust region radius and Lagrange multiplier
"""
function update_delta_more(delta::T, lam::T, n::Int, v::AbstractVector{T},
                          dtd::AbstractMatrix{T}, rho::T, C::T, Cnew::T,
                          dirder::T, actred::T, av::T, avmax::T,
                          tmp::AbstractVector{T} = similar(v)) where {T<:AbstractFloat}

    mul!(tmp, dtd, v)
    pnorm = sqrt(dot(v, tmp))

    if rho > T(0.25)
        if lam > zero(T) && rho < T(0.75)
            temp = one(T)
        else
            temp = 2 * pnorm / delta
        end
    else
        if actred >= zero(T)
            temp = T(0.5)
        else
            temp = T(0.5) * dirder / (dirder + T(0.5) * actred)
        end

        if T(0.01) * Cnew >= C || temp < T(0.1)
            temp = T(0.1)
        end
    end

    # Make sure that if acceleration is too big, we decrease the step size
    if av > avmax
        temp = min(temp, max(avmax / av, T(0.1)))
    end

    delta = temp * min(delta, 10 * pnorm)
    lam = lam / temp
    
    return (delta, lam)
end
