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
function trust_region(n::Int, m::Int, fvec::Vector{Float64}, fjac::Matrix{Float64}, 
                     dtd::Matrix{Float64}, delta::Float64)
    
    # Parameters for dgqt
    rtol = 1.0e-3
    atol = 1.0e-3
    itmax = 10
    lam = 1.0  # Initial estimate
    
    # Transform to diagonal scaling
    # Compute J-tilde and gradient
    jtilde = similar(fjac)
    for i in 1:n
        jtilde[:, i] = fjac[:, i] / sqrt(dtd[i, i])
    end
    
    gradCtilde = fvec' * jtilde  # Row vector
    g = jtilde' * jtilde  # Compute J'*J
    
    # Solve the trust region problem
    (v, lam, info, f) = dgqt(n, g, vec(gradCtilde), delta, rtol, atol, itmax, lam)
    
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
function update_lam_factor(lam::Float64, accepted::Int, factoraccept::Float64, factorreject::Float64)
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
function update_lam_nelson(lam::Float64, accepted::Int, factoraccept::Float64, factorreject::Float64, rho::Float64)
    if accepted >= 0
        lam = lam * max(1.0 / factoraccept, 1.0 - (factorreject - 1.0) * (2.0 * rho - 1.0)^3)
    else
        nu = factorreject
        for i in 1:(-accepted)
            # Double nu for each rejection
            nu = nu * 2.0
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
function update_lam_umrigar(m::Int, n::Int, lam::Float64, accepted::Int, v::Vector{Float64}, 
                           vold::Vector{Float64}, fvec::Vector{Float64}, fjac::Matrix{Float64}, 
                           dtd::Matrix{Float64}, a_param::Float64, C::Float64, Cnew::Float64)
    
    amemory = exp(-1.0 / 5.0)
    
    # Compute cosine
    cos_on = dot(v, dtd * vold)
    cos_on = cos_on / sqrt(dot(v, dtd * v) * dot(vold, dtd * vold))
    
    if accepted >= 0
        if Cnew <= C
            if cos_on > 0.0
                a_param = amemory * a_param + (1.0 - amemory)
            else
                a_param = amemory * a_param + 0.5 * (1.0 - amemory)
            end
        else
            a_param = amemory * a_param + 0.5 * (1.0 - amemory)
        end
        
        factor = min(100.0, max(1.1, 1.0 / (2.2e-16 + 1.0 - abs(2.0 * a_param - 1.0))^2))
        
        if Cnew <= C && cos_on >= 0.0
            lam = lam / factor
        elseif Cnew > C
            lam = lam * sqrt(factor)
        end
    else
        a_param = amemory * a_param
        factor = min(100.0, max(1.1, 1.0 / (2.2e-16 + 1.0 - abs(2.0 * a_param - 1.0))^2))
        
        if cos_on > 0.0
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
function update_delta_factor(delta::Float64, accepted::Int, factoraccept::Float64, factorreject::Float64)
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
function update_delta_more(delta::Float64, lam::Float64, n::Int, v::Vector{Float64}, 
                          dtd::Matrix{Float64}, rho::Float64, C::Float64, Cnew::Float64, 
                          dirder::Float64, actred::Float64, av::Float64, avmax::Float64)
    
    pnorm = sqrt(dot(v, dtd * v))
    
    if rho > 0.25
        if lam > 0.0 && rho < 0.75
            temp = 1.0
        else
            temp = 2.0 * pnorm / delta
        end
    else
        if actred >= 0.0
            temp = 0.5
        else
            temp = 0.5 * dirder / (dirder + 0.5 * actred)
        end
        
        if 0.01 * Cnew >= C || temp < 0.1
            temp = 0.1
        end
    end
    
    # Make sure that if acceleration is too big, we decrease the step size
    if av > avmax
        temp = min(temp, max(avmax / av, 0.1))
    end
    
    delta = temp * min(delta, 10.0 * pnorm)
    lam = lam / temp
    
    return (delta, lam)
end
