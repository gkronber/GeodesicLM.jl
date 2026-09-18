# -*- julia -*-
# ****************************************
# Routine for calculating finite-difference jacobian

"""
    fdjac!(fjac, m, n, x, fvec, func, eps, center_diff,
           x_work = similar(x), f1 = similar(fvec), f2 = similar(fvec))

Calculate the finite-difference Jacobian matrix of a system of m functions
in n variables, writing the result into `fjac`.

# Arguments
- `fjac`: m by n output Jacobian matrix
- `m`: number of functions
- `n`: number of parameters
- `x`: current point
- `fvec`: function values at x
- `func`: user-supplied function computing fvec (of form func(x, fvec))
- `eps`: finite-difference step size parameter
- `center_diff`: if true, use central differences; if false, use forward differences
- `x_work`, `f1`, `f2`: scratch arrays (supply them to keep the call allocation-free)

# Returns
- `fjac` (modified in place)
"""
function fdjac!(fjac::AbstractMatrix{T}, m::Int, n::Int, x::AbstractVector{T},
                fvec::AbstractVector{T}, func::F, eps::T, center_diff::Bool,
                x_work::AbstractVector{T} = similar(x),
                f1::AbstractVector{T} = similar(fvec),
                f2::AbstractVector{T} = similar(fvec)) where {T<:AbstractFloat,F}

    epsmach = dpmpar(T, 1)
    copyto!(x_work, x)

    if center_diff
        # Use central differences
        for i in 1:n
            h = eps * abs(x[i])
            if h < epsmach
                h = eps
            end

            x_work[i] = x[i] + T(0.5) * h
            func(x_work, f1)

            x_work[i] = x[i] - T(0.5) * h
            func(x_work, f2)

            x_work[i] = x[i]

            @inbounds for k in 1:m
                fjac[k, i] = (f1[k] - f2[k]) / h
            end
        end
    else
        # Use forward differences
        for i in 1:n
            h = eps * abs(x[i])
            if h < epsmach
                h = eps
            end

            x_work[i] = x[i] + h
            func(x_work, f1)
            x_work[i] = x[i]

            @inbounds for k in 1:m
                fjac[k, i] = (f1[k] - fvec[k]) / h
            end
        end
    end

    return fjac
end

"""
    fdjac(m, n, x, fvec, func, eps, center_diff)

Allocating wrapper around [`fdjac!`](@ref); returns a freshly allocated
m by n Jacobian matrix.
"""
function fdjac(m::Int, n::Int, x::AbstractVector{T}, fvec::AbstractVector{T},
               func::F, eps::T, center_diff::Bool) where {T<:AbstractFloat,F}
    fdjac!(zeros(T, m, n), m, n, x, fvec, func, eps, center_diff)
end
