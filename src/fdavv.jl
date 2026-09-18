# -*- julia -*-
# ****************************************
# Routine for calculating finite-difference second directional derivative

"""
    fd_avv!(acc, m, n, x, v, fvec, fjac, func, jac_uptodate, h2,
            x_work = similar(x), ftmp = similar(fvec))

Calculate the finite-difference second directional derivative of the
objective functions in the direction v, writing the result into `acc`.

# Arguments
- `acc`: length-`m` output vector
- `m`: number of functions
- `n`: number of parameters
- `x`: current point
- `v`: direction vector
- `fvec`: function values at x
- `fjac`: Jacobian matrix at x
- `func`: user-supplied function computing fvec (of form func(x, fvec))
- `jac_uptodate`: whether the Jacobian is current
- `h2`: finite-difference step size
- `x_work`, `ftmp`: scratch arrays (supply them to keep the call allocation-free)

# Returns
- `acc` (modified in place)
"""
function fd_avv!(acc::AbstractVector{T}, m::Int, n::Int, x::AbstractVector{T},
                 v::AbstractVector{T}, fvec::AbstractVector{T}, fjac::AbstractMatrix{T},
                 func::F, jac_uptodate::Bool, h2::T,
                 x_work::AbstractVector{T} = similar(x),
                 ftmp::AbstractVector{T} = similar(fvec)) where {T<:AbstractFloat,F}

    if jac_uptodate
        # If jacobian is up to date, use it to reduce function evaluations
        @inbounds for i in 1:n
            x_work[i] = x[i] + h2 * v[i]
        end
        func(x_work, ftmp)
        mul!(acc, fjac, v)                    # acc := J*v
        @inbounds for k in 1:m
            acc[k] = (2 / h2) * ((ftmp[k] - fvec[k]) / h2 - acc[k])
        end
    else
        # If jacobian not up to date, do not use jacobian in finite difference
        # This requires one more function call
        @inbounds for i in 1:n
            x_work[i] = x[i] + h2 * v[i]
        end
        func(x_work, ftmp)

        @inbounds for i in 1:n
            x_work[i] = x[i] - h2 * v[i]
        end
        func(x_work, acc)                     # acc := f(x - h2*v)

        @inbounds for k in 1:m
            acc[k] = (ftmp[k] - 2 * fvec[k] + acc[k]) / (h2 * h2)
        end
    end

    return acc
end

"""
    fd_avv(m, n, x, v, fvec, fjac, func, jac_uptodate, h2)

Allocating wrapper around [`fd_avv!`](@ref); returns a freshly allocated
length-`m` vector.
"""
function fd_avv(m::Int, n::Int, x::AbstractVector{T}, v::AbstractVector{T},
                fvec::AbstractVector{T}, fjac::AbstractMatrix{T}, func::F,
                jac_uptodate::Bool, h2::T) where {T<:AbstractFloat,F}
    fd_avv!(zeros(T, m), m, n, x, v, fvec, fjac, func, jac_uptodate, h2)
end
