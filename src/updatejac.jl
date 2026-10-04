# -*- julia -*-
#****************************************
# Routine for rank-deficient jacobian update

"""
    update_jac!(m, n, fjac, fvec, fvec_new, acc, v, a,
                r1 = similar(fvec), djac = similar(fvec), v2 = similar(v))

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
- `r1`, `djac`: length-`m` scratch vectors
- `v2`: length-`n` scratch vector

# Returns
- Updated `fjac` matrix (modified in place)
"""
function update_jac!(m::Int, n::Int, fjac::AbstractMatrix{T}, fvec::AbstractVector{T},
                     fvec_new::AbstractVector{T}, acc::AbstractVector{T},
                     v::AbstractVector{T}, a::AbstractVector{T},
                     r1::AbstractVector{T} = similar(fvec),
                     djac::AbstractVector{T} = similar(fvec),
                     v2::AbstractVector{T} = similar(v)) where {T<:AbstractFloat}

    # First-order update.  `r1 = fvec + 0.5*(J*v) + 0.125*acc`, and the update
    # direction reduces to `djac = 2*(r1 - fvec - 0.5*(J*v))/(v'v)` = the
    # acceleration contribution alone -- but it is kept in this form so the
    # result matches the Fortran original bit for bit.
    _gemv!(djac, fjac, v)                     # djac := J*v (reused as scratch)
    @inbounds for i in 1:m
        r1[i] = fvec[i] + T(0.5) * djac[i] + T(0.125) * acc[i]
    end
    vtv = _dot(v, v)
    @inbounds for i in 1:m
        djac[i] = 2 * (r1[i] - fvec[i] - T(0.5) * djac[i]) / vtv
    end

    @inbounds for j in 1:n
        vj = T(0.5) * v[j]
        for i in 1:m
            fjac[i, j] = fjac[i, j] + djac[i] * vj
        end
    end

    # Second-order update
    @inbounds for j in 1:n
        v2[j] = T(0.5) * (v[j] + a[j])
    end
    _gemv!(djac, fjac, v2)                    # djac := J*v2
    v2tv2 = _dot(v2, v2)
    @inbounds for i in 1:m
        djac[i] = T(0.5) * (fvec_new[i] - r1[i] - djac[i]) / v2tv2
    end

    @inbounds for j in 1:n
        v2j = v2[j]
        for i in 1:m
            fjac[i, j] = fjac[i, j] + djac[i] * v2j
        end
    end
    fjac
end
