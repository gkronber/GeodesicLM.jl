# -*- julia -*-
# src/smallblas.jl
# Pure-Julia replacements for the BLAS / LAPACK calls on the CPU solver path.
#
# `geodesiclm` works on an m×n Jacobian with a small number of parameters n and
# calls these kernels several times per iteration.  Many OpenBLAS interfaces
# (gemm, gemv, potrf, ...) allocate a work buffer under a global lock on every
# call, so fits running concurrently on different Julia threads serialize on
# that lock even with `BLAS.set_num_threads(1)`
# (https://github.com/OpenMathLib/OpenBLAS/issues/5589).  For these sizes plain
# loops are also faster on a single thread, and they keep the solver
# independent of the BLAS backend that happens to be loaded.
#
# The kernels are generic in the element type and accept views.  All matrices
# are column-major, so the inner loops run down columns.  The products are
# accumulated with `muladd`, which becomes a fused multiply-add where the
# hardware has one, as the optimized BLAS kernels do.

# x'y
function _dot(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        s = muladd(x[i], y[i], s)
    end
    return s
end

# Euclidean norm.  The plain sum of squares is used unless it overflowed or
# underflowed, in which case the vector is rescaled by its largest magnitude
# (as BLAS `nrm2` guards against).  NaN and Inf entries propagate.
function _nrm2(x::AbstractVector{T}) where {T<:AbstractFloat}
    s = _dot(x, x)
    (isfinite(s) && s >= floatmin(T)) && return sqrt(s)
    xmax = zero(T)
    @inbounds for i in eachindex(x)
        xmax = max(xmax, abs(x[i]))
    end
    (iszero(xmax) || !isfinite(xmax)) && return xmax
    t = zero(T)
    @inbounds @simd for i in eachindex(x)
        xi = x[i] / xmax
        t = muladd(xi, xi, t)
    end
    return xmax * sqrt(t)
end

# y += alpha * x, elementwise for vectors and matrices of the same shape
function _axpy!(alpha::T, x::AbstractArray{T}, y::AbstractArray{T}) where {T<:AbstractFloat}
    @inbounds @simd for i in eachindex(x, y)
        y[i] = muladd(alpha, x[i], y[i])
    end
    return y
end

# x *= alpha
function _scal!(x::AbstractArray{T}, alpha::T) where {T<:AbstractFloat}
    @inbounds @simd for i in eachindex(x)
        x[i] *= alpha
    end
    return x
end

# y = A * x
function _gemv!(y::AbstractVector{T}, A::AbstractMatrix{T},
                x::AbstractVector{T}) where {T<:AbstractFloat}
    m, n = size(A)
    (length(y) == m && length(x) == n) ||
        throw(DimensionMismatch("A is $m×$n, x has length $(length(x)), y has length $(length(y))"))
    fill!(y, zero(T))
    @inbounds for j in 1:n
        xj = x[j]
        @simd for i in 1:m
            y[i] = muladd(A[i, j], xj, y[i])
        end
    end
    return y
end

# y = A' * x
function _gemv_t!(y::AbstractVector{T}, A::AbstractMatrix{T},
                  x::AbstractVector{T}) where {T<:AbstractFloat}
    m, n = size(A)
    (length(y) == n && length(x) == m) ||
        throw(DimensionMismatch("A is $m×$n, x has length $(length(x)), y has length $(length(y))"))
    @inbounds for j in 1:n
        s = zero(T)
        @simd for i in 1:m
            s = muladd(A[i, j], x[i], s)
        end
        y[j] = s
    end
    return y
end

# C = A' * A, both triangles
function _syrk_t!(C::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    m, n = size(A)
    size(C) == (n, n) ||
        throw(DimensionMismatch("A is $m×$n, C is $(size(C, 1))×$(size(C, 2))"))
    @inbounds for j in 1:n, i in 1:j
        s = zero(T)
        @simd for k in 1:m
            s = muladd(A[k, i], A[k, j], s)
        end
        C[i, j] = s
        C[j, i] = s
    end
    return C
end

# Cholesky factorization A = U'U of the symmetric matrix stored in the upper
# triangle of `A`, like LAPACK `dpotrf` with `uplo = 'U'`: the upper triangle is
# overwritten with `U`, the strict lower triangle is neither read nor written.
# Returns 0 on success, or the index of the first pivot that is not positive
# (or NaN), where the factorization stops.
function _potrf_upper!(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A is not square"))
    @inbounds for j in 1:n
        s = A[j, j]
        @simd for k in 1:(j - 1)
            s = muladd(-A[k, j], A[k, j], s)
        end
        s > zero(T) || return j
        ujj = sqrt(s)
        A[j, j] = ujj
        for c in (j + 1):n
            t = A[j, c]
            @simd for k in 1:(j - 1)
                t = muladd(-A[k, j], A[k, c], t)
            end
            A[j, c] = t / ujj
        end
    end
    return 0
end

# b = U' \ b (forward substitution) with U the upper triangle of `A`
function _trsv_ut!(b::AbstractVector{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    @inbounds for j in eachindex(b)
        s = b[j]
        @simd for k in 1:(j - 1)
            s = muladd(-A[k, j], b[k], s)
        end
        b[j] = s / A[j, j]
    end
    return b
end

# b = U \ b (back substitution) with U the upper triangle of `A`
function _trsv_u!(b::AbstractVector{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat}
    @inbounds for j in reverse(eachindex(b))
        bj = b[j] / A[j, j]
        b[j] = bj
        @simd for k in 1:(j - 1)
            b[k] = muladd(-A[k, j], bj, b[k])
        end
    end
    return b
end

# b = (U'U) \ b with the factor from `_potrf_upper!` in the upper triangle of `A`
_potrs_upper!(b::AbstractVector{T}, A::AbstractMatrix{T}) where {T<:AbstractFloat} =
    _trsv_u!(_trsv_ut!(b, A), A)

# `any(isnan, A)` without the short circuit, so that the loop vectorizes.  The
# check runs on the m×n Jacobian and the m residuals every iteration, and NaN
# is the rare case, so scanning to the end is cheaper than branching.
function _hasnan(A::AbstractArray{T}) where {T<:AbstractFloat}
    r = false
    @inbounds @simd for i in eachindex(A)
        r |= isnan(A[i])
    end
    return r
end
