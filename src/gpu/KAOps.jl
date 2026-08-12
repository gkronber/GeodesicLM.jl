# -*- julia -*-
# src/gpu/KAOps.jl
# Backend-agnostic GPU building-block kernels for GeodesicLM, written with
# KernelAbstractions.jl so the same code runs on the CPU test backend, CUDA,
# AMDGPU, Metal and oneAPI backends.
#
# This is the FIRST increment of the GPU port (see PLAN.md, milestones M0-M2).
# It implements the low-level dense linear-algebra primitives that the LM loop
# needs, each exposed as a small kernel + a thin host wrapper that picks the
# backend automatically from the arrays it is given. Correctness is established
# with unit tests that run on the CPU backend (no GPU required), and the kernels
# are deliberately simple (not highly tuned): the plan assumes the expensive
# part of a GPU run is the user's `f`/gradient evaluation, not this Linear
# Algebra.
#
# Reasoning behind the layout:
#   * Parameters `x` have length `n` (small), residuals `fvec` length `m`
#     (typically large), Jacobian `fjac` m×n (column-major).
#   * Every kernel takes a KernelAbstractions `Backend` as its first argument.
#     The owner passes the backend of the device arrays; on CPU during testing
#     that is `CPU()`, on a GPU it is `CUDA()`/`Metal()`/...
#   * Reductions (dot / norm / cost) are the only operations that need a result
#     back on the host; they use a two-stage tree reduction (per-workgroup
#     partials + a small final reduction) so they work on any backend.

module KAOps

using KernelAbstractions
using LinearAlgebra

export backend, backend_of, nthreads
# name clashes between kernel ops and Base/LinearAlgebra are expected; qualify
# or `using .KAOps` within a local scope where they are unambiguous.
export fill!, copyto!, axpy!, scale!
export dot!, norm2, nanflag
export mul!, AtA!
export alloc, device_array

###############################################################################
# Backend helpers
###############################################################################

"""
    backend(A_or_device)

Return the KernelAbstractions backend for an array (via `get_backend`) or, when
passed a backend value (e.g. `CPU()`, `CUDA()`), return it unchanged. Useful to
write backend-agnostic code that works whether the caller passes a backend or
an example array.
"""
backend(a) = get_backend(a)
backend(b::KernelAbstractions.KernelAbstractions.Backend) = b
# Some backends are singletons (CPU()); accept them directly too.
backend(::typeof(CPU())) = CPU()

"""
    backend_of(::Type{T}) where T

Backend to use for arrays of the given element type: `Array` -> `CPU()`, and a
hook for GPU array types (extend this when wiring up CUDA.jl / Metal.jl).
"""
backend_of(::Type{<:Array}) = CPU()

# On the CPU backend a kernel launch runs synchronously and returns `nothing`;
# on a GPU it returns an event that must be synchronized. `_sync` handles both.
_sync(ev) = (ev === nothing) ? nothing : synchronize(ev)

"""
    nthreads(backend)

Number of virtual threads to launch for embarrassingly parallel elementwise /
reduction kernels. A heuristic default that is portable across backends.
"""
nthreads(::CPU) = 256
nthreads(::Backend) = 256

"""
    alloc(::Type{T}, backend, dims...) -> device array

Allocate a device array of type `T` (e.g. `Array`, or a GPU array type) on the
given backend. On the CPU backend this is just `Array`. Extend `backend_of` and
this together to add real GPU array types. The returned array must work with
`get_backend` returning `backend` so KA can dispatch kernels onto it.
"""
alloc(::Type{T}, ::CPU, dims...) where {T} = Array{T}(undef, dims...)

# Convenience: device vector / matrix
device_array(T, b::CPU) = Array{T}(undef)
device_array(T, b, dims...) = alloc(T, b, dims...)

###############################################################################
# Elementwise kernels
###############################################################################

@kernel function _fill_kernel!(A, val)
    i = @index(Global, Linear)
    A[i] = val
end

@kernel function _copy_kernel!(dst, src)
    i = @index(Global, Linear)
    dst[i] = src[i]
end

@kernel function _axpy_kernel!(y, a, x)
    i = @index(Global, Linear)
    y[i] += a * x[i]
end

@kernel function _scale_kernel!(y, a, x)   # y = a .* x  (overwrite)
    i = @index(Global, Linear)
    y[i] = a * x[i]
end

"""
    fill!(y, val)
    copyto!(dst, src)
    axpy!(y, a, x)          # y .= y .+ a .* x
    scale!(y, a, x)         # y .= a .* x

In-place elementwise operations on device arrays, executed on the arrays' own
backend (CPU() during tests, GPU otherwise).
"""
function fill!(y, val)
    b = backend(y)
    ev = _fill_kernel!(b)(y, val; ndrange=length(y))
    _sync(ev)
    return y
end

function copyto!(dst, src)
    @assert length(dst) == length(src)
    b = backend(dst)
    ev = _copy_kernel!(b)(dst, src; ndrange=length(dst))
    _sync(ev)
    return dst
end

function axpy!(y, a, x)
    @assert length(y) == length(x)
    b = backend(y)
    ev = _axpy_kernel!(b)(y, a, x; ndrange=length(y))
    _sync(ev)
    return y
end

function scale!(y, a, x)
    @assert length(y) == length(x)
    b = backend(y)
    ev = _scale_kernel!(b)(y, a, x; ndrange=length(y))
    _sync(ev)
    return y
end

###############################################################################
# Matrix-vector products
###############################################################################

# y (m) = A (m×n) * x (n)
@kernel function _matvec_kernel!(y, A, x, m, n)
    i = @index(Global, Linear)
    if i <= m
        s = zero(eltype(y))
        @inbounds for j in 1:n
            s += A[i, j] * x[j]
        end
        y[i] = s
    end
end

# y (n) = A' (n×m) * x (m)
@kernel function _matTvec_kernel!(y, A, x, m, n)
    j = @index(Global, Linear)
    if j <= n
        s = zero(eltype(y))
        @inbounds for i in 1:m
            s += A[i, j] * x[i]
        end
        y[j] = s
    end
end

"""
    mul!(y, A, x)   # y = A*x   (A m×n, x n, y m)
    mul!(y, A', x)  # y = A'*x  (A m×n, x m, y n)

Matrix-vector products on device arrays, dispatched on the arrays' backend.
"""
function mul!(y, A, x)
    m, n = size(A)
    b = backend(y)
    ev = _matvec_kernel!(b)(y, A, x, m, n; ndrange=m)
    _sync(ev)
    return y
end

function mul!(y, A::Adjoint, x)
    A = parent(A)
    m, n = size(A)
    b = backend(y)
    ev = _matTvec_kernel!(b)(y, A, x, m, n; ndrange=n)
    _sync(ev)
    return y
end

###############################################################################
# Matrix product C = A' * A  (m×n matrices)   -- used for J'J
###############################################################################
@kernel function _AtA_kernel!(C, A, B, m, n)
    idx = @index(Global, Linear)
    if idx <= n * n
        ci = (idx - 1) % n + 1
        cj = (idx - 1) ÷ n + 1
        s = zero(eltype(C))
        @inbounds for k in 1:m
            s += A[k, ci] * B[k, cj]
        end
        C[ci, cj] = s
    end
end

"""
    AtA!(C, A, B)  # C (n×n) = A' * B  (A, B are m×n)

Used to build `J'J` from the Jacobian on the device. Returns `C`.
"""
function AtA!(C, A, B)
    m, n = size(A)
    b = backend(C)
    ev = _AtA_kernel!(b)(C, A, B, m, n; ndrange=n * n)
    _sync(ev)
    return C
end

###############################################################################
# Reductions: dot / norm2 (grid-strided partial + serial final, portable)
###############################################################################
# Stage 1: nt threads, thread i accumulates the strided dot product over the
# data into tmp[i]. No local memory / barriers, so it is correct on every
# backend (including the CPU test backend).
@kernel function _dot_stride_kernel!(tmp, a, b, len, nt)
    i = @index(Global, Linear)
    if i <= nt
        s = zero(eltype(a))
        @inbounds for q in i:nt:len
            s += a[q] * b[q]
        end
        tmp[i] = s
    end
end

# Stage 2: out[1] = Σ tmp (single thread; nt is small)
@kernel function _reduce_sum_kernel!(out, tmp)
    i = @index(Global, Linear)
    if i == 1
        s = zero(eltype(tmp))
        @inbounds for j in eachindex(tmp)
            s += tmp[j]
        end
        out[1] = s
    end
end

"""
    dot!(out, a, b)   # out[1] = a⋅b   (out is a length-1 device array)

Two-stage reduction computing the dot product on the device. Used for
`cost = ½'s's⋅f` and `norm`. Returns `out`.
"""
function dot!(out, a, b)
    @assert length(a) == length(b)
    baux = backend(out)
    len = length(a)
    len > 0 || (fill!(out, zero(eltype(a))); return out)

    nt = min(nthreads(baux), len)      # threads to use
    tmp = alloc(eltype(a), baux, nt)
    nev = _dot_stride_kernel!(baux)(tmp, a, b, len, nt; ndrange=nt)
    _sync(nev)

    nev2 = _reduce_sum_kernel!(baux)(out, tmp; ndrange=1)
    _sync(nev2)
    return out
end

"""
    norm2(out, a)   # out[1] = a⋅a  (squared 2-norm)
"""
function norm2(out, a)
    return dot!(out, a, a)
end

"""
    nanflag(a) -> Bool

Whether any element of a (device array) is NaN/Inf. Copies to host; small arrays
only (used for NaN guards on x / fvec / fjac). Returns a host Bool.
"""
function nanflag(a)
    arr = collect(a)                 # host copy (small arrays in the LM loop)
    return any(x -> !isfinite(x), arr)
end

end # module KAOps
