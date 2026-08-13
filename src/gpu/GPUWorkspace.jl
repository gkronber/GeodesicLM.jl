# -*- julia -*-
# src/gpu/GPUWorkspace.jl
# Backend-aware device buffer workspace + OncePerTask lazy cache for the GPU
# port (PLAN.md, M4). Mirrors the CPU `GLMWorkspace` but every buffer is a
# device array (CPU()-backed `Array` during tests; `MtlArray`/`CuArray` etc.
# once `KAOps.alloc` is extended for those backends).
#
# The orchestrator (M6/M7) uses the buffers in place so repeated
# `geodesiclm(...)` calls on the same task with the same `(n, m, T, backend)`
# reuse the same buffers instead of re-allocating — and distinct tasks / sizes /
# backends do not interfere (OncePerTask).

using KernelAbstractions

"""
    GPUWorkspace(n, m, T, backend)

Pre-allocated device buffers for an `n`-param, `m`-residual optimization of
element type `T` running on `backend`. Every array is allocated once and reused
across `geodesiclm` calls. See `plan.md` M4.

# Fields (all device arrays)

- state/step: `x`, `x_new`, `x_best`, `v`, `vold`, `a`  (length `n`)
- residuals: `fvec`, `fvec_new`, `fvec_best`, `jv` (length `m`)
- matrices: `fjac` (m×n), `jtj`, `g`, `dtd` (n×n)
- scratch vectors/matrices for the LM step and finite differences
- `scalar`, `scalar2` length-1 buffers for reductions/cost/norm
- `info` length-1 `Int` for the in-place Cholesky failure flag
"""
struct GPUWorkspace{T,B}
    n::Int
    m::Int
    backend::B
    # state / step (n)
    x::AbstractVector{T}
    x_new::AbstractVector{T}
    x_best::AbstractVector{T}
    v::AbstractVector{T}
    vold::AbstractVector{T}
    a::AbstractVector{T}
    # residuals (m)
    fvec::AbstractVector{T}
    fvec_new::AbstractVector{T}
    fvec_best::AbstractVector{T}
    jv::AbstractVector{T}
    acc::AbstractVector{T}             # second directional derivative (m)
    # matrices
    fjac::AbstractMatrix{T}      # m×n
    jtj::AbstractMatrix{T}       # n×n
    g::AbstractMatrix{T}         # n×n
    dtd::AbstractMatrix{T}       # n×n
    # scratch (n)
    tmp1::AbstractVector{T}
    tmp2::AbstractVector{T}
    tmp3::AbstractVector{T}
    # finite-difference temporaries
    x_plus::AbstractVector{T}        # n
    x_minus::AbstractVector{T}       # n
    fvec_plus::AbstractVector{T}     # m
    fvec_minus::AbstractVector{T}    # m
    xtmp::AbstractVector{T}          # n
    ftmp::AbstractVector{T}          # m
    acc_tmp::AbstractVector{T}       # m
    grad::AbstractVector{T}          # n
    # trust-region scratch
    jtilde::AbstractMatrix{T}        # m×n
    gt::AbstractMatrix{T}            # n×n
    gradC::AbstractVector{T}         # n
    dgqtA::AbstractMatrix{T}         # n×n
    z::AbstractVector{T}             # n
    wa1::AbstractVector{T}           # n
    wa2::AbstractVector{T}           # n
    # scalars
    scalar::AbstractVector{T}        # length 1
    scalar2::AbstractVector{T}       # length 1
    info::AbstractVector{Int}        # length 1
end

function GPUWorkspace(n::Int, m::Int, ::Type{T}, backend) where {T}
    # NOTE: allocate a fresh array per field — the fields must NOT alias one
    # another (an earlier version shared one `vecn`/`vecm` across many fields,
    # which made e.g. `fvec_plus` alias `fvec` and produced wrong kernels).
    vn(tag) = KAOps.alloc(T, backend, n)
    vm(tag) = KAOps.alloc(T, backend, m)
    nn(tag) = KAOps.alloc(T, backend, n, n)
    mn(tag) = KAOps.alloc(T, backend, m, n)
    one = KAOps.alloc(T, backend, 1)
    GPUWorkspace{T,typeof(backend)}(
        n, m, backend,
        vn(:x), vn(:x_new), vn(:x_best), vn(:v), vn(:vold), vn(:a),
        vm(:fvec), vm(:fvec_new), vm(:fvec_best), vm(:jv), vm(:acc),
        mn(:fjac), nn(:jtj), nn(:g), nn(:dtd),
        vn(:tmp1), vn(:tmp2), vn(:tmp3),
        vn(:x_plus), vn(:x_minus), vm(:fvec_plus), vm(:fvec_minus),
        vn(:xtmp), vm(:ftmp), vm(:acc_tmp),
        vn(:grad),
        mn(:jtilde), nn(:gt), vn(:gradC),
        nn(:dgqtA), vn(:z), vn(:wa1), vn(:wa2),
        one, KAOps.alloc(T, backend, 1), KAOps.alloc(Int, backend, 1),
    )
end

# Eltype/backend convenience
Base.eltype(::GPUWorkspace{T}) where {T} = T

###############################################################################
# OncePerTask lazy cache (keyed by n, m, element type, backend)
###############################################################################

const _WS_GPU_KEY = :GeodesicLM_workspace_gpu

function _get_gpu_workspace(n::Int, m::Int, ::Type{T}, backend) where {T}
    tls = task_local_storage()
    cache = if haskey(tls, _WS_GPU_KEY)
        tls[_WS_GPU_KEY]::Dict{Tuple{Int,Int,DataType,Any},GPUWorkspace}
    else
        c = Dict{Tuple{Int,Int,DataType,Any},GPUWorkspace}()
        tls[_WS_GPU_KEY] = c
        c
    end
    return get!(cache, (n, m, T, backend)) do
        GPUWorkspace(n, m, T, backend)
    end
end