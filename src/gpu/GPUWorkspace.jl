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
    x::Vector{T}
    x_new::Vector{T}
    x_best::Vector{T}
    v::Vector{T}
    vold::Vector{T}
    a::Vector{T}
    # residuals (m)
    fvec::Vector{T}
    fvec_new::Vector{T}
    fvec_best::Vector{T}
    jv::Vector{T}
    acc::Vector{T}             # second directional derivative (m)
    # matrices
    fjac::Matrix{T}      # m×n
    jtj::Matrix{T}       # n×n
    g::Matrix{T}         # n×n
    dtd::Matrix{T}       # n×n
    # scratch (n)
    tmp1::Vector{T}
    tmp2::Vector{T}
    tmp3::Vector{T}
    # finite-difference temporaries
    x_plus::Vector{T}        # n
    x_minus::Vector{T}       # n
    fvec_plus::Vector{T}     # m
    fvec_minus::Vector{T}    # m
    xtmp::Vector{T}          # n
    ftmp::Vector{T}          # m
    acc_tmp::Vector{T}       # m
    grad::Vector{T}          # n
    # trust-region scratch
    jtilde::Matrix{T}        # m×n
    gt::Matrix{T}            # n×n
    gradC::Vector{T}         # n
    dgqtA::Matrix{T}         # n×n
    z::Vector{T}             # n
    wa1::Vector{T}           # n
    wa2::Vector{T}           # n
    # scalars
    scalar::Vector{T}        # length 1
    scalar2::Vector{T}       # length 1
    info::Vector{Int}        # length 1
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