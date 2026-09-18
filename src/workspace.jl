# -*- julia -*-
# file workspace.jl
#
# Reusable scratch storage for `geodesiclm`.
#
# `geodesiclm` is used in workloads (symbolic regression, model search) that
# fit millions of small models, each with a different number of parameters.
# Allocating the algorithm's ~20 work arrays per call, and its per-iteration
# temporaries, dominates the run time there.  A `GeodesicLMWorkspace` holds all
# of them; it grows to the largest `(m, n)` it has seen and hands out views of
# exactly the requested size, so a sequence of differently-sized problems
# allocates only while the sizes are still growing.

"""
    GeodesicLMWorkspace{T}()
    GeodesicLMWorkspace{T}(m, n)

Reusable scratch storage for [`geodesiclm`](@ref).  Pass the same workspace to
successive calls to make them allocation-free; it grows automatically when a
larger problem is encountered.  A workspace is *not* thread-safe -- use one per
task.
"""
mutable struct GeodesicLMWorkspace{T<:AbstractFloat}
    m::Int          # capacity: number of residuals
    n::Int          # capacity: number of parameters
    # m-sized
    acc::Vector{T}
    fvec::Vector{T}
    fvec_new::Vector{T}
    fvec_best::Vector{T}
    jv::Vector{T}
    ftmp::Vector{T}
    mtmp1::Vector{T}
    mtmp2::Vector{T}
    # n-sized
    x::Vector{T}
    v::Vector{T}
    vold::Vector{T}
    a::Vector{T}
    x_new::Vector{T}
    x_best::Vector{T}
    ntmp1::Vector{T}
    ntmp2::Vector{T}
    ntmp3::Vector{T}
    # matrices
    fjac::Matrix{T}     # m x n
    jtj::Matrix{T}      # n x n
    g::Matrix{T}        # n x n
    dtd::Matrix{T}      # n x n
end

GeodesicLMWorkspace{T}() where {T<:AbstractFloat} = GeodesicLMWorkspace{T}(0, 0)

function GeodesicLMWorkspace{T}(m::Int, n::Int) where {T<:AbstractFloat}
    GeodesicLMWorkspace{T}(m, n,
        zeros(T, m), zeros(T, m), zeros(T, m), zeros(T, m),
        zeros(T, m), zeros(T, m), zeros(T, m), zeros(T, m),
        zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
        zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
        zeros(T, m, n), zeros(T, n, n), zeros(T, n, n), zeros(T, n, n))
end

# Grow the workspace so it can serve a problem of size (m, n).  Capacities only
# ever grow, so a run over many differently-sized problems settles after the
# largest one has been seen.
function _ensure_capacity!(ws::GeodesicLMWorkspace{T}, m::Int, n::Int) where {T}
    (m <= ws.m && n <= ws.n) && return ws
    m = max(m, ws.m)
    n = max(n, ws.n)
    for f in (:acc, :fvec, :fvec_new, :fvec_best, :jv, :ftmp, :mtmp1, :mtmp2)
        setfield!(ws, f, zeros(T, m))
    end
    for f in (:x, :v, :vold, :a, :x_new, :x_best, :ntmp1, :ntmp2, :ntmp3)
        setfield!(ws, f, zeros(T, n))
    end
    ws.fjac = zeros(T, m, n)
    ws.jtj = zeros(T, n, n)
    ws.g = zeros(T, n, n)
    ws.dtd = zeros(T, n, n)
    ws.m = m
    ws.n = n
    ws
end
