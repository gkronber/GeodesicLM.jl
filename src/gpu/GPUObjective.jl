# -*- julia -*-
# src/gpu/GPUObjective.jl
# User objective + GPU finite-difference Jacobian/acceleration (PLAN.md, M5).
#
# The user supplies an objective whose `f(x,p)` (and optionally `∂f/∂p`, and
# optionally the second directional derivative) are evaluated on the device. The
# kernels capture a tuple of user device buffers via the `data` field, launched
# per the contract below (per §9: pass a tuple of user buffers).

using KernelAbstractions

"""
    GPUObjective(fun!, grad!, data)

Wrapper for a GPU-evaluated objective. `fun!`, `grad!` and `avv!` (all optional
except `fun!`) are kernel *launchers* with the signatures

- `fun!(backend, x, fvec, data)` — write residuals `fvec` (length `m`) for
  parameters `x` (length `n`) of element type `T`, over `data` (a tuple of user
  device buffers, e.g. the predictor matrix and observations).
- `grad!(backend, x, fjac, data)` — write the `m×n` Jacobian `∂f/∂p`.
- `avv!(backend, x, v, acc, data)` — write the second directional derivative
  `d²f/dv²` (length `m`).

If `grad!`/`avv!` are `nothing`, finite-difference versions are computed from
`fun!` using `KAOps` (see `jac!`, `avv!`).
"""
struct GPUObjective{F,G,A,D}
    fun!::F
    grad!::G
    avv!::A
    data::D
end

GPUObjective(fun!; grad! = nothing, avv! = nothing, data = ()) =
    GPUObjective(fun!, grad!, avv!, data)

###############################################################################
# Small helper kernels
###############################################################################

# Add `val` to component `i` of device vector `x` (single writer).
@kernel function _addto_kernel!(x, i, val)
    k = @index(Global, Linear)
    if k == i
        x[k] += val
    end
end

# fjac[:, i] = (plus - minus) / h   (central difference)
@kernel function _fdcol_kernel!(fjac, plus, minus, m, i, invh)
    k = @index(Global, Linear)
    if k <= m
        fjac[k, i] = (plus[k] - minus[k]) * invh
    end
end

# fjac[:, i] = (plus - base) / h   (forward difference)
@kernel function _fdfwd_kernel!(fjac, plus, base, m, i, invh)
    k = @index(Global, Linear)
    if k <= m
        fjac[k, i] = (plus[k] - base[k]) * invh
    end
end

# acc = (2/h) * ((ftmp - fvec)/h - fjac*v)   (jacobian up-to-date path)
@kernel function _avv_jac_kernel!(acc, ftmp, fvec, jtv, m, h, invh)
    k = @index(Global, Linear)
    if k <= m
        acc[k] = 2.0f0 * invh * ((ftmp[k] - fvec[k]) * invh - jtv[k])
    end
end

# acc = (ftmp - 2*fvec + atmp)/h²   (jacobian stale path)
@kernel function _avv_n0_kernel!(acc, ftmp, fvec, atmp, m, invh2)
    k = @index(Global, Linear)
    if k <= m
        acc[k] = (ftmp[k] - 2 * fvec[k] + atmp[k]) * invh2
    end
end

###############################################################################
# GPU finite-difference Jacobian (writes into workspace.fjac)
###############################################################################

"""
    jac!(w::GPUWorkspace, obj::GPUObjective, center_diff=true, h1=1e-5)

Compute the Jacobian `m×n` into `w.fjac` on the device. If
`obj.grad!` is provided uses it; otherwise finite differences from `obj.fun!`
(central if `center_diff`, else forward). Parameter `x` is read from `w.x`.
"""
function jac!(w::GPUWorkspace, obj::GPUObjective, center_diff::Bool = true,
              h1::Real = 1.0e-5)
    T = eltype(w)
    be = w.backend
    n, m = w.n, w.m

    if obj.grad! !== nothing
        obj.grad!(be, w.x, w.fjac, obj.data)
        return w.fjac
    end

    xh = collect(w.x)                 # host copy of params (small n)
    epsmach = eps(T)
    for i in 1:n
        h = max(h1 * abs(xh[i]), epsmach)
        invh = one(T) / h
        if center_diff
            KAOps.copyto!(w.x_plus, w.x)
            _addto_kernel!(be)(w.x_plus, i, T(0.5) * h; ndrange = n)
            obj.fun!(be, w.x_plus, w.fvec_plus, obj.data)
            KAOps.copyto!(w.x_minus, w.x)
            _addto_kernel!(be)(w.x_minus, i, -T(0.5) * h; ndrange = n)
            obj.fun!(be, w.x_minus, w.fvec_minus, obj.data)
            ev = _fdcol_kernel!(be)(w.fjac, w.fvec_plus, w.fvec_minus, m, i, invh; ndrange = m)
            KAOps._sync(ev)
        else
            KAOps.copyto!(w.x_plus, w.x)
            _addto_kernel!(be)(w.x_plus, i, h; ndrange = n)
            obj.fun!(be, w.x_plus, w.fvec_plus, obj.data)
            ev = _fdfwd_kernel!(be)(w.fjac, w.fvec_plus, w.fvec, m, i, invh; ndrange = m)
            KAOps._sync(ev)
        end
    end
    return w.fjac
end

"""
    avv!(w::GPUWorkspace, obj::GPUObjective, v, jac_uptodate, h2=1e-5)

Compute the second directional derivative `d²f/dv²` into `w.acc` on the device.
Uses `obj.avv!` if provided, else finite differences from `obj.fun!`.
"""
function avv!(w::GPUWorkspace, obj::GPUObjective, v::AbstractVector, jac_uptodate::Bool,
              h2::Real = 1.0e-5)
    T = eltype(w)
    be = w.backend
    n, m = w.n, w.m

    if obj.avv! !== nothing
        obj.avv!(be, w.x, v, w.acc, obj.data)
        return w.acc
    end

    invh = one(T) / h2
    if jac_uptodate
        # x_tmp = x + h2*v ; f(x_tmp) -> ftmp
        KAOps.copyto!(w.xtmp, w.x)
        KAOps.axpy!(w.xtmp, T(h2), v)
        obj.fun!(be, w.xtmp, w.ftmp, obj.data)
        # jtv = fjac*v
        KAOps.mul!(w.acc_tmp, w.fjac, v)
        ev = _avv_jac_kernel!(be)(w.acc, w.ftmp, w.fvec, w.acc_tmp, m, h2, invh; ndrange = m)
        KAOps._sync(ev)
    else
        # x_tmp = x ± h2*v ; acc = (f(x+h) - 2 f(x) + f(x-h))/h²
        KAOps.copyto!(w.xtmp, w.x)
        KAOps.axpy!(w.xtmp, T(h2), v)
        obj.fun!(be, w.xtmp, w.ftmp, obj.data)
        KAOps.copyto!(w.xtmp, w.x)
        KAOps.axpy!(w.xtmp, T(-h2), v)
        obj.fun!(be, w.xtmp, w.acc_tmp, obj.data)
        ev = _avv_n0_kernel!(be)(w.acc, w.ftmp, w.fvec, w.acc_tmp, m, invh * invh; ndrange = m)
        KAOps._sync(ev)
    end
    return w.acc
end