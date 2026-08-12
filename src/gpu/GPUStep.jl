# -*- julia -*-
# src/gpu/GPUStep.jl
# One on-device LM iteration-interior (PLAN.md, M6): builds jtj, g = jtj+λ·dtd,
# factors with the KAOps `cholesky!`, solves for the Gauss-Newton step `v`, and
# computes `cos_alpha`, `pred_red`, the acceleration step `a`, `av`, and the
# new cost / acceptance — all with large arrays staying on the device; only a
# handful of scalars round-trip to the host.
#
# This is the single most important correctness gate (a snapshot test at
# M6 reproduces the CPU routine's `v`, `av`, `cos_alpha`, `pred_red`).

# Read one scalar (a device length-1 reduction buffer) onto the host.
# `Array` is the identity on `Array`, and copies NdArray→host on GPUs.
@inline _hscalar(buf) = Array(buf)[1]

@inline _dotv(w::GPUWorkspace, v) = (KAOps.dot!(w.scalar, v, v); _hscalar(w.scalar))

# Host acceptance rule using precomputed scalars (faithful port of accept.jl;
# the beta dots are computed on-device in `gpu_step!`). beta already clamped
# to (max 1) of (1 - cos(v, vold)) by the caller.
function _accept(C::T, Cnew::T, Cbest::T, ibold::Int, beta::T) where {T}
    if Cnew <= C
        return 1
    end
    ibold == 0 && return -1
    if ibold == 1
        return (beta * Cnew <= Cbest) ? 1 : -1
    elseif ibold == 2
        return (beta * beta * Cnew <= Cbest) ? 1 : -1
    elseif ibold == 3
        return (beta * Cnew <= C) ? 1 : -1
    else # ibold == 4
        return (beta * beta * Cnew <= C) ? 1 : -1
    end
end

"""
    gpu_step!(w::GPUWorkspace, obj::GPUObjective, lam, C, Cbest,
              prev_accepted=0, ibold=1, avmax=10.0, jac_uptodate=true, h2=1.0e-5)

Perform one LM step entirely on the device (buffers in `w`), given the current
parameters `x` (`w.x`), residuals `fvec` (`w.fvec`), Jacobian `fjac`
(`w.fjac`), damping `lam` and damping matrix `w.dtd`. Uses `obj` (analytic
`avv!`, or finite-difference fallback) for the geodesic acceleration. Writes
`v`, `a`, `x_new`, `fvec_new`, `jtj`, `g` into the workspace.

Returns a `NamedTuple`: `ok`, `cos_alpha`, `pred_red`, `dirder`, `delta`, `av`,
`Cnew`, `rho`, `accepted`.
"""
function gpu_step!(w::GPUWorkspace, obj::GPUObjective, lam::T, C::T, Cbest::T,
                   prev_accepted::Int = 0, ibold::Int = 1, avmax::T = T(10.0),
                   jac_uptodate::Bool = true, h2::T = T(1.0e-5)) where {T}
    be = w.backend
    n, m = w.n, w.m
    nev = 0   # number of user `fun!` invocations this step (for nfev bookkeeping)

    # 1. jtj = J'J ; g = jtj + λ·dtd
    KAOps.AtA!(w.jtj, w.fjac, w.fjac)
    KAOps.copyto!(w.g, w.jtj)
    KAOps.axpy!(w.g, lam, w.dtd)

    # 2. Factor g in place (upper triangle).
    ok = KAOps.cholesky!(w.g, w.info)
    if !ok
        return (ok = false, cos_alpha = T(NaN), pred_red = T(NaN),
                dirder = T(NaN), delta = T(NaN), av = T(NaN), Cnew = T(NaN),
                rho = T(NaN), actred = T(NaN), fnew_nan = true,
                nev = 0, accepted = min(prev_accepted - 1, -1))
    end

    # 3. v = g \ (-J' fvec)
    KAOps.mul!(w.tmp3, adjoint(w.fjac), w.fvec)   # tmp3 = J' fvec
    KAOps.scale!(w.tmp3, T(-1), w.tmp3)
    KAOps.solve_chol!(w.v, w.g, w.tmp3)

    # 4. pred_red / dirder  (tmp1 = jtj·v, tmp2 = dtd·v)
    KAOps.mul!(w.tmp1, w.jtj, w.v)
    KAOps.mul!(w.tmp2, w.dtd, w.v)
    KAOps.dot!(w.scalar, w.v, w.tmp1)
    temp1 = T(0.5) * _hscalar(w.scalar) / C
    KAOps.dot!(w.scalar, w.v, w.tmp2)
    td = _hscalar(w.scalar)                 # dot(v, dtd·v)  (host, reused)
    temp2 = T(0.5) * lam * td / C
    pred_red = temp1 + T(2) * temp2
    dirder = T(-1) * (temp1 + temp2)

    # 5. cos_alpha = |fvec · jv| / (|fvec|·|jv|)
    KAOps.mul!(w.jv, w.fjac, w.v)
    KAOps.dot!(w.scalar, w.fvec, w.jv);  dfj = _hscalar(w.scalar)
    KAOps.norm2(w.scalar, w.fvec);       nf = _hscalar(w.scalar)
    KAOps.norm2(w.scalar, w.jv);         nj = _hscalar(w.scalar)
    cos_alpha = abs(dfj) / (sqrt(nf) * sqrt(nj))

    delta = sqrt(td)     # imethod < 10

    # 6. acceleration step a = -g \ (J' acc)
    if obj.avv! !== nothing
        obj.avv!(be, w.x, w.v, w.acc, obj.data)
    else
        avv!(w, obj, w.v, jac_uptodate, h2)
        nev += jac_uptodate ? 1 : 2
    end
    if !KAOps.nanflag(w.acc)
        KAOps.mul!(w.a, adjoint(w.fjac), w.acc)
        KAOps.scale!(w.a, T(-1), w.a)
        KAOps.solve_chol!(w.a, w.g, w.a)
    else
        KAOps.fill!(w.a, T(0))
    end

    # 7. av = sqrt(a'·dtd·a / v'·dtd·v)
    KAOps.mul!(w.tmp1, w.dtd, w.a)
    KAOps.dot!(w.scalar, w.a, w.tmp1);  na = _hscalar(w.scalar)
    av = sqrt(na / td)

    # 8. candidate x_new = x + v + ½a ; recompute residuals/cost
    if av <= avmax
        KAOps.copyto!(w.x_new, w.x)
        KAOps.axpy!(w.x_new, T(1), w.v)
        KAOps.axpy!(w.x_new, T(0.5), w.a)
        obj.fun!(be, w.x_new, w.fvec_new, obj.data)
        nev += 1
        KAOps.norm2(w.scalar, w.fvec_new)
        Cnew = T(0.5) * _hscalar(w.scalar)
        if !KAOps.nanflag(w.fvec_new)
            actred = T(1) - Cnew / C
            rho = pred_red != T(0) ? actred / pred_red : T(0)
            fnew_nan = false
            # bold criterion: beta = clamp01(1 - cos(v, vold))
            if _dotv(w, w.vold) == T(0)      # vold zero (first step)
                beta = T(1)
            else
                KAOps.mul!(w.tmp3, w.dtd, w.vold)     # tmp3 = dtd·vold
                KAOps.dot!(w.scalar, w.v, w.tmp3);  ben = _hscalar(w.scalar)
                KAOps.dot!(w.scalar, w.vold, w.tmp3); den = _hscalar(w.scalar)
                beta = ben / sqrt(td * den)
                beta = min(T(1), T(1) - beta)
            end
            accepted = _accept(C, Cnew, Cbest, ibold, beta)
        else
            actred = T(0); rho = T(0); fnew_nan = true
            accepted = min(prev_accepted - 1, -1)
        end
    else
        Cnew = T(NaN); rho = T(NaN); actred = T(NaN); fnew_nan = true
        accepted = min(prev_accepted - 1, -1)
    end

    return (ok = true, cos_alpha, pred_red, dirder, delta, av, Cnew, rho,
            actred, fnew_nan, nev, accepted)
end