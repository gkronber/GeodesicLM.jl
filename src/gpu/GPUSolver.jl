# -*- julia -*-
# src/gpu/GPUSolver.jl
# M7: the full `geodesiclm` orchestrator for a `GPUObjective`. A faithful port
# of `geodesiclm_alg.jl` control flow, but the user's `f`/gradient/avv are
# device kernels and every array op runs through KAOps. The host carries only
# control flow and a few scalars per iteration.

###############################################################################
# Scalar λ/δ helpers (faithful ports of lambda.jl; array-dependent quantities
# are precomputed on-device and passed in as host scalars)
###############################################################################

# cos(v, vold) in the dtd metric, computed on-device (ports the cosine term of
# update_lam_umrigar).
function gpu_cos_v_vold!(w::GPUWorkspace)
    T = eltype(w)
    # tmp3 = dtd·vold ; tmp2 = dtd·v
    KAOps.mul!(w.tmp3, w.dtd, w.vold)
    KAOps.mul!(w.tmp2, w.dtd, w.v)
    KAOps.dot!(w.scalar, w.v, w.tmp3);  ben = _hscalar(w.scalar)   # v'·dtd·vold
    KAOps.dot!(w.scalar, w.vold, w.tmp3); den = _hscalar(w.scalar) # vold'·dtd·vold
    KAOps.dot!(w.scalar, w.v, w.tmp2);  dap = _hscalar(w.scalar)   # v'·dtd·v
    return ben / sqrt(dap * den)
end

function _lam_umrigar_scalars(lam::T, accepted::Int, a_param::T, C::T, Cnew::T,
                              cos_on::T) where {T}
    amemory = T(exp(-1.0 / 5.0))
    if accepted >= 0
        if Cnew <= C
            a_param = cos_on > T(0) ? amemory * a_param + (T(1) - amemory) :
                                       amemory * a_param + T(0.5) * (T(1) - amemory)
        else
            a_param = amemory * a_param + T(0.5) * (T(1) - amemory)
        end
        factor = min(T(100), max(T(1.1), T(1) / (T(2.2e-16) + T(1) -
                       abs(T(2) * a_param - T(1)))^2))
        if Cnew <= C && cos_on >= T(0)
            lam = lam / factor
        elseif Cnew > C
            lam = lam * sqrt(factor)
        end
    else
        a_param = amemory * a_param
        factor = min(T(100), max(T(1.1), T(1) / (T(2.2e-16) + T(1) -
                       abs(T(2) * a_param - T(1)))^2))
        if cos_on > T(0)
            lam = lam * sqrt(factor)
        else
            lam = lam * factor
        end
    end
    return (lam, a_param)
end

# pnorm = sqrt(v'·dtd·v) on-device (port of update_delta_more's pnorm).
function gpu_dtd_pnorm!(w::GPUWorkspace, v)
    KAOps.mul!(w.tmp2, w.dtd, v)
    KAOps.dot!(w.scalar, v, w.tmp2)
    return sqrt(_hscalar(w.scalar))
end

function _delta_more_scalars(delta::T, lam::T, pnorm::T, rho::T, C::T, Cnew::T,
                             actred::T, dirder::T, av::T, avmax::T) where {T}
    if rho > T(0.25)
        temp = (lam > T(0) && rho < T(0.75)) ? T(1) : T(2) * pnorm / delta
    else
        temp = actred >= T(0) ? T(0.5) : T(0.5) * dirder / (dirder + T(0.5) * actred)
        if T(0.01) * Cnew >= C || temp < T(0.1)
            temp = T(0.1)
        end
    end
    if av > avmax
        temp = min(temp, max(avmax / av, T(0.1)))
    end
    delta = temp * min(delta, T(10) * pnorm)
    lam = lam / temp
    return (delta, lam)
end

###############################################################################
# Broyden Jacobian update (port of updatejac.jl; rank-1 updates on-device)
###############################################################################
function gpu_update_jac!(w::GPUWorkspace)
    T = eltype(w)
    # fjac·v -> jv ; r1 = fvec + 0.5 fjac·v + 0.125 acc  (in acc_tmp)
    KAOps.mul!(w.jv, w.fjac, w.v)
    KAOps.copyto!(w.acc_tmp, w.fvec)
    KAOps.axpy!(w.acc_tmp, T(0.5), w.jv)
    KAOps.axpy!(w.acc_tmp, T(0.125), w.acc)
    # djac1 = 2*(r1 - fvec - 0.5 fjac·v)/dot(v,v)   (in ftmp)
    KAOps.copyto!(w.ftmp, w.acc_tmp)
    KAOps.axpy!(w.ftmp, T(-1), w.fvec)
    KAOps.axpy!(w.ftmp, T(-0.5), w.jv)
    dvv = _dotv(w, w.v)
    KAOps.scale!(w.ftmp, T(2) / dvv, w.ftmp)
    KAOps.rank1_update!(w.fjac, w.ftmp, w.v, T(0.5))
    # v2 = 0.5*(v + a)  (in tmp1)
    KAOps.copyto!(w.tmp1, w.v)
    KAOps.scale!(w.tmp1, T(0.5), w.tmp1)
    KAOps.axpy!(w.tmp1, T(0.5), w.a)
    # fjac·v2 -> jv
    KAOps.mul!(w.jv, w.fjac, w.tmp1)
    # djac2 = 0.5*(fvec_new - r1 - fjac·v2)/dot(v2,v2)   (in ftmp)
    KAOps.copyto!(w.ftmp, w.fvec_new)
    KAOps.axpy!(w.ftmp, T(-1), w.acc_tmp)     # r1
    KAOps.axpy!(w.ftmp, T(-1), w.jv)          # fjac·v2
    dv2 = _dotv(w, w.tmp1)
    KAOps.scale!(w.ftmp, T(0.5) / dv2, w.ftmp)
    KAOps.rank1_update!(w.fjac, w.ftmp, w.tmp1, T(1))
    return w.fjac
end

###############################################################################
# Trust region step (port of trust_region + dgqt; n-sized arrays on the host,
# the m×n Jacobian scaling stays on-device)
###############################################################################
function gpu_trust_region!(w::GPUWorkspace, fvec, fjac, dtd, delta)
    T = eltype(w)
    # s[j] = sqrt(dtd[j,j]); jtilde = fjac scaled by column
    s = [sqrt(Array(dtd)[j, j]) for j in 1:w.n]
    KAOps.scale_cols!(w.jtilde, fjac, s)
    # gradCtilde = fvec'·jtilde  (n-vector) ; g = jtilde'·jtilde (n×n)
    KAOps.mul!(w.gradC, adjoint(w.jtilde), w.fvec)
    KAOps.AtA!(w.g, w.jtilde, w.jtilde)
    # Solve the trust-region subproblem on the (small) n-sized arrays; the
    # returned Lagrange multiplier becomes λ in the jtj+λ·dtd step (matches
    # the CPU port, which discards the trust-region v and recomputes it via
    # the Cholesky step with `lam = trust_region(...)`).
    gh = Array(w.g); bh = Array(w.gradC)
    vh, lam, _info, _f = dgqt(w.n, gh, vec(bh), delta, 1.0e-3, 1.0e-3, 10, T(1))
    KAOps.copyto!(w.v, vh)
    return T(lam)
end

###############################################################################
# Convergence check (port of converge.jl; gradient norm on-device, small x, xnew
# copied to the host)
###############################################################################
function gpu_convergence_check!(w::GPUWorkspace, accepted::Int, counter::Int,
                               C::T, Cnew::T, lam::T, xh, xnewh,
                               nfev::Int, maxfev::Int, njev::Int, maxjev::Int,
                               naev::Int, maxaev::Int, maxlam::T, minlam::T,
                               artol::T, Cgoal::T, gtol::T, xtol::T, xrtol::T,
                               ftol::T, frtol::T, cos_alpha::T) where {T}
    n = w.n
    if maxfev > 0 && nfev >= maxfev
        return (-2, 0)
    end
    if maxjev > 0 && njev >= maxjev
        return (-3, counter)
    end
    if maxaev > 0 && naev >= maxaev
        return (-4, counter)
    end
    if maxlam > 0 && lam >= maxlam
        return (-5, counter)
    end
    if minlam > 0 && lam > 0 && lam <= minlam
        counter += 1
        return counter >= 3 ? (-6, counter) : (0, counter)
    end
    if artol > 0 && cos_alpha <= artol
        return (1, counter)
    end
    # gradient norm on-device: grad = -(fvec'·fjac); norm² = |fjac'·fvec|²
    KAOps.mul!(w.tmp1, adjoint(w.fjac), w.fvec)
    KAOps.norm2(w.scalar, w.tmp1)
    if sqrt(_hscalar(w.scalar)) <= gtol
        return (3, counter)
    end
    if C < Cgoal
        return (2, counter)
    end
    if accepted < 0
        return (0, 0)
    end
    # step size (n-vector diff on the host)
    if sqrt(sum((xnewh .- xh) .^ 2)) < xtol
        return (4, counter)
    end
    # relative parameter change
    conv = true
    for i in 1:n
        if abs(xh[i] - xnewh[i]) > xrtol * abs(xh[i]) || xnewh[i] != xnewh[i]
            conv = false
            break
        end
    end
    if conv
        return (5, counter)
    end
    if (C - Cnew) <= ftol && (C - Cnew) >= 0.0
        counter += 1
        return counter >= 3 ? (6, counter) : (0, counter)
    end
    if (C - Cnew) <= (frtol * C) && (C - Cnew) >= 0.0
        counter += 1
        return counter >= 3 ? (7, counter) : (0, counter)
    end
    return (0, 0)
end
###############################################################################
# Orchestrator: geodesiclm(::GPUObjective, ...)
###############################################################################

"""
    geodesiclm(obj::GPUObjective; x, fvec, n, m, kwargs...)

GPU port of the Geodesic-Levenberg-Marquardt optimizer (see `geodesiclm`
for the CPU version). `obj::GPUObjective` supplies the residuals / Jacobian /
second-derivative as device kernels. `x` and `fvec` are device (or CPU)
arrays that are mutated with the best fit found; everything is computed on
`x`'s backend. Returns `(x, fvec, niters, nfev, njev, naev, converged)`.

Most `kwargs` match the CPU routine: `analytic_jac`, `analytic_Avv`,
`center_diff`, `h1`, `h2`, `dtd`, `damp_mode`, `maxiter`, `maxfev`, `maxjev`,
`maxaev`, `maxlam`, `minlam`, `artol`, `Cgoal`, `gtol`, `xtol`, `xrtol`,
`ftol`, `frtol`, `imethod`, `iaccel`, `ibold`, `ibroyden`, `initialfactor`,
`factoraccept`, `factorreject`, `avmax`, `callback`, `ws`.
"""
function geodesiclm(obj::GPUObjective; x, fvec, n::Int, m::Int,
                    callback = nothing, info = 0,
                    analytic_jac::Bool = obj.grad! !== nothing,
                    analytic_Avv::Bool = obj.avv! !== nothing,
                    center_diff::Bool = true, h1 = 1.0e-5, h2 = 1.0e-5,
                    dtd = nothing, damp_mode::Int = 0,
                    maxiter::Int = 100, maxfev::Int = 0, maxjev::Int = 0,
                    maxaev::Int = 0, maxlam::Float64 = -1.0, minlam::Float64 = -1.0,
                    artol::Float64 = 0.0, Cgoal::Float64 = 0.0,
                    gtol::Float64 = 1.0e-8, xtol::Float64 = 1.0e-8,
                    xrtol::Float64 = 1.0e-8, ftol::Float64 = 1.0e-8,
                    frtol::Float64 = 1.0e-8, imethod::Int = 0, iaccel::Int = 1,
                    ibold::Int = 1, ibroyden::Int = 1,
                    initialfactor::Float64 = 100.0, factoraccept::Float64 = 2.0,
                    factorreject::Float64 = 2.0, avmax::Float64 = 10.0,
                    ws = nothing)
    T = eltype(x)
    be = KAOps.backend_of(typeof(x))
    W = ws === nothing ? _get_gpu_workspace(n, m, T, be) : ws
    (length(W.v) == n && length(W.acc) == m) || error(
        "GPUWorkspace sized for (n=$(W.n), m=$(W.m)) does not match (n=$n, m=$m)")

    # ---- init scalars (mirror CPU) ----
    KAOps.fill!(W.v, zero(T)); KAOps.fill!(W.vold, zero(T)); KAOps.fill!(W.a, zero(T))
    lam = T(0); delta = T(0); cos_alpha = T(1); av = T(0); a_param = T(0.5)
    pred_red = T(0); dirder = T(0); actred = T(0); rho = T(0)
    C = T(0); Cnew = T(0); converged = 0; nfev = 0; njev = 0; naev = 0
    counter = 0; accepted = 0; niters = 0
    jac_uptodate = true; jac_force_update = false

    # copy caller arrays into workspace; evaluate f at x
    KAOps.copyto!(W.x, x)
    obj.fun!(be, W.x, W.fvec, obj.data)
    nfev += 1
    C = T(0.5) * _dotv(W, W.fvec)
    if KAOps.nanflag(W.fvec)
        converged = -11; maxiter = 0
    end
    Cbest = C
    KAOps.copyto!(W.fvec_best, W.fvec); KAOps.copyto!(W.x_best, W.x)

    # initial Jacobian
    if analytic_jac && obj.grad! !== nothing
        obj.grad!(be, W.x, W.fjac, obj.data); njev += 1
    else
        jac!(W, obj, center_diff, h1)
        nfev += center_diff ? 2 * n : n
    end
    jac_uptodate = true; jac_force_update = false
    KAOps.AtA!(W.jtj, W.fjac, W.fjac)
    if KAOps.nanflag(W.fjac)
        converged = -11; maxiter = 0
    end
    KAOps.fill!(W.acc, zero(T)); KAOps.fill!(W.a, zero(T))

    # damping matrix
    if dtd === nothing
        KAOps.fill!(W.dtd, zero(T))
    else
        KAOps.copyto!(W.dtd, dtd)
    end
    if damp_mode == 0
        KAOps.fill!(W.dtd, zero(T))
        KAOps.fill_diag!(W.dtd, T(1))
    elseif damp_mode == 1
        KAOps.maxdiag!(W.dtd, W.jtj)
    end

    # lambda / delta initialization
    if imethod < 10
        diagjtj = diag(Array(W.jtj))
        lam = T(diagjtj[1]); for i in 2:n; lam = max(lam, T(diagjtj[i])); end
        lam = T(initialfactor) * lam
    elseif converged == 0
        KAOps.mul!(W.tmp1, W.dtd, W.x)
        KAOps.dot!(W.scalar, W.x, W.tmp1)
        delta = T(initialfactor) * sqrt(_hscalar(W.scalar))
        lam = T(1)
        if delta == 0
            delta = T(100)
        end
        lam = gpu_trust_region!(W, W.fvec, W.fjac, W.dtd, delta)   # -> lamba
    end

    # ---- main loop ----
    for istep in 1:maxiter
        niters = istep
        info = 0
        if callback !== nothing
            ret = callback(W.x, W.v, W.a, W.fvec, W.fjac, W.acc, lam, W.dtd,
                           W.fvec_new, accepted, info)
            ret !== nothing && (info = ret)
        end
        if info != 0
            converged = -10; break
        end

        if accepted > 0 && ibroyden <= 0
            jac_force_update = true
        end
        if accepted + ibroyden <= 0 && !jac_uptodate
            jac_force_update = true
        end
        if accepted > 0 && ibroyden > 0 && !jac_force_update
            gpu_update_jac!(W)
            jac_uptodate = false
        end

        if accepted > 0
            KAOps.copyto!(W.fvec, W.fvec_new)
            KAOps.copyto!(W.x, W.x_new)
            KAOps.copyto!(W.vold, W.v)
            C = Cnew
            if C <= Cbest
                KAOps.copyto!(W.x_best, W.x)
                Cbest = C
                KAOps.copyto!(W.fvec_best, W.fvec)
            end
        end

        if jac_force_update
            if analytic_jac && obj.grad! !== nothing
                obj.grad!(be, W.x, W.fjac, obj.data); njev += 1
            else
                jac!(W, obj, center_diff, h1)
                nfev += center_diff ? 2 * n : n
            end
            jac_uptodate = true
            jac_force_update = false
        end

        if !KAOps.nanflag(W.fjac)
            KAOps.AtA!(W.jtj, W.fjac, W.fjac)

            if istep > 1
                if damp_mode == 1
                    KAOps.maxdiag!(W.dtd, W.jtj)
                end
                if imethod == 0
                    lam = T(update_lam_factor(lam, accepted, factoraccept, factorreject))
                elseif imethod == 1
                    lam = T(update_lam_nelson(lam, accepted, factoraccept, factorreject,
                                              rho))
                elseif imethod == 2
                    cos_on = gpu_cos_v_vold!(W)
                    (lam, a_param) = _lam_umrigar_scalars(lam, accepted, a_param, C, Cnew,
                                                          cos_on)
                elseif imethod == 10
                    delta = T(update_delta_factor(delta, accepted, factoraccept,
                                                  factorreject))
                    lam = gpu_trust_region!(W, W.fvec, W.fjac, W.dtd, delta)
                elseif imethod == 11
                    pnorm = gpu_dtd_pnorm!(W, W.v)
                    (delta, lam) = _delta_more_scalars(delta, lam, pnorm, rho, C, Cnew,
                                                       actred, dirder, av, avmax)
                    lam = gpu_trust_region!(W, W.fvec, W.fjac, W.dtd, delta)
                end
            end

            r = gpu_step!(W, obj, lam, C, Cbest, accepted, ibold, T(avmax),
                          jac_uptodate, T(h2))
            nfev += r.nev
            accepted = r.accepted
            cos_alpha = r.cos_alpha; pred_red = r.pred_red
            dirder = r.dirder; delta = r.delta; av = r.av
            Cnew = r.Cnew; rho = r.rho; actred = r.actred
        else
            converged = -11
            break
        end

        if converged == 0
            xh = Array(W.x); xnewh = Array(W.x_new)
            (converged, counter) = gpu_convergence_check!(
                W, accepted, counter, C, Cnew, lam, xh, xnewh,
                nfev, maxfev, njev, maxjev, naev, maxaev,
                T(maxlam), T(minlam), T(artol), T(Cgoal), T(gtol), T(xtol),
                T(xrtol), T(ftol), T(frtol), cos_alpha)
            if converged == 1 && !jac_uptodate
                converged = 0
                jac_force_update = true
            end
        end

        if converged != 0
            break
        end
        if accepted >= 0
            jac_uptodate = false
        end
    end

    converged == 0 && (converged = -1)

    KAOps.copyto!(x, W.x_best)
    KAOps.copyto!(fvec, W.fvec_best)
    return (x, fvec, niters, nfev, njev, naev, converged)
end
