# -*- julia -*-
# src/workspace.jl
# Reusable buffer workspace + OncePerTask lazy cache for GeodesicLM.
#
# Calling `geodesiclm` repeatedly (e.g. in an inner loop) currently re-allocates
# all of its internal arrays every call. To avoid that, GeodesicLM can operate
# on a `GLMWorkspace`: a single bundle of pre-allocated buffers sized to a given
# (n, m) problem. A per-task lazy cache (`_get_workspace`) hands out one
# workspace per task so repeated calls reuse the same buffers with no
# allocation, while distinct tasks do not interfere ("OncePerTask" pattern).
#
# All in-place helper routines (fdjac!, fd_avv!, update_jac!, convergence_check!,
# acceptance!, update_lam_umrigar!, update_delta_more!, trust_region!, dgqt!,
# destsv!) write into provided buffers/workspace and perform the same arithmetic
# in the same order as their allocating counterparts.

"""
    GLMWorkspace(n, m)

A reusable buffer workspace for solving an `n`-parameter, `m`-residual problem
with `geodesiclm`. All internal scratch arrays, vectors and matrices are
pre-allocated here so they can be reused across calls without re-allocating.
Construct one via `GLMWorkspace(n, m)`, or let `geodesiclm` fetch a cached,
task-local instance automatically.
"""
mutable struct GLMWorkspace
    n::Int
    m::Int
    # --- main algorithm vectors ---
    v::Vector{Float64}            # n : LM step
    vold::Vector{Float64}         # n : previous step
    a::Vector{Float64}            # n : acceleration step
    x_new::Vector{Float64}        # n : proposed new point
    x_best::Vector{Float64}       # n : best point seen
    acc::Vector{Float64}          # m : second directional derivative
    fvec_new::Vector{Float64}     # m : function values at proposed point
    fvec_best::Vector{Float64}    # m : function values at best point
    jv::Vector{Float64}           # m : fjac * v
    # --- n×n matrices ---
    jtj::Matrix{Float64}          # J'*J
    g::Matrix{Float64}            # jtj + lam*dtd
    dtd::Matrix{Float64}          # damping matrix
    # --- m×n ---
    fjac::Matrix{Float64}         # Jacobian
    # --- general n-vector scratch ---
    tmp1::Vector{Float64}
    tmp2::Vector{Float64}
    tmp3::Vector{Float64}
    # --- fdjac temporaries ---
    x_plus::Vector{Float64}       # n
    x_minus::Vector{Float64}      # n
    fvec_plus::Vector{Float64}    # m
    fvec_minus::Vector{Float64}   # m
    # --- fd_avv temporaries ---
    xtmp::Vector{Float64}         # n
    ftmp::Vector{Float64}         # m
    acc_tmp::Vector{Float64}      # m
    # --- update_jac! temporaries ---
    r1::Vector{Float64}           # m
    fv::Vector{Float64}           # m
    djac::Vector{Float64}         # m
    v2::Vector{Float64}           # n
    # --- convergence_check temporaries ---
    grad::Vector{Float64}         # n
    # --- trust_region temporaries ---
    jtilde::Matrix{Float64}       # m×n
    gt::Matrix{Float64}           # n×n
    gradC::Vector{Float64}        # n
    # --- dgqt temporaries ---
    dgqtA::Matrix{Float64}        # n×n (copy of A, mutated)
    z::Vector{Float64}            # n (destsv eigenvector / neg. curvature dir)
    wa1::Vector{Float64}          # n
    wa2::Vector{Float64}          # n
end

function GLMWorkspace(n::Int, m::Int)
    GLMWorkspace(
        n, m,
        zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
        zeros(m), zeros(m), zeros(m), zeros(m),
        zeros(n, n), zeros(n, n), zeros(n, n),
        zeros(m, n),
        zeros(n), zeros(n), zeros(n),
        zeros(n), zeros(n), zeros(m), zeros(m),
        zeros(n), zeros(m), zeros(m),
        zeros(m), zeros(m), zeros(m), zeros(n),
        zeros(n),
        zeros(m, n), zeros(n, n), zeros(n),
        zeros(n, n), zeros(n), zeros(n), zeros(n),
    )
end

###############################################################################
# OncePerTask lazy cache
###############################################################################

const _WS_KEY = :GeodesicLM_workspace_cache

"""
    _get_workspace(n, m) -> GLMWorkspace

Return a `GLMWorkspace` for an `(n, m)` problem, allocating it once per Julia
task and caching it in that task's local storage so subsequent calls on the
same task reuse the same buffers (no allocation). Different tasks get their own
workspace, so the cache is safe to use from threads/async tasks.
"""
function _get_workspace(n::Int, m::Int)::GLMWorkspace
    tls = task_local_storage()
    cache = if haskey(tls, _WS_KEY)
        tls[_WS_KEY]::Dict{Tuple{Int,Int},GLMWorkspace}
    else
        c = Dict{Tuple{Int,Int},GLMWorkspace}()
        tls[_WS_KEY] = c
        c
    end
    return get!(cache, (n, m)) do
        GLMWorkspace(n, m)
    end
end

###############################################################################
# In-place finite-difference Jacobian (mirrors fdjac)
###############################################################################
function fdjac!(fjac::Matrix{Float64}, ws::GLMWorkspace, m::Int, n::Int,
                x::Vector{Float64}, fvec::Vector{Float64}, func,
                eps::Float64, center_diff::Bool)
    epsmach = dpmpar(1)
    xp = ws.x_plus
    xm = ws.x_minus
    fp = ws.fvec_plus
    fm = ws.fvec_minus
    if center_diff
        for i in 1:n
            h = eps * abs(x[i])
            if h < epsmach
                h = eps
            end
            copyto!(xp, x); xp[i] += 0.5 * h
            copyto!(xm, x); xm[i] -= 0.5 * h
            func(xp, fp)
            func(xm, fm)
            @inbounds for k in 1:m
                fjac[k, i] = (fp[k] - fm[k]) / h
            end
        end
    else
        for i in 1:n
            h = eps * abs(x[i])
            if h < epsmach
                h = eps
            end
            copyto!(xp, x); xp[i] += h
            func(xp, fp)
            @inbounds for k in 1:m
                fjac[k, i] = (fp[k] - fvec[k]) / h
            end
        end
    end
    return fjac
end

###############################################################################
# In-place finite-difference acceleration (mirrors fd_avv)
###############################################################################
function fd_avv!(acc::Vector{Float64}, ws::GLMWorkspace, m::Int, n::Int,
                 x::Vector{Float64}, v::Vector{Float64}, fvec::Vector{Float64},
                 fjac::Matrix{Float64}, func, jac_uptodate::Bool, h2::Float64)
    xtmp = ws.xtmp
    ftmp = ws.ftmp
    atmp = ws.acc_tmp
    if jac_uptodate
        @inbounds for i in 1:n
            xtmp[i] = x[i] + h2 * v[i]
        end
        func(xtmp, ftmp)
        mul!(atmp, fjac, v)
        @inbounds for k in 1:m
            acc[k] = (2.0 / h2) * ((ftmp[k] - fvec[k]) / h2 - atmp[k])
        end
    else
        @inbounds for i in 1:n
            xtmp[i] = x[i] + h2 * v[i]
        end
        func(xtmp, ftmp)
        @inbounds for i in 1:n
            xtmp[i] = x[i] - h2 * v[i]
        end
        func(xtmp, atmp)
        @inbounds for k in 1:m
            acc[k] = (ftmp[k] - 2 * fvec[k] + atmp[k]) / (h2 * h2)
        end
    end
    return acc
end

###############################################################################
# In-place rank-deficient Broyden Jacobian update (mirrors update_jac!)
###############################################################################
function update_jac!(ws::GLMWorkspace, m::Int, n::Int, fjac::Matrix{Float64},
                     fvec::Vector{Float64}, fvec_new::Vector{Float64},
                     acc::Vector{Float64}, v::Vector{Float64}, a::Vector{Float64})
    r1 = ws.r1
    djac = ws.djac
    v2 = ws.v2
    fv = ws.fv

    mul!(fv, fjac, v)
    vv = dot(v, v)
    @inbounds for k in 1:m
        r1[k] = fvec[k] + 0.5 * fv[k] + 0.125 * acc[k]
        djac[k] = 2.0 * (r1[k] - fvec[k] - 0.5 * fv[k]) / vv
    end
    @inbounds for k in 1:m
        for j in 1:n
            fjac[k, j] += djac[k] * 0.5 * v[j]
        end
    end
    @inbounds for j in 1:n
        v2[j] = 0.5 * (v[j] + a[j])
    end
    mul!(fv, fjac, v2)
    dotv2 = dot(v2, v2)
    @inbounds for k in 1:m
        djac[k] = 0.5 * (fvec_new[k] - r1[k] - fv[k]) / dotv2
    end
    @inbounds for k in 1:m
        for j in 1:n
            fjac[k, j] += djac[k] * v2[j]
        end
    end
    return fjac
end

###############################################################################
# In-place smallest singular value estimator (mirrors destsv)
###############################################################################
function destsv!(z::Vector{Float64}, n::Int, R::AbstractMatrix{Float64})
    fill!(z, 0.0)
    const_p01 = 1.0e-2
    const_one = 1.0
    const_zero = 0.0

    e = abs(R[1, 1])
    if e == const_zero
        svmin = const_zero
        z[1] = const_one
        return (svmin, z)
    end

    for i in 1:n
        e = copysign(abs(e), -z[i])
        if abs(e - z[i]) > abs(R[i, i])
            temp = min(const_p01, abs(R[i, i]) / abs(e - z[i]))
            @inbounds for k in 1:n
                z[k] *= temp
            end
            e = temp * e
        end
        if R[i, i] == const_zero
            w = const_one
            wm = const_one
        else
            w = (e - z[i]) / R[i, i]
            wm = -(e + z[i]) / R[i, i]
        end
        s = abs(e - z[i])
        sm = abs(e + z[i])
        @inbounds for j in (i + 1):n
            sm += abs(z[j] + wm * R[i, j])
        end
        if i < n
            @inbounds for j in (i + 1):n
                z[j] += w * R[i, j]
            end
            @inbounds for j in (i + 1):n
                s += abs(z[j])
            end
        end
        if s < sm
            temp = wm - w
            w = wm
            if i < n
                @inbounds for j in (i + 1):n
                    z[j] += temp * R[i, j]
                end
            end
        end
        z[i] = w
    end

    ynorm = norm(z)
    for j in n:-1:1
        if abs(z[j]) > abs(R[j, j])
            temp = min(const_p01, abs(R[j, j]) / abs(z[j]))
            @inbounds for i in 1:n
                z[i] *= temp
            end
            ynorm = temp * ynorm
        end
        if R[j, j] == const_zero
            z[j] = const_one
        else
            z[j] = z[j] / R[j, j]
        end
        temp = -z[j]
        @inbounds for i in 1:(j - 1)
            z[i] += temp * R[i, j]
        end
    end
    znorm = 1.0 / norm(z)
    svmin = ynorm * znorm
    @inbounds for i in 1:n
        z[i] *= znorm
    end
    return (svmin, z)
end

###############################################################################
# In-place trust-region / dgqt
###############################################################################
function dgqt!(x::Vector{Float64}, ws::GLMWorkspace, n::Int, A_input::Matrix{Float64},
               b::Vector{Float64}, delta::Float64, rtol::Float64,
               atol::Float64, itmax::Int, par::Float64)
    A = ws.dgqtA
    z = ws.z
    wa1 = ws.wa1
    wa2 = ws.wa2
    copyto!(A, A_input)
    fill!(x, 0.0)

    const_p001 = 1.0e-3
    const_p5 = 0.5
    const_zero = 0.0
    const_one = 1.0

    parf = const_zero
    xnorm = const_zero
    rxnorm = const_zero
    rednc = false
    info = 0

    @inbounds for j in 1:n
        wa1[j] = A[j, j]
    end
    for j in 1:(n - 1)
        @inbounds for i in (j + 1):n
            A[i, j] = A[j, i]
        end
    end
    anorm = const_zero
    for j in 1:n
        s = 0.0
        @inbounds for i in 1:n
            s += abs(A[i, j])
        end
        wa2[j] = s
        anorm = max(anorm, s)
    end
    @inbounds for j in 1:n
        wa2[j] -= abs(wa1[j])
    end
    bnorm = norm(b)

    pars = -anorm
    parl = -anorm
    paru = -anorm
    @inbounds for j in 1:n
        pars = max(pars, -wa1[j])
        parl = max(parl, wa1[j] + wa2[j])
        paru = max(paru, -wa1[j] + wa2[j])
    end
    parl = max(const_zero, bnorm / delta - parl, pars)
    paru = max(const_zero, bnorm / delta + paru)
    par = max(par, parl)
    par = min(par, paru)
    paru = max(paru, (1.0 + rtol) * parl)

    f = const_zero
    for iter in 1:itmax
        if par <= pars && paru > const_zero
            par = max(const_p001, sqrt(parl / paru)) * paru
        end
        for j in 1:(n - 1)
            @inbounds for i in (j + 1):n
                A[j, i] = A[i, j]
            end
        end
        @inbounds for j in 1:n
            A[j, j] = wa1[j] + par
        end

        indef = 1
        L = nothing
        try
            L = cholesky(Hermitian(A, :U))
            indef = 0
        catch
            indef = 1
        end

        if indef == 0
            parf = par
            copyto!(wa2, b)
            ldiv!(wa2, transpose(L.U), wa2)
            rxnorm = norm(wa2)
            ldiv!(x, L.U, wa2)
            rmul!(x, -1.0)
            xnorm = norm(x)

            if abs(xnorm - delta) <= rtol * delta ||
               (par == const_zero && xnorm <= (1.0 + rtol) * delta)
                info = 1
            end

            (rznorm, _) = destsv!(z, n, L.U)
            pars = max(pars, par - rznorm^2)

            rednc = false
            if xnorm < delta
                prod = dot(z, x) / delta
                temp = (delta - xnorm) * ((delta + xnorm) / delta)
                alpha = temp / (abs(prod) + sqrt(prod^2 + temp / delta))
                alpha = sign(alpha) * abs(alpha)
                if prod < 0.0
                    alpha = -alpha
                end
                rznorm = abs(alpha) * rznorm
                if (rznorm / delta)^2 + par * (xnorm / delta)^2 <= par
                    rednc = true
                end
                if const_p5 * (rznorm / delta)^2 <=
                   rtol * (1.0 - const_p5 * rtol) * (par + (rxnorm / delta)^2)
                    info = 1
                elseif const_p5 * (par + (rxnorm / delta)^2) <= (atol / delta) / delta && info == 0
                    info = 2
                elseif xnorm == const_zero
                    info = 1
                end
            end

            if xnorm == const_zero
                parc = -par
            else
                copyto!(wa2, x)
                temp = 1.0 / xnorm
                rmul!(wa2, temp)
                ldiv!(wa2, transpose(L.U), wa2)
                temp = norm(wa2)
                parc = (((xnorm - delta) / delta) / temp) / temp
            end

            if xnorm > delta
                parl = max(parl, par)
            end
            if xnorm < delta
                paru = min(paru, par)
            end
        else
            parc = -par * 0.1
            pars = max(pars, par, par + parc)
            paru = max(paru, (1.0 + rtol) * pars)
        end

        parl = max(parl, pars)
        if info == 0
            if iter == itmax
                info = 4
            end
            if paru <= (1.0 + const_p5 * rtol) * pars
                info = 3
            end
            if paru == const_zero
                info = 2
            end
        end

        if info != 0
            par = parf
            f = -const_p5 * (rxnorm^2 + par * xnorm^2)
            if rednc
                f = -const_p5 * ((rxnorm^2 + par * delta^2) - rznorm^2)
                axpy!(alpha, z, x)
            end
            break
        end

        par = max(parl, par + parc)
    end

    if info == 0
        info = 4
    end
    return (par, info, f)
end

function trust_region!(v::Vector{Float64}, ws::GLMWorkspace, n::Int, m::Int,
                       fvec::Vector{Float64}, fjac::Matrix{Float64},
                       dtd::Matrix{Float64}, delta::Float64)
    rtol = 1.0e-3
    atol = 1.0e-3
    itmax = 10
    lam = 1.0

    jtilde = ws.jtilde
    gradC = ws.gradC
    g = ws.gt
    for i in 1:n
        s = sqrt(dtd[i, i])
        @inbounds for k in 1:m
            jtilde[k, i] = fjac[k, i] / s
        end
    end
    mul!(gradC, transpose(jtilde), fvec)
    mul!(g, transpose(jtilde), jtilde)
    (lam, info, f) = dgqt!(v, ws, n, g, gradC, delta, rtol, atol, itmax, lam)
    return lam
end

###############################################################################
# In-place acceptance criterion (mirrors acceptance)
###############################################################################
function acceptance!(ws::GLMWorkspace, n::Int, C::Float64, Cnew::Float64, Cbest::Float64,
                     ibold::Int, dtd::Matrix{Float64}, v::Vector{Float64},
                     vold::Vector{Float64})
    accepted = 0
    if Cnew <= C
        accepted = max(accepted + 1, 1)
    else
        if dot(vold, vold) == 0.0
            beta = 1.0
        else
            mul!(ws.tmp3, dtd, vold)
            mul!(ws.tmp1, dtd, v)
            beta = dot(v, ws.tmp3)
            beta = beta / sqrt(dot(v, ws.tmp1) * dot(vold, ws.tmp3))
            beta = min(1.0, 1.0 - beta)
        end
        if ibold == 0
            if Cnew <= C
                accepted = max(accepted + 1, 1)
            else
                accepted = min(accepted - 1, -1)
            end
        elseif ibold == 1
            if beta * Cnew <= Cbest
                accepted = max(accepted + 1, 1)
            else
                accepted = min(accepted - 1, -1)
            end
        elseif ibold == 2
            if beta * beta * Cnew <= Cbest
                accepted = max(accepted + 1, 1)
            else
                accepted = min(accepted - 1, -1)
            end
        elseif ibold == 3
            if beta * Cnew <= C
                accepted = max(accepted + 1, 1)
            else
                accepted = min(accepted - 1, -1)
            end
        elseif ibold == 4
            if beta * beta * Cnew <= C
                accepted = max(accepted + 1, 1)
            else
                accepted = min(accepted - 1, -1)
            end
        end
    end
    return accepted
end

###############################################################################
# In-place lambda/delta updates that need dtd*v products (mirror their allocating counterparts)
###############################################################################
function update_lam_umrigar!(ws::GLMWorkspace, m::Int, n::Int, lam::Float64,
                             accepted::Int, v::Vector{Float64}, vold::Vector{Float64},
                             fvec::Vector{Float64}, fjac::Matrix{Float64},
                             dtd::Matrix{Float64}, a_param::Float64, C::Float64,
                             Cnew::Float64)
    amemory = exp(-1.0 / 5.0)
    mul!(ws.tmp3, dtd, vold)
    mul!(ws.tmp1, dtd, v)
    cos_on = dot(v, ws.tmp3)
    cos_on = cos_on / sqrt(dot(v, ws.tmp1) * dot(vold, ws.tmp3))

    if accepted >= 0
        if Cnew <= C
            if cos_on > 0.0
                a_param = amemory * a_param + (1.0 - amemory)
            else
                a_param = amemory * a_param + 0.5 * (1.0 - amemory)
            end
        else
            a_param = amemory * a_param + 0.5 * (1.0 - amemory)
        end
        factor = min(100.0, max(1.1, 1.0 / (2.2e-16 + 1.0 - abs(2.0 * a_param - 1.0))^2))
        if Cnew <= C && cos_on >= 0.0
            lam = lam / factor
        elseif Cnew > C
            lam = lam * sqrt(factor)
        end
    else
        a_param = amemory * a_param
        factor = min(100.0, max(1.1, 1.0 / (2.2e-16 + 1.0 - abs(2.0 * a_param - 1.0))^2))
        if cos_on > 0.0
            lam = lam * sqrt(factor)
        else
            lam = lam * factor
        end
    end
    return (lam, a_param)
end

function update_delta_more!(ws::GLMWorkspace, delta::Float64, lam::Float64, n::Int,
                            v::Vector{Float64}, dtd::Matrix{Float64}, rho::Float64,
                            C::Float64, Cnew::Float64, dirder::Float64,
                            actred::Float64, av::Float64, avmax::Float64)
    mul!(ws.tmp1, dtd, v)
    pnorm = sqrt(dot(v, ws.tmp1))

    if rho > 0.25
        if lam > 0.0 && rho < 0.75
            temp = 1.0
        else
            temp = 2.0 * pnorm / delta
        end
    else
        if actred >= 0.0
            temp = 0.5
        else
            temp = 0.5 * dirder / (dirder + 0.5 * actred)
        end
        if 0.01 * Cnew >= C || temp < 0.1
            temp = 0.1
        end
    end
    if av > avmax
        temp = min(temp, max(avmax / av, 0.1))
    end
    delta = temp * min(delta, 10.0 * pnorm)
    lam = lam / temp
    return (delta, lam)
end

###############################################################################
# In-place convergence check (mirrors convergence_check)
###############################################################################
function convergence_check!(ws::GLMWorkspace, m::Int, n::Int, accepted::Int, counter::Int,
                            C::Float64, Cnew::Float64, x::Vector{Float64},
                            fvec::Vector{Float64}, fjac::Matrix{Float64}, lam::Float64,
                            xnew::Vector{Float64}, nfev::Int, maxfev::Int, njev::Int,
                            maxjev::Int, naev::Int, maxaev::Int, maxlam::Float64,
                            minlam::Float64, artol::Float64, Cgoal::Float64,
                            gtol::Float64, xtol::Float64, xrtol::Float64, ftol::Float64,
                            frtol::Float64, cos_alpha::Float64)
    converged = 0
    if maxfev > 0
        if nfev >= maxfev
            converged = -2
            counter = 0
            return (converged, counter)
        end
    end
    if maxjev > 0
        if njev >= maxjev
            converged = -3
            return (converged, counter)
        end
    end
    if maxaev > 0
        if naev >= maxaev
            converged = -4
            return (converged, counter)
        end
    end
    if maxlam > 0.0
        if lam >= maxlam
            converged = -5
            return (converged, counter)
        end
    end
    if minlam > 0.0 && lam > 0.0
        if lam <= minlam
            counter = counter + 1
            if counter >= 3
                converged = -6
                return (converged, counter)
            end
            return (converged, counter)
        end
    end
    if artol > 0.0
        if cos_alpha <= artol
            converged = 1
            return (converged, counter)
        end
    end

    mul!(ws.grad, transpose(fjac), fvec)
    rmul!(ws.grad, -1.0)
    if sqrt(dot(ws.grad, ws.grad)) <= gtol
        converged = 3
        return (converged, counter)
    end
    if C < Cgoal
        converged = 2
        return (converged, counter)
    end
    if accepted < 0
        counter = 0
        converged = 0
        return (converged, counter)
    end

    copyto!(ws.tmp1, x)
    axpy!(-1.0, xnew, ws.tmp1)
    if sqrt(dot(ws.tmp1, ws.tmp1)) < xtol
        converged = 4
        return (converged, counter)
    end

    converged = 5
    for i in 1:n
        if abs(x[i] - xnew[i]) > xrtol * abs(x[i]) || (xnew[i] != xnew[i])
            converged = 0
            break
        end
    end
    if converged == 5
        return (converged, counter)
    end

    if (C - Cnew) <= ftol && (C - Cnew) >= 0.0
        counter = counter + 1
        if counter >= 3
            converged = 6
            return (converged, counter)
        end
        return (converged, counter)
    end
    if (C - Cnew) <= (frtol * C) && (C - Cnew) >= 0.0
        counter = counter + 1
        if counter >= 3
            converged = 7
            return (converged, counter)
        end
        return (converged, counter)
    end
    counter = 0
    converged = 0
    return (converged, counter)
end
