# -*- julia -*-
# file geodesiclm.jl
# Main Geodesic-Bold-BroydenUpdate-Levenberg-Marquardt routine
# version 1.0.2

using LinearAlgebra

"""
    default_fd_step(::Type{T}, center_diff::Bool = false) where {T<:AbstractFloat}

Default finite-difference step `h1` for the Jacobian.

The reference implementation's interface uses `1.49012e-8` with *forward*
differences, which is `sqrt(eps(Float64))` -- the usual optimum when the
truncation error is O(h) and the rounding error O(eps/h).  Returning
`sqrt(eps(T))` reproduces that value exactly in double precision and stays
correct in others: a fixed double-precision step is far below `eps(Float32)`,
where every difference would be rounding noise.

With central differences the truncation error is O(h^2) instead, so the
optimum moves to `cbrt(eps(T))`; pass `center_diff` to get that step.
"""
default_fd_step(::Type{T}, center_diff::Bool = false) where {T<:AbstractFloat} =
    center_diff ? cbrt(eps(T)) : sqrt(eps(T))

"""
    default_avv_step(::Type{T}) where {T<:AbstractFloat}

Default step `h2` for the finite-difference second directional derivative.

Unlike `h1` this is not a differencing epsilon in the usual sense: `fd_avv`
probes at `x + h2*v`, so `h2` is a *fraction of the proposed step* and is
dimensionless.  The reference implementation's interface uses `0.1`, and that
value is precision-independent.  (The value matters: a step sized like `h1`
makes the second difference, which is divided by `h2^2`, almost pure rounding
noise, and the resulting spurious acceleration gets every step rejected.)
"""
default_avv_step(::Type{T}) where {T<:AbstractFloat} = T(0.1)

"""
    default_tolerance(::Type{T}) where {T<:AbstractFloat}

Default value of the `Cgoal`, `gtol`, `xtol` and `ftol` convergence
tolerances.  The reference implementation's interface uses `1.49012e-8` for
all four, which is `sqrt(eps(Float64))`; scaling with the working precision
reproduces that exactly in double precision and keeps the tolerances
meaningful in others.
"""
default_tolerance(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))

"""
    default_initialfactor(imethod::Int)

Default value of `initialfactor`, whose meaning depends on the update method.

For the direct-lambda methods (`imethod < 10`) it multiplies the largest
diagonal entry of `J'J` to give the initial Levenberg--Marquardt parameter; for
the trust-region methods it scales the initial trust-region radius.  The two
need very different magnitudes -- starting `lambda` a hundred times *above*
`max(diag(J'J))` makes the first steps vanishingly small and wastes iterations
bringing it back down -- so, as in the reference implementation's interface,
the default is `0.001` for the direct methods and `100.0` for the trust-region
methods.
"""
default_initialfactor(imethod::Int) = imethod < 10 ? 0.001 : 100.0

# Convergence status strings (module-level: building this per call would
# allocate a dictionary on every optimization).
const CONVERGED_INFO = Dict{Int,String}(
    1 => "artol reached",
    2 => "Cgoal reached",
    3 => "gtol reached",
    4 => "xtol reached",
    5 => "xrtol reached",
    6 => "ftol reached",
    7 => "frtol reached",
    -1 => "maxiters exceeded",
    -2 => "maxfev exceeded",
    -3 => "maxjev exceeded",
    -4 => "maxaev exceeded",
    -10 => "User Termination",
    -11 => "NaN Produced",
)

"""
    geodesiclm(func::Function, jacobian::Union{Function, Nothing}, Avv::Union{Function, Nothing};
               x::AbstractVector{T}, fvec::AbstractVector{T}, n::Int, m::Int,
               workspace::Union{GeodesicLMWorkspace{T},Nothing}=nothing,
               callback::Union{Function, Nothing}=nothing, info::Int=0,
               analytic_jac::Bool=false, analytic_Avv::Bool=false,
               center_diff::Bool=false,
               h1::Union{Nothing,Real}=nothing, h2::Union{Nothing,Real}=nothing,
               dtd::Union{AbstractMatrix,Nothing}=nothing, damp_mode::Int=1,
               maxiter::Int=200*(n+1), maxfev::Int=0, maxjev::Int=0, maxaev::Int=0,
               maxlam::Real=-1.0, minlam::Real=-1.0,
               artol::Real=0.001,
               Cgoal::Union{Nothing,Real}=nothing, gtol::Union{Nothing,Real}=nothing,
               xtol::Union{Nothing,Real}=nothing, xrtol::Real=-1.0,
               ftol::Union{Nothing,Real}=nothing, frtol::Real=-1.0,
               converged::Int=0,
               print_level::Int=0, print_unit::IO=stdout,
               imethod::Int=0, iaccel::Int=1, ibold::Int=2, ibroyden::Int=0,
               initialfactor::Union{Nothing,Real}=nothing, factoraccept::Real=3.0,
               factorreject::Real=2.0, avmax::Real=0.75, incremental_jtj::Bool=false)

Minimize the sum of squares of m nonlinear functions of n variables using the
Geodesic-Levenberg-Marquardt algorithm with geodesic acceleration, bold acceptance
criterion, and Broyden Jacobian updates.

The keyword defaults follow the reference implementation's own interface
(`original/pythonInterface/geodesiclm.py`).  Where the reference hard-codes a
double-precision constant -- the finite-difference step and the `Cgoal`,
`gtol`, `xtol`, `ftol` tolerances, all `sqrt(eps(Float64))` -- the default here
scales with the working precision, so it reproduces the reference value
exactly in `Float64` and stays meaningful in lower precision.

 The purpose of geodesiclm is to minimize the sum of the squares of m nonlinear
 functions of n variables by a modification of the Levenberg-Marquardt algorithm
 that utilizes the geodesic acceleration step correction, bold acceptance criterion,
 and a Broyden update of the jacobian matrix. The method employs one of several
 possible schemes for updating the Levenberg-Marquardt parameter.
 If you use this code, please acknowledge such by referencing one of the
 following papers in any published work:

 Transtrum M.K., Machta B.B., and Sethna J.P, Why are nonlinear fits to data
 so challenging? Phys. Rev. Lett. 104, 060201 (2010)

 Transtrum M.K., Machta B.B., and Sethna J.P., The geometry of nonlinear least
 squares with applications to sloppy model and optimization. Phys. Rev. E. 80, 036701 (2011)
 # Arguments
 - `func`: User-supplied function computing residuals: func(x, fvec) modifies fvec in place
 - `jacobian`: User-supplied Jacobian function: jacobian(x, fjac) or nothing
 - `Avv`: User-supplied second derivative function: Avv(x, v, acc) or nothing
 - `x`: Initial estimate of the solution (modified in place with final solution)
 - `fvec`: Output array for function values at final solution
 - `n`: Number of parameters
 - `m`: Number of functions
 - `workspace`: Reusable [`GeodesicLMWorkspace`](@ref).  Passing one across calls
   makes the optimization allocation-free once the workspace has grown to the
   largest problem seen; a fresh workspace is allocated when omitted.
 - `callback`: User-supplied callback function called after each iteration.
   The callback receives `(x, v, a, fvec, fjac, acc, lam, dtd, fvec_new, accepted, info)`
   and may return a new (nonzero) `info` value to request early termination
   (returning `nothing` leaves `info` unchanged).  The arrays it receives are
   views into the workspace and are only valid for the duration of the call.
 - `info`: User-provided control flag (set to non-zero to terminate)
 - `analytic_jac`: Whether to use analytical Jacobian
 - `analytic_Avv`: Whether to use analytical second derivatives
 - `center_diff`: Use central differences (true) or forward differences (false)
 - `h1`: Step size for Jacobian finite differences (default: [`default_fd_step`](@ref))
 - `h2`: Step size for second derivative finite differences, as a fraction of
   the proposed step (default: [`default_avv_step`](@ref))
 - `dtd`: Damping matrix (diagonal or full); copied into the workspace, the
   caller's matrix is not modified
 - `damp_mode`: Damping mode (0=identity, 1=dynamic diagonal)
 - `maxiter`: Maximum number of iterations
 - `maxfev`: Maximum function evaluations (0=unlimited)
 - `maxjev`: Maximum Jacobian evaluations (0=unlimited)
 - `maxaev`: Maximum second derivative evaluations (0=unlimited)
 - `maxlam`: Maximum Levenberg-Marquardt parameter
 - `minlam`: Minimum Levenberg-Marquardt parameter
 - `artol`: Angle convergence tolerance
 - `Cgoal`: Target cost value
 - `gtol`: Gradient convergence tolerance
 - `xtol`: Step size convergence tolerance
 - `xrtol`: Relative parameter change convergence tolerance (negative disables it)
 - `ftol`: Cost stagnation tolerance (absolute)
 - `frtol`: Cost stagnation tolerance (relative; negative disables it)
 - `print_level`: Verbosity level (0-5)
 - `print_unit`: Output stream for printing
 - `imethod`: Lambda update method (0-2 direct, 10-11 trust region)
 - `iaccel`: Include geodesic acceleration (0=no, 1=yes)
 - `ibold`: Bold acceptance criterion type (0-4)
 - `ibroyden`: Use Broyden updates (positive=yes)
 - `initialfactor`: Initial lambda (`imethod < 10`) or delta value
   (default: [`default_initialfactor`](@ref))
 - `factoraccept`: Factor for lambda/delta on acceptance
 - `factorreject`: Factor for lambda/delta on rejection
 - `avmax`: Maximum allowed acceleration norm
 - `incremental_jtj`: Update `J'J` with each Broyden update in O(m n) instead of
   forming it anew in O(m n²); it is formed anew after every evaluated
   Jacobian.  Changes the results by rounding.  `J'J` is formed only when the
   Jacobian has changed, also without this option.
"""
function geodesiclm(func::Function, jacobian::Union{Function,Nothing}, Avv::Union{Function,Nothing};
                    x::AbstractVector{T}, fvec::AbstractVector{T}, n::Int, m::Int,
                    workspace::Union{GeodesicLMWorkspace{T},Nothing}=nothing,
                    callback::Union{Function,Nothing}=nothing, info::Int=0,
                    analytic_jac::Bool=false, analytic_Avv::Bool=false,
                    center_diff::Bool=false,
                    h1::Union{Nothing,Real}=nothing, h2::Union{Nothing,Real}=nothing,
                    dtd::Union{AbstractMatrix,Nothing}=nothing, damp_mode::Int=1,
                    maxiter::Int=200 * (n + 1), maxfev::Int=0, maxjev::Int=0, maxaev::Int=0,
                    maxlam::Real=-1.0, minlam::Real=-1.0,
                    artol::Real=0.001,
                    Cgoal::Union{Nothing,Real}=nothing, gtol::Union{Nothing,Real}=nothing,
                    xtol::Union{Nothing,Real}=nothing, xrtol::Real=-1.0,
                    ftol::Union{Nothing,Real}=nothing, frtol::Real=-1.0,
                    converged::Int=0,
                    print_level::Int=0, print_unit::IO=stdout,
                    imethod::Int=0, iaccel::Int=1, ibold::Int=2, ibroyden::Int=0,
                    initialfactor::Union{Nothing,Real}=nothing, factoraccept::Real=3.0,
                    factorreject::Real=2.0, avmax::Real=0.75,
                    incremental_jtj::Bool=false) where {T<:AbstractFloat}

    # Scalar options at the working precision.  The finite-difference steps and
    # the absolute tolerances default to precision-dependent values that
    # reproduce the reference interface exactly in double precision (see
    # `default_fd_step`, `default_avv_step` and `default_tolerance`).
    h1 = T(h1 === nothing ? default_fd_step(T, center_diff) : h1)
    h2 = T(h2 === nothing ? default_avv_step(T) : h2)
    maxlam = T(maxlam); minlam = T(minlam)
    artol = T(artol)
    Cgoal = T(Cgoal === nothing ? default_tolerance(T) : Cgoal)
    gtol = T(gtol === nothing ? default_tolerance(T) : gtol)
    xtol = T(xtol === nothing ? default_tolerance(T) : xtol)
    ftol = T(ftol === nothing ? default_tolerance(T) : ftol)
    xrtol = T(xrtol); frtol = T(frtol)
    initialfactor = T(initialfactor === nothing ? default_initialfactor(imethod) : initialfactor)
    factoraccept = T(factoraccept); factorreject = T(factorreject); avmax = T(avmax)

    # Keep references to the caller's arrays so we can write results back
    x_in = x
    fvec_in = fvec

    ws = workspace === nothing ? GeodesicLMWorkspace{T}(m, n) : _ensure_capacity!(workspace, m, n)

    # All work happens in workspace views of exactly the requested size
    acc       = view(ws.acc, 1:m)
    fv        = view(ws.fvec, 1:m)
    fvec_new  = view(ws.fvec_new, 1:m)
    fvec_best = view(ws.fvec_best, 1:m)
    jv        = view(ws.jv, 1:m)
    ftmp      = view(ws.ftmp, 1:m)
    mtmp1     = view(ws.mtmp1, 1:m)
    mtmp2     = view(ws.mtmp2, 1:m)
    xc        = view(ws.x, 1:n)
    v         = view(ws.v, 1:n)
    vold      = view(ws.vold, 1:n)
    a         = view(ws.a, 1:n)
    x_new     = view(ws.x_new, 1:n)
    x_best    = view(ws.x_best, 1:n)
    ntmp1     = view(ws.ntmp1, 1:n)
    ntmp2     = view(ws.ntmp2, 1:n)
    ntmp3     = view(ws.ntmp3, 1:n)
    fjac      = view(ws.fjac, 1:m, 1:n)
    jtj       = view(ws.jtj, 1:n, 1:n)
    g         = view(ws.g, 1:n, 1:n)
    dtdm      = view(ws.dtd, 1:n, 1:n)

    copyto!(xc, x)
    copyto!(fv, fvec)

    # Initialize internal parameters
    fill!(acc, zero(T))
    fill!(v, zero(T))
    fill!(vold, zero(T))
    fill!(a, zero(T))
    lam = zero(T)
    delta = zero(T)
    cos_alpha = one(T)
    av = zero(T)

    temp1 = zero(T)
    temp2 = zero(T)
    pred_red = zero(T)
    dirder = zero(T)
    actred = zero(T)
    rho = zero(T)
    a_param = T(0.5)
    Cnew = zero(T)

    if print_level >= 1
        println(print_unit, "Optimizing with Geodesic-Levenberg-Marquardt algorithm, version 1.0.2")
        println(print_unit, "Method Details:")
        println(print_unit, "  Update method:   ", imethod)
        println(print_unit, "  acceleration:    ", iaccel)
        println(print_unit, "  Bold method:     ", ibold)
        println(print_unit, "  Broyden updates: ", ibroyden)
        flush(print_unit)
    end

    # Initialize variables
    niters = 0
    nfev = 0
    naev = 0
    njev = 0
    converged = 0

    accepted = 0
    counter = 0

    # Evaluate function at initial point
    func(xc, fv)
    nfev = nfev + 1
    C = T(0.5) * _dot(fv, fv)

    if print_level >= 1
        println(print_unit, "  Initial Cost:    ", C)
        flush(print_unit)
    end

    # Check for NaNs in initial fvec
    if _hasnan(fv)
        converged = -11
        maxiter = 0
    end

    Cbest = C
    copyto!(fvec_best, fv)
    copyto!(x_best, xc)

    # Compute initial Jacobian
    if analytic_jac && jacobian !== nothing
        jacobian(xc, fjac)
        njev = njev + 1
    else
        fdjac!(fjac, m, n, xc, fv, func, h1, center_diff, ntmp1, mtmp1, mtmp2)
        if center_diff
            nfev = nfev + 2 * n
        else
            nfev = nfev + n
        end
    end

    jac_uptodate = true
    jac_force_update = false
    _syrk_t!(jtj, fjac)
    # Has fjac changed since the last NaN check, and since jtj was formed?
    # A rejected step leaves both as they are.
    fjac_changed = false
    jtj_stale = false

    # Check fjac for NaNs
    if _hasnan(fjac)
        converged = -11
        maxiter = 0
    end

    fill!(acc, zero(T))
    fill!(a, zero(T))

    # Initialize damping matrix
    if dtd === nothing
        fill!(dtdm, zero(T))
    else
        copyto!(dtdm, dtd)
    end

    if damp_mode == 0
        # Identity matrix
        fill!(dtdm, zero(T))
        for i in 1:n
            dtdm[i, i] = one(T)
        end
    elseif damp_mode == 1
        # Diagonal scaling
        for i in 1:n
            dtdm[i, i] = max(jtj[i, i], dtdm[i, i])
        end
    end

    # Initialize lambda or delta
    if imethod < 10
        # Initialize lambda
        lam = jtj[1, 1]
        for i in 2:n
            lam = max(jtj[i, i], lam)
        end
        lam = lam * initialfactor
    else
        # Initialize trust region radius
        _gemv!(ntmp1, dtdm, xc)
        delta = initialfactor * sqrt(_dot(xc, ntmp1))
        lam = one(T)
        if delta == zero(T)
            delta = T(100)
        end
        if converged == 0
            (vtr, lam) = trust_region(n, m, fv, fjac, dtdm, delta)
            copyto!(v, vtr)
        end
    end

    # Main optimization loop
    for istep in 1:maxiter
        niters = istep

        info = 0
        if callback !== nothing
            ret = callback(xc, v, a, fv, fjac, acc, lam, dtdm, fvec_new, accepted, info)
            if ret !== nothing
                info = ret
            end
        end

        if info != 0
            converged = -10
            break
        end

        # Update Functions
        # Full or partial Jacobian Update?
        if accepted > 0 && ibroyden <= 0
            jac_force_update = true
        end
        if accepted + ibroyden <= 0 && !jac_uptodate
            jac_force_update = true  # Force jac update after too many failed attempts
        end

        if accepted > 0 && ibroyden > 0 && !jac_force_update
            # Rank deficient update of Jacobian matrix
            if incremental_jtj
                # ntmp2 is free here: it is written before its next use
                update_jac!(m, n, fjac, fv, fvec_new, acc, v, a, mtmp1, mtmp2, ntmp1, jtj, ntmp2)
            else
                update_jac!(m, n, fjac, fv, fvec_new, acc, v, a, mtmp1, mtmp2, ntmp1)
                jtj_stale = true
            end
            fjac_changed = true
            jac_uptodate = false
        end

        if accepted > 0
            # Accepted step
            copyto!(fv, fvec_new)
            copyto!(xc, x_new)
            copyto!(vold, v)
            C = Cnew
            if C <= Cbest
                copyto!(x_best, xc)
                Cbest = C
                copyto!(fvec_best, fv)
            end
        end

        if jac_force_update
            # Full rank update of Jacobian
            if analytic_jac && jacobian !== nothing
                jacobian(xc, fjac)
                njev = njev + 1
            else
                fdjac!(fjac, m, n, xc, fv, func, h1, center_diff, ntmp1, mtmp1, mtmp2)
                if center_diff
                    nfev = nfev + 2 * n
                else
                    nfev = nfev + n
                end
            end
            jac_uptodate = true
            jac_force_update = false
            fjac_changed = true
            jtj_stale = true
        end

        if fjac_changed
            # Check fjac for NaNs
            if _hasnan(fjac)
                # If NaNs in Jacobian
                converged = -11
                break
            end
            fjac_changed = false
        end

        if jtj_stale
            _syrk_t!(jtj, fjac)
            jtj_stale = false
        end

        # Update Scaling/lam/TrustRegion
        if istep > 1
            if damp_mode == 1
                # Update diagonal scaling
                for i in 1:n
                    dtdm[i, i] = max(jtj[i, i], dtdm[i, i])
                end
            end

            # Update lambda or delta
            if imethod == 0
                # Update lam directly by fixed factors
                lam = update_lam_factor(lam, accepted, factoraccept, factorreject)
            elseif imethod == 1
                # Update lam based on Gain Factor rho (Nelson method)
                lam = update_lam_nelson(lam, accepted, factoraccept, factorreject, rho)
            elseif imethod == 2
                # Update lam using Umrigar and Nightingale method
                (lam, a_param) = update_lam_umrigar(m, n, lam, accepted, v, vold, fv, fjac,
                                                    dtdm, a_param, C, Cnew, ntmp1)
            elseif imethod == 10
                # Update delta by fixed factors
                delta = update_delta_factor(delta, accepted, factoraccept, factorreject)
                (vtr, lam) = trust_region(n, m, fv, fjac, dtdm, delta)
                copyto!(v, vtr)
            elseif imethod == 11
                # Update delta as described in Moré reference
                (delta, lam) = update_delta_more(delta, lam, n, v, dtdm, rho, C, Cnew,
                                                 dirder, actred, av, avmax, ntmp1)
                (vtr, lam) = trust_region(n, m, fv, fjac, dtdm, delta)
                copyto!(v, vtr)
            end
        end

        # Propose Step
        # g = jtj + lam*dtd
        @inbounds for j in 1:n, i in 1:n
            g[i, j] = jtj[i, j] + lam * dtdm[i, j]
        end

        # Cholesky decomposition g = U'U in place (upper triangle of g);
        # a matrix that is not numerically positive definite gives info != 0
        chol_info = _potrf_upper!(g)

        # The norm of the gradient J'f for the convergence check.  fv and fjac
        # do not change before the check in this iteration, so the right-hand
        # side of the normal equations serves; computed there if not solved.
        gradnorm = nothing
        if chol_info == 0
            # If matrix decomposition successful, solve the normal equations
            # (J'J + lam*dtd)*v = -J'*f
            _gemv_t!(v, fjac, fv)
            gradnorm = sqrt(_dot(v, v))
            _scal!(v, -one(T))
            _potrs_upper!(v, g)

            # Calculate the predicted reduction and directional derivative
            _gemv!(ntmp2, jtj, v)
            temp1 = T(0.5) * _dot(v, ntmp2) / C
            _gemv!(ntmp3, dtdm, v)
            vdtdv = _dot(v, ntmp3)
            temp2 = T(0.5) * lam * vdtdv / C
            pred_red = temp1 + 2 * temp2
            dirder = -(temp1 + temp2)

            # Calculate cos_alpha -- cos of angle between step direction and residual
            _gemv!(jv, fjac, v)
            cos_alpha = abs(_dot(fv, jv)) / (sqrt(_dot(fv, fv)) * sqrt(_dot(jv, jv)))

            if imethod < 10
                delta = sqrt(vdtdv)
            end

            # Update acceleration
            if iaccel > 0
                if analytic_Avv && Avv !== nothing
                    Avv(xc, v, acc)
                    naev = naev + 1
                else
                    fd_avv!(acc, m, n, xc, v, fv, fjac, func, jac_uptodate, h2, ntmp1, ftmp)
                    if jac_uptodate
                        nfev = nfev + 1
                    else
                        nfev = nfev + 2  # We don't use Jacobian if not up to date
                    end
                end

                # Check acceleration for NaNs
                if !_hasnan(acc)
                    _gemv_t!(a, fjac, acc)
                    _scal!(a, -one(T))
                    _potrs_upper!(a, g)
                else
                    fill!(a, zero(T))  # If NaNs in acc, ignore acceleration term
                end
            end

            # Evaluate at proposed step -- only necessary if av <= avmax
            _gemv!(ntmp2, dtdm, a)
            av = sqrt(_dot(a, ntmp2) / vdtdv)

            if av <= avmax
                @inbounds for i in 1:n
                    x_new[i] = xc[i] + v[i] + T(0.5) * a[i]
                end
                func(x_new, fvec_new)
                nfev = nfev + 1
                Cnew = T(0.5) * _dot(fvec_new, fvec_new)

                # Check for NaNs in fvec_new
                if !_hasnan(fvec_new)
                    # If no NaNs, proceed as normal
                    actred = one(T) - Cnew / C
                    rho = zero(T)
                    if pred_red != zero(T)
                        rho = (one(T) - Cnew / C) / pred_red
                    end

                    # Accept or Reject proposed step
                    accepted = acceptance(n, C, Cnew, Cbest, ibold, dtdm, v, vold, ntmp2)
                else
                    # If NaNs in fvec_new, reject step
                    actred = zero(T)
                    rho = zero(T)
                    accepted = min(accepted - 1, -1)
                end
            else
                # If acceleration too large, reject
                accepted = min(accepted - 1, -1)
            end
        else
            # If matrix factorization fails, reject step
            accepted = min(accepted - 1, -1)
        end

        # Check Convergence
        if converged == 0
            (converged, counter) = convergence_check(m, n, accepted, counter, C, Cnew, xc, fv,
                                                     fjac, lam, x_new, nfev, maxfev, njev, maxjev,
                                                     naev, maxaev, maxlam, minlam, artol, Cgoal,
                                                     gtol, xtol, xrtol, ftol, frtol, cos_alpha,
                                                     ntmp1; gradnorm)

            if converged == 1 && !jac_uptodate
                # If converged by artol with out-of-date Jacobian, update to confirm
                converged = 0
                jac_force_update = true
            end
        end

        # Print status
        if (print_level == 2 && accepted > 0) || print_level == 3
            println(print_unit, "  istep, nfev, njev, naev, accepted: ", istep, " ", nfev, " ", njev, " ", naev, " ", accepted)
            println(print_unit, "  Cost, lam, delta: ", C, " ", lam, " ", delta)
            println(print_unit, "  av, cos_alpha: ", av, " ", cos_alpha)
            flush(print_unit)
        elseif (print_level == 4 && accepted > 0) || print_level == 5
            println(print_unit, "  istep, nfev, njev, naev, accepted: ", istep, " ", nfev, " ", njev, " ", naev, " ", accepted)
            println(print_unit, "  Cost, lam, delta: ", C, " ", lam, " ", delta)
            println(print_unit, "  av, cos_alpha: ", av, " ", cos_alpha)
            println(print_unit, "  x = ", xc)
            println(print_unit, "  v = ", v)
            println(print_unit, "  a = ", a)
            flush(print_unit)
        end

        # If converged -- return
        if converged != 0
            break
        end

        if accepted >= 0
            jac_uptodate = false  # Jacobian is now out of date
        end
    end

    # End main loop

    # If not converged
    if converged == 0
        converged = -1
    end

    # Return best fit found (also write it back into the caller's arrays)
    copyto!(x_in, x_best)
    copyto!(fvec_in, fvec_best)

    if print_level >= 1
        println(print_unit, "Optimization finished")
        println(print_unit, "Results:")
        println(print_unit, "  Converged:    ", get(CONVERGED_INFO, converged, "Unknown"), " (", converged, ")")
        println(print_unit, "  Final Cost:   ", T(0.5) * _dot(fvec_best, fvec_best))
        if m > n
            println(print_unit, "  Cost/DOF:     ", T(0.5) * _dot(fvec_best, fvec_best) / (m - n))
        end
        println(print_unit, "  niters:       ", niters)
        println(print_unit, "  nfev:         ", nfev)
        println(print_unit, "  njev:         ", njev)
        println(print_unit, "  naev:         ", naev)
        flush(print_unit)
    end

    return (x_in, fvec_in, niters, nfev, njev, naev, converged)
end
