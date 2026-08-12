# -*- julia -*-
# file geodesiclm.jl
# Main Geodesic-Bold-BroydenUpdate-Levenberg-Marquardt routine
# version 1.0.2

using LinearAlgebra

# Convergence/termination status strings. Hoisted to a module constant so it is
# allocated once instead of on every `geodesiclm` call.
const CONVERGED_INFO = Dict(
    1 => "artol reached",
    2 => "Cgoal reached",
    3 => "gtol reached",
    4 => "xtol reached",
    5 => "xrtol reached",
    6 => "ftol reached",
    7 => "frtol reached",
    -1 => "maxiter exceeded",
    -2 => "maxfev exceeded",
    -3 => "maxjev exceeded",
    -4 => "maxaev exceeded",
    -5 => "maxlam exceeded",
    -6 => "minlam reached",
    -10 => "User Termination",
    -11 => "NaN Produced",
)

"""
    geodesiclm(func::Function, jacobian::Union{Function, Nothing}, Avv::Union{Function, Nothing};
               x::Vector{Float64}, fvec::Vector{Float64}, n::Int, m::Int,
               callback::Union{Function, Nothing}=nothing, info::Int=0,
               analytic_jac::Bool=false, analytic_Avv::Bool=false,
               center_diff::Bool=true, h1::Float64=1.0e-5, h2::Float64=1.0e-5,
               dtd::Union{Matrix{Float64}, Nothing}=nothing, damp_mode::Int=0,
               maxiter::Int=100, maxfev::Int=0, maxjev::Int=0, maxaev::Int=0,
               maxlam::Float64=-1.0, minlam::Float64=-1.0,
               artol::Float64=0.0, Cgoal::Float64=0.0, gtol::Float64=1.0e-8,
               xtol::Float64=1.0e-8, xrtol::Float64=1.0e-8, ftol::Float64=1.0e-8,
               frtol::Float64=1.0e-8, converged::Int=0,
               print_level::Int=0, print_unit::IO=stdout,
               imethod::Int=0, iaccel::Int=1, ibold::Int=1, ibroyden::Int=1,
               initialfactor::Float64=100.0, factoraccept::Float64=2.0,
               factorreject::Float64=2.0, avmax::Float64=10.0,
               ws::Union{Nothing, GLMWorkspace}=nothing)

Minimize the sum of squares of m nonlinear functions of n variables using the 
Geodesic-Levenberg-Marquardt algorithm with geodesic acceleration, bold acceptance 
criterion, and Broyden Jacobian updates.

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
 - `callback`: User-supplied callback function called after each iteration.
   The callback receives `(x, v, a, fvec, fjac, acc, lam, dtd, fvec_new, accepted, info)`
   and may return a new (nonzero) `info` value to request early termination
   (returning `nothing` leaves `info` unchanged).
 - `info`: User-provided control flag (set to non-zero to terminate)
 - `analytic_jac`: Whether to use analytical Jacobian
 - `analytic_Avv`: Whether to use analytical second derivatives
 - `center_diff`: Use central differences (true) or forward differences (false)
 - `h1`: Step size for Jacobian finite differences
 - `h2`: Step size for second derivative finite differences
 - `dtd`: Damping matrix (diagonal or full)
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
 - `xrtol`: Relative parameter change convergence tolerance
 - `ftol`: Cost stagnation tolerance (absolute)
 - `frtol`: Cost stagnation tolerance (relative)
 - `print_level`: Verbosity level (0-5)
 - `print_unit`: Output stream for printing
 - `imethod`: Lambda update method (0-2 direct, 10-11 trust region)
 - `iaccel`: Include geodesic acceleration (0=no, 1=yes)
 - `ibold`: Bold acceptance criterion type (0-4)
 - `ibroyden`: Use Broyden updates (positive=yes)
 - `initialfactor`: Initial lambda or delta value
 - `factoraccept`: Factor for lambda/delta on acceptance
 - `factorreject`: Factor for lambda/delta on rejection
 - `avmax`: Maximum allowed acceleration norm
 - `ws`: Optional `GLMWorkspace` to reuse buffers across calls for inner-loop
   use. If `nothing` (default), a per-task cached workspace is used (see
   `GLMWorkspace`), so repeated calls on the same task avoid re-allocating.

# Bytecode / allocations

This implementation reuses a pre-allocated `GLMWorkspace` and performs all
solves in place, so repeated calls with the same `(n, m)` allocate nothing
beyond Julia's internal solver scratch (if any). It is therefore suitable for
calling from a tight inner loop.
"""
function geodesiclm(func::Function, jacobian::Union{Function, Nothing}, Avv::Union{Function, Nothing};
                   x::Vector{Float64}, fvec::Vector{Float64}, n::Int, m::Int,
                   callback::Union{Function, Nothing}=nothing, info::Int=0,
                   analytic_jac::Bool=false, analytic_Avv::Bool=false,
                   center_diff::Bool=true, h1::Float64=1.0e-5, h2::Float64=1.0e-5,
                   dtd::Union{Matrix{Float64}, Nothing}=nothing, damp_mode::Int=0,
                   maxiter::Int=100, maxfev::Int=0, maxjev::Int=0, maxaev::Int=0,
                   maxlam::Float64=-1.0, minlam::Float64=-1.0,
                   artol::Float64=0.0, Cgoal::Float64=0.0, gtol::Float64=1.0e-8,
                   xtol::Float64=1.0e-8, xrtol::Float64=1.0e-8, ftol::Float64=1.0e-8,
                   frtol::Float64=1.0e-8, converged::Int=0,
                   print_level::Int=0, print_unit::IO=stdout,
                   imethod::Int=0, iaccel::Int=1, ibold::Int=1, ibroyden::Int=1,
                   initialfactor::Float64=100.0, factoraccept::Float64=2.0,
                   factorreject::Float64=2.0, avmax::Float64=10.0,
                   ws::Union{Nothing, GLMWorkspace}=nothing)

    # Fetch a reusable workspace (OncePerTask lazy cache) unless the caller
    # supplied one explicitly.
    W = ws === nothing ? _get_workspace(n, m) : ws
    (length(W.v) == n && length(W.acc) == m) ||
        error("GLMWorkspace sized for (n=$(W.n), m=$(W.m)) does not match (n=$n, m=$m)")

    # Work on workspace buffers; `x`/`fvec` are the caller's arrays (mutated).
    acc = W.acc
    v = W.v
    vold = W.vold
    a = W.a
    x_new = W.x_new
    x_best = W.x_best
    fvec_new = W.fvec_new
    fvec_best = W.fvec_best
    jv = W.jv
    fjac = W.fjac
    jtj = W.jtj
    g = W.g
    dtdw = W.dtd

    fill!(v, 0.0)
    fill!(vold, 0.0)
    fill!(a, 0.0)
    lam = 0.0
    delta = 0.0
    cos_alpha = 1.0
    av = 0.0
    a_param = 0.5
    temp1 = 0.0
    temp2 = 0.0
    pred_red = 0.0
    dirder = 0.0
    actred = 0.0
    rho = 0.0
    C = 0.0
    Cnew = 0.0

    if print_level >= 1
        println(print_unit, "Optimizing with Geodesic-Levenberg-Marquardt algorithm, version 1.0.2")
        println(print_unit, "Method Details:")
        println(print_unit, "  Update method:   ", imethod)
        println(print_unit, "  acceleration:    ", iaccel)
        println(print_unit, "  Bold method:     ", ibold)
        println(print_unit, "  Broyden updates: ", ibroyden)
        flush(print_unit)
    end

    niters = 0
    nfev = 0
    naev = 0
    njev = 0
    converged = 0
    accepted = 0
    counter = 0

    # Evaluate function at initial point
    func(x, fvec)
    nfev = nfev + 1
    C = 0.5 * dot(fvec, fvec)

    if print_level >= 1
        println(print_unit, "  Initial Cost:    ", C)
        flush(print_unit)
    end

    # Check for NaNs in initial fvec
    if any(isnan, fvec)
        converged = -11
        maxiter = 0
    end

    Cbest = C
    copyto!(fvec_best, fvec)
    copyto!(x_best, x)

    # Compute initial Jacobian
    if analytic_jac && jacobian !== nothing
        jacobian(x, fjac)
        njev = njev + 1
    else
        fdjac!(fjac, W, m, n, x, fvec, func, h1, center_diff)
        if center_diff
            nfev = nfev + 2 * n
        else
            nfev = nfev + n
        end
    end

    jac_uptodate = true
    jac_force_update = false
    mul!(jtj, transpose(fjac), fjac)

    if any(isnan, fjac)
        converged = -11
        maxiter = 0
    end

    fill!(acc, 0.0)
    fill!(a, 0.0)

    # Initialize damping matrix (into workspace, reset each call)
    if dtd === nothing
        fill!(dtdw, 0.0)
    else
        copyto!(dtdw, dtd)
    end
    if damp_mode == 0
        fill!(dtdw, 0.0)
        for i in 1:n
            dtdw[i, i] = 1.0
        end
    elseif damp_mode == 1
        for i in 1:n
            dtdw[i, i] = max(jtj[i, i], dtdw[i, i])
        end
    end

    # Initialize lambda or delta
    if imethod < 10
        lam = jtj[1, 1]
        for i in 2:n
            lam = max(jtj[i, i], lam)
        end
        lam = lam * initialfactor
    else
        mul!(W.tmp1, dtdw, x)
        delta = initialfactor * sqrt(dot(x, W.tmp1))
        lam = 1.0
        if delta == 0.0
            delta = 100.0
        end
        if converged == 0
            lam = trust_region!(v, W, n, m, fvec, fjac, dtdw, delta)
        end
    end

    # Main optimization loop
    for istep in 1:maxiter
        niters = istep

        info = 0
        if callback !== nothing
            ret = callback(x, v, a, fvec, fjac, acc, lam, dtdw, fvec_new, accepted, info)
            if ret !== nothing
                info = ret
            end
        end

        if info != 0
            converged = -10
            break
        end

        # Update Functions
        if accepted > 0 && ibroyden <= 0
            jac_force_update = true
        end
        if accepted + ibroyden <= 0 && !jac_uptodate
            jac_force_update = true  # Force jac update after too many failed attempts
        end

        if accepted > 0 && ibroyden > 0 && !jac_force_update
            update_jac!(W, m, n, fjac, fvec, fvec_new, acc, v, a)
            jac_uptodate = false
        end

        if accepted > 0
            copyto!(fvec, fvec_new)
            copyto!(x, x_new)
            copyto!(vold, v)
            C = Cnew
            if C <= Cbest
                copyto!(x_best, x)
                Cbest = C
                copyto!(fvec_best, fvec)
            end
        end

        if jac_force_update
            if analytic_jac && jacobian !== nothing
                jacobian(x, fjac)
                njev = njev + 1
            else
                fdjac!(fjac, W, m, n, x, fvec, func, h1, center_diff)
                if center_diff
                    nfev = nfev + 2 * n
                else
                    nfev = nfev + n
                end
            end
            jac_uptodate = true
            jac_force_update = false
        end

        if !any(isnan, fjac)
            mul!(jtj, transpose(fjac), fjac)

            if istep > 1
                if damp_mode == 1
                    for i in 1:n
                        dtdw[i, i] = max(jtj[i, i], dtdw[i, i])
                    end
                end

                if imethod == 0
                    lam = update_lam_factor(lam, accepted, factoraccept, factorreject)
                elseif imethod == 1
                    lam = update_lam_nelson(lam, accepted, factoraccept, factorreject, rho)
                elseif imethod == 2
                    (lam, a_param) = update_lam_umrigar!(W, m, n, lam, accepted, v, vold,
                                        fvec, fjac, dtdw, a_param, C, Cnew)
                elseif imethod == 10
                    delta = update_delta_factor(delta, accepted, factoraccept, factorreject)
                    lam = trust_region!(v, W, n, m, fvec, fjac, dtdw, delta)
                elseif imethod == 11
                    (delta, lam) = update_delta_more!(W, delta, lam, n, v, dtdw, rho, C,
                                        Cnew, dirder, actred, av, avmax)
                    lam = trust_region!(v, W, n, m, fvec, fjac, dtdw, delta)
                end
            end

            copyto!(g, jtj)
            axpy!(lam, dtdw, g)

            L = nothing
            info = 1
            try
                # Factor in place (overwrites the upper triangle of `g`, which
                # is rebuilt every iteration, so this is safe).
                L = cholesky!(Hermitian(g, :U))
                info = 0
            catch
                info = 1
            end

            if info == 0
                mul!(v, transpose(fjac), fvec)
                rmul!(v, -1.0)
                ldiv!(v, L, v)

                mul!(W.tmp1, jtj, v)
                mul!(W.tmp2, dtdw, v)
                temp1 = 0.5 * dot(v, W.tmp1) / C
                temp2 = 0.5 * lam * dot(v, W.tmp2) / C
                pred_red = temp1 + 2.0 * temp2
                dirder = -1.0 * (temp1 + temp2)

                mul!(jv, fjac, v)
                cos_alpha = abs(dot(fvec, jv)) / (sqrt(dot(fvec, fvec)) * sqrt(dot(jv, jv)))

                if imethod < 10
                    delta = sqrt(dot(v, W.tmp2))
                end

                if iaccel > 0
                    if analytic_Avv && Avv !== nothing
                        Avv(x, v, acc)
                        naev = naev + 1
                    else
                        fd_avv!(acc, W, m, n, x, v, fvec, fjac, func, jac_uptodate, h2)
                        if jac_uptodate
                            nfev = nfev + 1
                        else
                            nfev = nfev + 2  # We don't use Jacobian if not up to date
                        end
                    end

                    if !any(isnan, acc)
                        mul!(a, transpose(fjac), acc)
                        rmul!(a, -1.0)
                        ldiv!(a, L, a)
                    else
                        fill!(a, 0.0)
                    end
                end

                mul!(W.tmp1, dtdw, a)
                mul!(W.tmp2, dtdw, v)
                av = sqrt(dot(a, W.tmp1) / dot(v, W.tmp2))

                if av <= avmax
                    copyto!(x_new, x)
                    axpy!(1.0, v, x_new)
                    axpy!(0.5, a, x_new)
                    func(x_new, fvec_new)
                    nfev = nfev + 1
                    Cnew = 0.5 * dot(fvec_new, fvec_new)

                    if !any(isnan, fvec_new)
                        actred = 1.0 - Cnew / C
                        rho = 0.0
                        if pred_red != 0.0
                            rho = (1.0 - Cnew / C) / pred_red
                        end
                        accepted = acceptance!(W, n, C, Cnew, Cbest, ibold, dtdw, v, vold)
                    else
                        actred = 0.0
                        rho = 0.0
                        accepted = min(accepted - 1, -1)
                    end
                else
                    accepted = min(accepted - 1, -1)
                end
            else
                accepted = min(accepted - 1, -1)
            end
        else
            converged = -11
            break
        end

        if converged == 0
            (converged, counter) = convergence_check!(W, m, n, accepted, counter, C, Cnew,
                                    x, fvec, fjac, lam, x_new, nfev, maxfev, njev, maxjev,
                                    naev, maxaev, maxlam, minlam, artol, Cgoal, gtol, xtol,
                                    xrtol, ftol, frtol, cos_alpha)
            if converged == 1 && !jac_uptodate
                converged = 0
                jac_force_update = true
            end
        end

        if print_level == 2 && accepted > 0
            println(print_unit, "  istep, nfev, njev, naev, accepted: ", istep, " ", nfev, " ", njev, " ", naev, " ", accepted)
            println(print_unit, "  Cost, lam, delta: ", C, " ", lam, " ", delta)
            println(print_unit, "  av, cos_alpha: ", av, " ", cos_alpha)
            flush(print_unit)
        elseif print_level == 3
            println(print_unit, "  istep, nfev, njev, naev, accepted: ", istep, " ", nfev, " ", njev, " ", naev, " ", accepted)
            println(print_unit, "  Cost, lam, delta: ", C, " ", lam, " ", delta)
            println(print_unit, "  av, cos_alpha: ", av, " ", cos_alpha)
            flush(print_unit)
        elseif print_level == 4 && accepted > 0
            println(print_unit, "  istep, nfev, njev, naev, accepted: ", istep, " ", nfev, " ", njev, " ", naev, " ", accepted)
            println(print_unit, "  Cost, lam, delta: ", C, " ", lam, " ", delta)
            println(print_unit, "  av, cos_alpha: ", av, " ", cos_alpha)
            println(print_unit, "  x = ", x)
            println(print_unit, "  v = ", v)
            println(print_unit, "  a = ", a)
            flush(print_unit)
        elseif print_level == 5
            println(print_unit, "  istep, nfev, njev, naev, accepted: ", istep, " ", nfev, " ", njev, " ", naev, " ", accepted)
            println(print_unit, "  Cost, lam, delta: ", C, " ", lam, " ", delta)
            println(print_unit, "  av, cos_alpha: ", av, " ", cos_alpha)
            println(print_unit, "  x = ", x)
            println(print_unit, "  v = ", v)
            println(print_unit, "  a = ", a)
            flush(print_unit)
        end

        if converged != 0
            break
        end

        if accepted >= 0
            jac_uptodate = false  # Jacobian is now out of date
        end
    end

    if converged == 0
        converged = -1
    end

    # Return best fit found (write it back into the caller's arrays)
    copyto!(x, x_best)
    copyto!(fvec, fvec_best)

    if print_level >= 1
        println(print_unit, "Optimization finished")
        println(print_unit, "Results:")
        println(print_unit, "  Converged:    ", get(CONVERGED_INFO, converged, "Unknown"), " (", converged, ")")
        println(print_unit, "  Final Cost:   ", 0.5 * dot(fvec_best, fvec_best))
        if m > n
            println(print_unit, "  Cost/DOF:     ", 0.5 * dot(fvec_best, fvec_best) / (m - n))
        end
        println(print_unit, "  niters:       ", niters)
        println(print_unit, "  nfev:         ", nfev)
        println(print_unit, "  njev:         ", njev)
        println(print_unit, "  naev:         ", naev)
        flush(print_unit)
    end

    return (x, fvec, niters, nfev, njev, naev, converged)
end