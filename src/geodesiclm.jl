# -*- julia -*-
# file geodesiclm.jl
# Main Geodesic-Bold-BroydenUpdate-Levenberg-Marquardt routine
# version 1.0.2

using LinearAlgebra

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
               factorreject::Float64=2.0, avmax::Float64=10.0)

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
 - `callback`: User-supplied callback function called after each iteration
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
                   factorreject::Float64=2.0, avmax::Float64=10.0)
    
    # Initialize internal parameters
    acc = zeros(Float64, m)
    v = zeros(Float64, n)
    vold = zeros(Float64, n)
    a = zeros(Float64, n)
    lam = 0.0
    delta = 0.0
    cos_alpha = 1.0
    av = 0.0
    
    fvec_new = zeros(Float64, m)
    fvec_best = copy(fvec)
    x_new = copy(x)
    x_best = copy(x)
    
    jtj = zeros(Float64, n, n)
    g = zeros(Float64, n, n)
    
    temp1 = 0.0
    temp2 = 0.0
    pred_red = 0.0
    dirder = 0.0
    actred = 0.0
    rho = 0.0
    a_param = 0.5
    
    # Convergence status strings
    converged_info = Dict(
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
        -11 => "NaN Produced"
    )
    
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
    
    v .= 0.0
    vold .= 0.0
    a .= 0.0
    cos_alpha = 1.0
    av = 0.0
    a_param = 0.5
    
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
    valid_result = !any(isnan.(fvec))
    if !valid_result
        converged = -11
        maxiter = 0
    end
    
    Cbest = C
    fvec_best = copy(fvec)
    x_best = copy(x)
    
    # Compute initial Jacobian
    if analytic_jac && jacobian !== nothing
        jacobian(x, fjac)
        njev = njev + 1
    else
        fjac = fdjac(m, n, x, fvec, func, h1, center_diff)
        if center_diff
            nfev = nfev + 2 * n
        else
            nfev = nfev + n
        end
    end
    
    jac_uptodate = true
    jac_force_update = false
    jtj = fjac' * fjac
    
    # Check fjac for NaNs
    valid_result = !any(isnan.(fjac))
    if !valid_result
        converged = -11
        maxiter = 0
    end
    
    acc .= 0.0
    a .= 0.0
    
    # Initialize damping matrix
    if dtd === nothing
        dtd = zeros(Float64, n, n)
    end
    
    if damp_mode == 0
        # Identity matrix
        dtd .= 0.0
        for i in 1:n
            dtd[i, i] = 1.0
        end
    elseif damp_mode == 1
        # Diagonal scaling
        for i in 1:n
            dtd[i, i] = max(jtj[i, i], dtd[i, i])
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
        delta = initialfactor * sqrt(dot(x, dtd * x))
        lam = 1.0
        if delta == 0.0
            delta = 100.0
        end
        if converged == 0
            (v, lam) = trust_region(n, m, fvec, fjac, dtd, delta)
        end
    end
    
    # Main optimization loop
    for istep in 1:maxiter
        
        info = 0
        if callback !== nothing
            callback(x, v, a, fvec, fjac, acc, lam, dtd, fvec_new, accepted, info)
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
            update_jac!(m, n, fjac, fvec, fvec_new, acc, v, a)
            jac_uptodate = false
        end
        
        if accepted > 0
            # Accepted step
            fvec = copy(fvec_new)
            x = copy(x_new)
            vold = copy(v)
            C = Cnew
            if C <= Cbest
                x_best = copy(x)
                Cbest = C
                fvec_best = copy(fvec)
            end
        end
        
        if jac_force_update
            # Full rank update of Jacobian
            if analytic_jac && jacobian !== nothing
                jacobian(x, fjac)
                njev = njev + 1
            else
                fjac = fdjac(m, n, x, fvec, func, h1, center_diff)
                if center_diff
                    nfev = nfev + 2 * n
                else
                    nfev = nfev + n
                end
            end
            jac_uptodate = true
            jac_force_update = false
        end
        
        # Check fjac for NaNs
        valid_result = !any(isnan.(fjac))
        
        if valid_result
            # If no NaNs in Jacobian
            jtj = fjac' * fjac
            
            # Update Scaling/lam/TrustRegion
            if istep > 1
                if damp_mode == 1
                    # Update diagonal scaling
                    for i in 1:n
                        dtd[i, i] = max(jtj[i, i], dtd[i, i])
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
                    (lam, a_param) = update_lam_umrigar(m, n, lam, accepted, v, vold, fvec, fjac, dtd, a_param, C, Cnew)
                elseif imethod == 10
                    # Update delta by fixed factors
                    delta = update_delta_factor(delta, accepted, factoraccept, factorreject)
                    (v, lam) = trust_region(n, m, fvec, fjac, dtd, delta)
                elseif imethod == 11
                    # Update delta as described in Moré reference
                    (delta, lam) = update_delta_more(delta, lam, n, v, dtd, rho, C, Cnew, dirder, actred, av, avmax)
                    (v, lam) = trust_region(n, m, fvec, fjac, dtd, delta)
                end
            end
            
            # Propose Step
            g = jtj + lam * dtd
            
            # Cholesky decomposition
            try
                L = cholesky(Hermitian(g, :U))
                info = 0
            catch
                info = 1
            end
            
            if info == 0
                # If matrix decomposition successful:
                v = -1.0 * (fvec' * fjac)[:]
                ldiv!(L.U, v)  # Solve for v
                v = -v  # Change sign because we solved U'*U*x = -J'*f
                
                # This is not quite right; let me recompute correctly
                # v = -J'*f, and we solve (J'*J + lambda*dtd)*v = -J'*f
                v = -1.0 * (fvec' * fjac)[:]
                sol = L \ v  # Cholesky solve
                v = sol[:]
                
                # Calculate the predicted reduction and directional derivative
                temp1 = 0.5 * dot(v, jtj * v) / C
                temp2 = 0.5 * lam * dot(v, dtd * v) / C
                pred_red = temp1 + 2.0 * temp2
                dirder = -1.0 * (temp1 + temp2)
                
                # Calculate cos_alpha -- cos of angle between step direction and residual
                jv = fjac * v
                cos_alpha = abs(dot(fvec, jv)) / (sqrt(dot(fvec, fvec)) * sqrt(dot(jv, jv)))
                
                if imethod < 10
                    delta = sqrt(dot(v, dtd * v))
                end
                
                # Update acceleration
                if iaccel > 0
                    if analytic_Avv && Avv !== nothing
                        Avv(x, v, acc)
                        naev = naev + 1
                    else
                        acc = fd_avv(m, n, x, v, fvec, fjac, func, jac_uptodate, h2)
                        if jac_uptodate
                            nfev = nfev + 1
                        else
                            nfev = nfev + 2  # We don't use Jacobian if not up to date
                        end
                    end
                    
                    # Check acceleration for NaNs
                    if !any(isnan.(acc))
                        a = -1.0 * (acc' * fjac)[:]
                        sol = L \ a  # Solve for a
                        a = sol[:]
                    else
                        a .= 0.0  # If NaNs in acc, ignore acceleration term
                    end
                end
                
                # Evaluate at proposed step -- only necessary if av <= avmax
                av = sqrt(dot(a, dtd * a) / dot(v, dtd * v))
                
                if av <= avmax
                    x_new = x + v + 0.5 * a
                    func(x_new, fvec_new)
                    nfev = nfev + 1
                    Cnew = 0.5 * dot(fvec_new, fvec_new)
                    Cold = C
                    
                    # Check for NaNs in fvec_new
                    if !any(isnan.(fvec_new))
                        # If no NaNs, proceed as normal
                        actred = 1.0 - Cnew / C
                        rho = 0.0
                        if pred_red != 0.0
                            rho = (1.0 - Cnew / C) / pred_red
                        end
                        
                        # Accept or Reject proposed step
                        accepted = acceptance(n, C, Cnew, Cbest, ibold, dtd, v, vold)
                    else
                        # If NaNs in fvec_new, reject step
                        actred = 0.0
                        rho = 0.0
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
        else
            # If NaNs in Jacobian
            converged = -11
            break
        end
        
        # Check Convergence
        if converged == 0
            (converged, counter) = convergence_check(m, n, accepted, counter, C, Cnew, x, fvec, fjac, lam, 
                                                    x_new, nfev, maxfev, njev, maxjev, naev, maxaev, maxlam, 
                                                    minlam, artol, Cgoal, gtol, xtol, xrtol, ftol, frtol, cos_alpha)
            
            if converged == 1 && !jac_uptodate
                # If converged by artol with out-of-date Jacobian, update to confirm
                converged = 0
                jac_force_update = true
            end
        end
        
        # Print status
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
    
    niters = istep
    
    # Return best fit found
    x = copy(x_best)
    fvec = copy(fvec_best)
    
    if print_level >= 1
        println(print_unit, "Optimization finished")
        println(print_unit, "Results:")
        println(print_unit, "  Converged:    ", get(converged_info, converged, "Unknown"), " (", converged, ")")
        println(print_unit, "  Final Cost:   ", 0.5 * dot(fvec, fvec))
        if m > n
            println(print_unit, "  Cost/DOF:     ", 0.5 * dot(fvec, fvec) / (m - n))
        end
        println(print_unit, "  niters:       ", niters)
        println(print_unit, "  nfev:         ", nfev)
        println(print_unit, "  njev:         ", njev)
        println(print_unit, "  naev:         ", naev)
        flush(print_unit)
    end
    
    return (x, fvec, niters, nfev, njev, naev, converged)
end
