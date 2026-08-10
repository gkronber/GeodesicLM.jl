# -*- julia -*-
# file dgqt.jl

using LinearAlgebra

"""
    dgqt(n::Int, A::Matrix{Float64}, b::Vector{Float64}, delta::Float64, 
         rtol::Float64=1.0e-3, atol::Float64=1.0e-3, itmax::Int=10, par::Float64=0.0)

Minimize a quadratic function subject to a Euclidean norm constraint.

Given an n by n symmetric matrix A, an n-vector b, and a positive number delta, 
this function determines a vector x which approximately minimizes the quadratic function

    f(x) = (1/2)*x'*A*x + b'*x

subject to the Euclidean norm constraint norm(x) <= delta.

This function computes an approximation x and a Lagrange multiplier par such that 
either par is zero and norm(x) <= (1+rtol)*delta, or par is positive and 
abs(norm(x) - delta) <= rtol*delta.

# Arguments
- `n`: order of A
- `A`: n by n symmetric matrix
- `b`: n-vector specifying the linear term in the quadratic
- `delta`: bound on the Euclidean norm of x
- `rtol`: relative accuracy desired in the solution
- `atol`: absolute accuracy desired in the solution
- `itmax`: maximum number of iterations
- `par`: initial estimate of the Lagrange multiplier

# Returns
- `(x, par, info, f)`: solution vector, Lagrange multiplier, convergence info, and final function value
"""
function dgqt(n::Int, A_input::Matrix{Float64}, b::Vector{Float64}, delta::Float64,
              rtol::Float64=1.0e-3, atol::Float64=1.0e-3, itmax::Int=10, par::Float64=0.0)
    
    # Constants
    const_p001 = 1.0e-3
    const_p5 = 0.5
    const_zero = 0.0
    const_one = 1.0
    
    # Initialize output arrays
    x = zeros(Float64, n)
    z = zeros(Float64, n)
    wa1 = zeros(Float64, n)
    wa2 = zeros(Float64, n)
    
    # Work with a copy of A since we modify it
    A = copy(A_input)
    
    # Initialize variables
    parf = const_zero
    xnorm = const_zero
    rxnorm = const_zero
    rednc = false
    info = 0
    iter = 0
    
    # Copy the diagonal and save A in its lower triangle.
    wa1 = diag(A)
    for j in 1:(n-1)
        A[(j+1):n, j] = A[j, (j+1):n]
    end
    
    # Calculate the l1-norm of A, the Gershgorin row sums, and the l2-norm of b.
    anorm = const_zero
    for j in 1:n
        wa2[j] = sum(abs.(A[:, j]))
        anorm = max(anorm, wa2[j])
    end
    
    for j in 1:n
        wa2[j] = wa2[j] - abs(wa1[j])
    end
    
    bnorm = norm(b)
    
    # Calculate a lower bound, pars, for the domain of the problem.
    # Also calculate an upper bound, paru, and a lower bound, parl, 
    # for the Lagrange multiplier.
    pars = -anorm
    parl = -anorm
    paru = -anorm
    
    for j in 1:n
        pars = max(pars, -wa1[j])
        parl = max(parl, wa1[j] + wa2[j])
        paru = max(paru, -wa1[j] + wa2[j])
    end
    
    parl = max(const_zero, bnorm / delta - parl, pars)
    paru = max(const_zero, bnorm / delta + paru)
    
    # If the input par lies outside of the interval (parl,paru),
    # set par to the closer endpoint.
    par = max(par, parl)
    par = min(par, paru)
    
    # Special case: parl = paru
    paru = max(paru, (1.0 + rtol) * parl)
    
    # Beginning of an iteration.
    f = 0.0
    for iter in 1:itmax
        
        # Safeguard par.
        if par <= pars && paru > const_zero
            par = max(const_p001, sqrt(parl / paru)) * paru
        end
        
        # Copy the lower triangle of A into its upper triangle and
        # compute A + par*I.
        for j in 1:(n-1)
            A[j, (j+1):n] = A[(j+1):n, j]
        end
        
        for j in 1:n
            A[j, j] = wa1[j] + par
        end
        
        # Attempt the Cholesky factorization of A without referencing
        # the lower triangular part.
        # (hoist these because `try` introduces a new scope)
        indef = 1
        L = nothing
        try
            L = cholesky(Hermitian(A, :U))
            indef = 0
        catch
            indef = 1
        end
        
        # Case 1: A + par*I is positive definite.
        if indef == 0
            
            # Compute an approximate solution x and save the
            # last value of par with A + par*I positive definite.
            parf = par
            wa2 = copy(b)
            wa2 = L.U' \ wa2  # Solve U'*y = b
            rxnorm = norm(wa2)
            x = L.U \ wa2     # Solve U*x = y
            x = -x
            xnorm = norm(x)
            
            # Test for convergence.
            if abs(xnorm - delta) <= rtol * delta || 
               (par == const_zero && xnorm <= (1.0 + rtol) * delta)
                info = 1
            end
            
            # Compute a direction of negative curvature and use this
            # information to improve pars.
            (rznorm, z) = destsv(n, L.U)
            pars = max(pars, par - rznorm^2)
            
            # Compute a negative curvature solution of the form
            # x + alpha*z where norm(x+alpha*z) = delta.
            rednc = false
            if xnorm < delta
                
                # Compute alpha
                prod = dot(z, x) / delta
                temp = (delta - xnorm) * ((delta + xnorm) / delta)
                alpha = temp / (abs(prod) + sqrt(prod^2 + temp / delta))
                alpha = sign(alpha) * abs(alpha)
                if prod < 0.0
                    alpha = -alpha
                end
                
                # Test to decide if the negative curvature step
                # produces a larger reduction than with z = 0.
                rznorm = abs(alpha) * rznorm
                if (rznorm / delta)^2 + par * (xnorm / delta)^2 <= par
                    rednc = true
                end
                
                # Test for convergence.
                if const_p5 * (rznorm / delta)^2 <= 
                   rtol * (1.0 - const_p5 * rtol) * (par + (rxnorm / delta)^2)
                    info = 1
                elseif const_p5 * (par + (rxnorm / delta)^2) <= (atol / delta) / delta && info == 0
                    info = 2
                elseif xnorm == const_zero
                    info = 1
                end
            end
            
            # Compute the Newton correction parc to par.
            if xnorm == const_zero
                parc = -par
            else
                wa2 = copy(x)
                temp = 1.0 / xnorm
                wa2 = wa2 .* temp
                wa2 = L.U' \ wa2  # Solve U'*y = wa2
                temp = norm(wa2)
                parc = (((xnorm - delta) / delta) / temp) / temp
            end
            
            # Update parl or paru.
            if xnorm > delta
                parl = max(parl, par)
            end
            if xnorm < delta
                paru = min(paru, par)
            end
        else
            # Case 2: A + par*I is not positive definite.
            
            # Use the rank information from the Cholesky decomposition to update par.
            # For simplicity, we use a different approach in Julia
            parc = -par * 0.1  # Simple update
            pars = max(pars, par, par + parc)
            paru = max(paru, (1.0 + rtol) * pars)
        end
        
        # Use pars to update parl.
        parl = max(parl, pars)
        
        # Test for termination.
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
        
        # If exiting, store the best approximation
        if info != 0
            par = parf
            f = -const_p5 * (rxnorm^2 + par * xnorm^2)
            if rednc
                f = -const_p5 * ((rxnorm^2 + par * delta^2) - rznorm^2)
                x = x + alpha * z
            end
            break
        end
        
        # Compute an improved estimate for par.
        par = max(parl, par + parc)
    end
    
    if info == 0
        info = 4
    end
    
    return (x, par, info, f)
end
