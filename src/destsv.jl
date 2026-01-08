# -*- julia -*-
# file destsv.jl

using LinearAlgebra

"""
    destsv(n::Int, R::Matrix{Float64})

Estimate the smallest singular value and associated singular vector of an 
n by n upper triangular matrix R.

Given an n by n upper triangular matrix R, this function estimates the smallest 
singular value and the associated singular vector of R.

In the algorithm a vector e is selected so that the solution y to the system 
R'*y = e is large. The choice of sign for the components of e cause maximal 
local growth in the components of y as the forward substitution proceeds. The 
vector z is the solution of the system R*z = y, and the estimate svmin is 
norm(y)/norm(z) in the Euclidean norm.

# Arguments
- `n`: order of R
- `R`: n by n upper triangular matrix

# Returns
- `(svmin, z)`: tuple with estimated smallest singular value and associated singular vector
"""
function destsv(n::Int, R::Matrix{Float64})
    
    const_p01 = 1.0e-2
    const_one = 1.0
    const_zero = 0.0
    
    # Initialize z
    z = zeros(Float64, n)
    
    # This choice of e makes the algorithm scale invariant.
    e = abs(R[1, 1])
    
    if e == const_zero
        svmin = const_zero
        z[1] = const_one
        return (svmin, z)
    end
    
    # Solve R'*y = e.
    for i in 1:n
        e = sign(e) * (-z[i])
        
        # Scale y. The factor of 0.01 reduces the number of scalings.
        if abs(e - z[i]) > abs(R[i, i])
            temp = min(const_p01, abs(R[i, i]) / abs(e - z[i]))
            z[1:n] .*= temp
            e = temp * e
        end
        
        # Determine the two possible choices of y(i).
        if R[i, i] == const_zero
            w = const_one
            wm = const_one
        else
            w = (e - z[i]) / R[i, i]
            wm = -(e + z[i]) / R[i, i]
        end
        
        # Choose y(i) based on the predicted value of y(j) for j > i.
        s = abs(e - z[i])
        sm = abs(e + z[i])
        
        for j in (i+1):n
            sm = sm + abs(z[j] + wm * R[i, j])
        end
        
        if i < n
            # Add contribution from upper triangular part
            for j in (i+1):n
                z[j] = z[j] + w * R[i, j]
            end
            s = s + sum(abs.(z[(i+1):n]))
        end
        
        if s < sm
            temp = wm - w
            w = wm
            if i < n
                for j in (i+1):n
                    z[j] = z[j] + temp * R[i, j]
                end
            end
        end
        z[i] = w
    end
    
    ynorm = norm(z)
    
    # Solve R*z = y.
    for j in n:-1:1
        # Scale z.
        if abs(z[j]) > abs(R[j, j])
            temp = min(const_p01, abs(R[j, j]) / abs(z[j]))
            z[1:n] .*= temp
            ynorm = temp * ynorm
        end
        
        if R[j, j] == const_zero
            z[j] = const_one
        else
            z[j] = z[j] / R[j, j]
        end
        
        temp = -z[j]
        for i in 1:(j-1)
            z[i] = z[i] + temp * R[i, j]
        end
    end
    
    # Compute svmin and normalize z.
    znorm = 1.0 / norm(z)
    svmin = ynorm * znorm
    z = z .* znorm
    
    return (svmin, z)
end
