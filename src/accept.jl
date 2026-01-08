# -*- julia -*-
# file accept.jl

"""
    acceptance(n::Int, C::Float64, Cnew::Float64, Cbest::Float64, ibold::Int, 
               dtd::Matrix{Float64}, v::Vector{Float64}, vold::Vector{Float64})

Accept or reject a step based on the cost function value and the bold acceptance criterion.

# Arguments
- `n`: number of parameters
- `C`: current cost
- `Cnew`: new cost
- `Cbest`: best cost found so far
- `ibold`: bold acceptance criterion type (0-4)
- `dtd`: damping matrix
- `v`: current step
- `vold`: previous step

# Returns
- `accepted`: acceptance status (>0 for accepted, <0 for rejected)
"""
function acceptance(n::Int, C::Float64, Cnew::Float64, Cbest::Float64, ibold::Int, 
                   dtd::Matrix{Float64}, v::Vector{Float64}, vold::Vector{Float64})
    
    accepted = 0
    
    if Cnew <= C
        # Accept all downhill steps
        accepted = max(accepted + 1, 1)
    else
        # Calculate beta
        if dot(vold, vold) == 0.0
            beta = 1.0
        else
            beta = dot(v, dtd * vold)
            beta = beta / sqrt(dot(v, dtd * v) * dot(vold, dtd * vold))
            beta = min(1.0, 1.0 - beta)
        end
        
        if ibold == 0
            # Only downhill steps
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
