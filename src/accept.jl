# -*- julia -*-
# file accept.jl

"""
    acceptance(n::Int, C, Cnew, Cbest, ibold::Int, dtd, v, vold, tmp = similar(v))

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
- `tmp`: length-`n` scratch vector (supply one to keep the call allocation-free)

# Returns
- `accepted`: acceptance status (>0 for accepted, <0 for rejected)
"""
function acceptance(n::Int, C::T, Cnew::T, Cbest::T, ibold::Int,
                    dtd::AbstractMatrix{T}, v::AbstractVector{T},
                    vold::AbstractVector{T},
                    tmp::AbstractVector{T} = similar(v)) where {T<:AbstractFloat}

    accepted = 0

    if Cnew <= C
        # Accept all downhill steps
        accepted = max(accepted + 1, 1)
    else
        # Calculate beta
        if dot(vold, vold) == zero(T)
            beta = one(T)
        else
            mul!(tmp, dtd, vold)
            beta = dot(v, tmp)
            vold_dtd_vold = dot(vold, tmp)
            mul!(tmp, dtd, v)
            beta = beta / sqrt(dot(v, tmp) * vold_dtd_vold)
            beta = min(one(T), one(T) - beta)
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
