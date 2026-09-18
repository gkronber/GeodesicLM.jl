# -*- julia -*-
# file dpmpar.jl

"""
    dpmpar(::Type{T}, i::Int) where {T<:AbstractFloat}
    dpmpar(i::Int)

Provide floating-point machine parameters for `T` (`Float64` when omitted).

If the machine has base b digits and its smallest and largest exponents are
emin and emax, respectively, then these parameters are:

- `dpmpar(T, 1)` = machine precision (b^(1-t))
- `dpmpar(T, 2)` = smallest magnitude (b^(emin-1))
- `dpmpar(T, 3)` = largest magnitude (b^emax*(1-b^(-t)))

# Arguments
- `T`: floating-point type the parameters are requested for
- `i`: integer set to 1, 2, or 3 to select the desired machine parameter

# Returns
- Machine parameter as a `T`
"""
function dpmpar(::Type{T}, i::Int) where {T<:AbstractFloat}
    if i == 1
        # Machine precision (relative machine epsilon)
        return eps(T)
    elseif i == 2
        # Smallest magnitude (smallest positive normalized number)
        return floatmin(T)
    elseif i == 3
        # Largest magnitude (largest representable number)
        return floatmax(T)
    else
        error("dpmpar: i must be 1, 2, or 3")
    end
end

dpmpar(i::Int) = dpmpar(Float64, i)
