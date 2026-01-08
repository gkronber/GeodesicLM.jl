# -*- julia -*-
# file dpmpar.jl

"""
    dpmpar(i::Int)

Provide double precision machine parameters.

This function provides double precision machine parameters when the 
appropriate parameters are requested. If the machine has base b digits 
and its smallest and largest exponents are emin and emax, respectively, 
then these parameters are:

- dpmpar(1) = machine precision (b^(1-t))
- dpmpar(2) = smallest magnitude (b^(emin-1))
- dpmpar(3) = largest magnitude (b^emax*(1-b^(-t)))

# Arguments
- `i`: integer set to 1, 2, or 3 to select the desired machine parameter

# Returns
- Machine parameter as a Float64
"""
function dpmpar(i::Int)
    
    # IEEE machine constants for double precision
    if i == 1
        # Machine precision (relative machine epsilon)
        return 2.22044604926e-16
    elseif i == 2
        # Smallest magnitude (smallest positive normalized number)
        return 2.22507385852e-308
    elseif i == 3
        # Largest magnitude (largest representable number)
        return 1.79769313485e+308
    else
        error("dpmpar: i must be 1, 2, or 3")
    end
end
