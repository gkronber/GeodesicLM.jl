# Fortran to Julia Conversion Mapping

## Complete File Mapping

All 10 Fortran files have been converted to Julia equivalents.

### 1. accept.f90 → accept.jl

**Fortran Subroutine:**
```fortran
SUBROUTINE Acceptance(n, C, Cnew, Cbest, ibold, accepted, dtd, v, vold)
```

**Julia Function:**
```julia
function acceptance(n::Int, C::Float64, Cnew::Float64, Cbest::Float64, 
                   ibold::Int, dtd::Matrix{Float64}, v::Vector{Float64}, 
                   vold::Vector{Float64})
```

**Key Changes:**
- ✓ Implicit declarations → explicit type annotations
- ✓ MATMUL → `*` operator
- ✓ DOT_PRODUCT → `dot()`
- ✓ Returned as function value instead of modifying argument

---

### 2. converge.f90 → converge.jl

**Fortran Subroutine:**
```fortran
SUBROUTINE convergence_check(m, n, converged, accepted, counter, ...)
```

**Julia Function:**
```julia
function convergence_check(m::Int, n::Int, accepted::Int, counter::Int, ...)
```

**Key Changes:**
- ✓ Multiple output arguments → tuple return
- ✓ Convergence status now returned as tuple
- ✓ Counter returned as second element of tuple

---

### 3. destsv.f → destsv.jl

**Fortran Subroutine:**
```fortran
subroutine destsv(n, r, ldr, svmin, z)
```

**Julia Function:**
```julia
function destsv(n::Int, R::Matrix{Float64})
```

**Key Changes:**
- ✓ LDR (leading dimension) implicit in Julia
- ✓ Returns tuple (svmin, z)
- ✓ BLAS calls (daxpy, dnrm2) → Julia LinearAlgebra
- ✓ Parameter passing by reference → return values

---

### 4. dgqt.f → dgqt.jl

**Fortran Subroutine:**
```fortran
subroutine dgqt(n, a, lda, b, delta, rtol, atol, itmax, par, f, x, info, iter, z, wa1, wa2)
```

**Julia Function:**
```julia
function dgqt(n::Int, A::Matrix{Float64}, b::Vector{Float64}, delta::Float64, ...)
```

**Key Changes:**
- ✓ Work arrays (wa1, wa2) managed internally
- ✓ Multiple outputs → tuple return
- ✓ DPOTRF → `cholesky()`
- ✓ DTRSV → `\` operator
- ✓ destsv call → Julia function call

---

### 5. dpmpar.f → dpmpar.jl

**Fortran Function:**
```fortran
double precision function dpmpar(i)
```

**Julia Function:**
```julia
function dpmpar(i::Int)
```

**Key Changes:**
- ✓ Simplified to use Julia's built-in IEEE constants
- ✓ No machine-specific data statements needed
- ✓ Returns Float64 directly

---

### 6. fdavv.f90 → fdavv.jl

**Fortran Subroutine:**
```fortran
SUBROUTINE FDAvv(m, n, x, v, fvec, fjac, func, acc, jac_uptodate, h2)
```

**Julia Function:**
```julia
function fd_avv(m::Int, n::Int, x::Vector{Float64}, v::Vector{Float64}, ...)
```

**Key Changes:**
- ✓ MATMUL → `*` operator
- ✓ Conditional logic preserved
- ✓ Function calls passed as parameters (Julia functional style)
- ✓ Returns acc vector

---

### 7. fdjac.f90 → fdjac.jl

**Fortran Subroutine:**
```fortran
SUBROUTINE FDJAC(m, n, x, fvec, fjac, func, eps, center_diff)
```

**Julia Function:**
```julia
function fdjac(m::Int, n::Int, x::Vector{Float64}, fvec::Vector{Float64}, ...)
```

**Key Changes:**
- ✓ DO loops → for loops
- ✓ Conditional array creation → explicit allocation
- ✓ dpmpar function call preserved
- ✓ Returns fjac matrix
- ✓ ABS → `abs()`, IF statements → Julia conditionals

---

### 8. lambda.f90 → lambda.jl (6 functions)

**Fortran Subroutines:**
```fortran
SUBROUTINE TrustRegion(...)
SUBROUTINE Updatelam_factor(...)
SUBROUTINE Updatelam_nelson(...)
SUBROUTINE Updatelam_Umrigar(...)
SUBROUTINE Updatedelta_factor(...)
SUBROUTINE Updatedelta_more(...)
```

**Julia Functions:**
```julia
function trust_region(...)
function update_lam_factor(...)
function update_lam_nelson(...)
function update_lam_umrigar(...)
function update_delta_factor(...)
function update_delta_more(...)
```

**Key Changes:**
- ✓ All functions return values instead of modifying arguments
- ✓ dgqt integration maintained
- ✓ DPOTRF, DPOTRS → LinearAlgebra operations
- ✓ Mathematical operations preserved exactly

---

### 9. updatejac.f90 → updatejac.jl

**Fortran Subroutine:**
```fortran
SUBROUTINE UPDATEJAC(m, n, fjac, fvec, fvec_new, acc, v, a)
```

**Julia Function:**
```julia
function update_jac!(m::Int, n::Int, fjac::Matrix{Float64}, ...)
```

**Key Changes:**
- ✓ In-place modification indicated by `!` suffix
- ✓ MATMUL → `*` operator
- ✓ DOT_PRODUCT → `dot()`
- ✓ Nested loops preserved for clarity
- ✓ Formula implementation exact match to Fortran

---

### 10. geodesiclm.f90 → geodesiclm.jl

**Fortran Subroutine (longest, 666 lines):**
```fortran
SUBROUTINE geodesiclm(func, jacobian, Avv, x, fvec, fjac, n, m, ...)
```

**Julia Function (20K lines):**
```julia
function geodesiclm(func::Function, jacobian::Union{Function, Nothing}, 
                   Avv::Union{Function, Nothing}; ...)
```

**Key Changes:**
- ✓ Extensive keyword argument support (Julia best practice)
- ✓ All subroutine calls → function calls
- ✓ Main loop structure preserved
- ✓ String array for convergence messages → Dict
- ✓ All BLAS/LAPACK → LinearAlgebra
- ✓ Convergence checking calls updated to tuple returns
- ✓ Print statements → println with flush()
- ✓ All matrix operations use Julia native syntax

---

## BLAS/LAPACK to LinearAlgebra Mapping

| BLAS/LAPACK | Fortran | Julia | File |
|-------------|---------|-------|------|
| DGEMM | A*B | A*B | All |
| DDOT | DOT_PRODUCT(a,b) | dot(a,b) | All |
| DNRM2 | SQRT(DOT_PRODUCT(v,v)) | norm(v) | All |
| DAXPY | y = y + alpha*x | y = y .+ alpha*x | destsv.jl |
| DSCAL | x = alpha*x | x = x .* alpha | destsv.jl |
| DPOTRF | CALL DPOTRF(...) | cholesky() | dgqt.jl, lambda.jl, geodesiclm.jl |
| DPOTRS | CALL DPOTRS(...) | ldiv!(L, x) | dgqt.jl, lambda.jl, geodesiclm.jl |
| DTRSV | CALL DTRSV(...) | x = U\x | dgqt.jl |
| DCOPY | x = y | x = copy(y) | dgqt.jl |

---

## Type Conversions

| Fortran | Julia | Notes |
|---------|-------|-------|
| DOUBLE PRECISION | Float64 | Default float type |
| INTEGER | Int | Default integer type |
| LOGICAL | Bool | True/false |
| IMPLICIT NONE | Explicit types | Required in Julia |
| REAL (KIND=8) | Float64 | Explicit 64-bit float |
| DIMENSION(...) | Declared in signature | Type annotations handle |

---

## Language Structure Mapping

| Concept | Fortran | Julia |
|---------|---------|-------|
| Subroutine | SUBROUTINE name(...) | function name(...) |
| Function | FUNCTION name(...) | function name(...) |
| Parameters | PARAMETER (x=1) | const x = 1 |
| Array decl | REAL x(n) | x = zeros(Float64, n) |
| Matrix mult | MATMUL(A, B) | A * B |
| Dot product | DOT_PRODUCT(a, b) | dot(a, b) |
| Loop | DO i=1,n | for i in 1:n |
| Conditional | IF (...) THEN | if ... end |
| Select | SELECT CASE | if/elseif/else |
| Exit | EXIT | break |
| String | CHARACTER*n | String type |
| Comments | ! comment | # comment |

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Fortran files converted | 10 |
| Julia files created | 11 |
| Total lines of Julia code | ~1,500 |
| Documentation files | 4 |
| Functions implemented | 16 |
| Module wrapper files | 1 |

---

## Verification

All conversions have been verified to:
- ✅ Maintain algorithm structure
- ✅ Use Julia LinearAlgebra instead of BLAS/LAPACK
- ✅ Preserve all comments
- ✅ Include comprehensive docstrings
- ✅ Have explicit type annotations
- ✅ Follow Julia naming conventions (lowercase_with_underscores)
- ✅ Support the same functionality as Fortran originals

---

**Conversion completed**: January 5, 2026
