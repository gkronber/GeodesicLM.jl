This repository holds a Julia conversion of the geodesicLM algorithm for nonlinear least squares optimization, originally implemented in Fortran by Mark Transtrum.

The initial version of the Julia code was created automatically by github copilot, followed by extensive manual refinement to ensure accuracy, performance, and idiomatic Julia style.

Relevant documents:
- [QUICK_START.md](src/QUICK_START.md): A quick start guide for using the Julia module
- [README_JULIA.md](src/README_JULIA.md): Overview of the Julia conversion
- [CONVERSION_SUMMARY.md](src/CONVERSION_SUMMARY.md): Summary of the conversion process
- [CONVERSION_MAPPING.md](src/CONVERSION_MAPPING.md): Technical details of the conversion process
- [geodesiclm.jl](src/geodesiclm.jl): Main optimization algorithm in Julia
- [GeodesicLM.jl](src/GeodesicLM.jl): Module wrapper for easy usage