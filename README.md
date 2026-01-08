This repository holds a Julia conversion of the geodesicLM algorithm for nonlinear least squares optimization, originally implemented in Fortran by Mark Transtrum.

The initial version of the Julia code was created automatically by github copilot, followed by extensive manual refinement to ensure accuracy, performance, and idiomatic Julia style.

Relevant documents:
- [src/QUICK_START.md](QUICK_START.md): A quick start guide for using the Julia module
- [src/README_JULIA.md](README_JULIA.md): Overview of the Julia conversion
- [src/CONVERSION_SUMMARY.md](CONVERSION_SUMMARY.md): Summary of the conversion process
- [src/CONVERSION_MAPPING.md](CONVERSION_MAPPING.md): Technical details of the conversion process
- [src/geodesiclm.jl](geodesiclm.jl): Main optimization algorithm in Julia
- [src/GeodesicLM.jl](GeodesicLM.jl): Module wrapper for easy usage