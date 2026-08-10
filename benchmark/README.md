# GeodesicLM.jl benchmarks

This directory contains the benchmark suite for GeodesicLM.jl, run with
[AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl).
AirspeedVelocity lets you compare the runtime and memory usage of GeodesicLM
across git revisions, which is useful when improving or extending the code.

## Layout

- `benchmarks.jl` — the benchmark suite. It defines `const SUITE =
  BenchmarkGroup()` (BenchmarkTools), with a few *central* benchmark problems,
  each run under the two main solver configurations:
  - `fd` — finite-difference Jacobian and (where applicable) finite-difference
    geodesic acceleration (the default configuration).
  - `analytic` — analytic Jacobian and acceleration provided by the user.

  The problems are:

  | Group                 | Description                                          | n | m  |
  |-----------------------|------------------------------------------------------|---|----|
  | `quadratic`           | over-determined linear least squares                 | 2 | 30 |
  | `rosenbrock`          | classic hard nonlinear test function                 | 2 | 2  |
  | `exponential_decay`   | nonlinear fit (one exponential) to noisy data        | 2 | 50 |
  | `sloppy_sum_exps`     | near-sloppy sum-of-exponentials fit (degenerate dirs)| 5 | 80 |

- `Project.toml` — benchmark environment declaring the dependencies needed to
  run the suite (`GeodesicLM`, `BenchmarkTools`, `LinearAlgebra`).

## Running

### Requirements
You need to build AirspeedVelocity.
```bash
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("AirspeedVelocity"); Pkg.build("AirspeedVelocity")'
```
This places binaries like `benchpkg` (and `benchpkgtable`) into ~/.julia/bin

### Run the benchmarks

Just run `benchmark/run_benchmark.sh` with any arguments you would otherwise
pass to `benchpkg` (they are forwarded unchanged). After benchmarking it
automatically calls `benchpkgtable` to print **both** the runtime table and the
memory/allocation table:

```bash
# current working tree (including uncommitted changes)
./benchmark/run_benchmark.sh --rev dirty

# compare the previous commit against the current tree
./benchmark/run_benchmark.sh --rev HEAD~1,dirty

# save the JSON results into a custom directory
./benchmark/run_benchmark.sh --rev HEAD~1,dirty -o ./benchresults

# tune, and only run the rosenbrock benchmarks
./benchmark/run_benchmark.sh --rev HEAD~1,dirty --tune -f rosenbrock
```

Useful options (identical to `benchpkg`, see `benchpkg --help`):

- `-r/--rev <rev1,rev2>` — revisions to benchmark (`dirty` = current working
  tree). The JSON is written as `results_GeodesicLM@<rev>.json` into the output
  directory, and the printed tables show a ratio column when exactly two
  revisions are compared.
- `-o/--output-dir <dir>` — where to save the JSON results (default `.`). The
  directory is created if needed.
- `-f case1,case2` — only run the named benchmarks (e.g. `-f rosenbrock`).
- `-s/--script <file>` — the benchmark script (defaults to this suite's
  `benchmark/benchmarks.jl`).
- `--tune` — tune each benchmark with BenchmarkTools before timing.
- `--exeflags "..."` — extra flags passed to the Julia process running the
  benchmark (e.g. `-O3` for performance).
- `--path <dir>` — path of the package (defaults to this checkout).

`--path`/`--script` are set automatically to this repository, and the two
result tables (median timings, then allocations/memory) are always printed via
`benchpkgtable`. Set `BENCHPKG_TABLE_MODE` to `time`, `memory`, or
`time,memory` to change which tables are printed. For example:

```bash
BENCHPKG_TABLE_MODE=memory ./benchmark/run_benchmark.sh --rev dirty
```

See the [AirspeedVelocity README](https://github.com/MilesCranmer/AirspeedVelocity.jl)
for the full CLI reference and CI integration options.

### Without AirspeedVelocity

If you just want to run the suite once without comparing revisions, you can do so
directly with BenchmarkTools:

```julia
using Pkg
Pkg.activate("benchmark")
Pkg.instantiate()
include("benchmark/benchmarks.jl")
using BenchmarkTools
results = run(SUITE; verbose=true)
```

## Notes

- Keep the benchmark problems fixed and deterministic (fixed data sets, fixed
  starting points) so different commits can be compared fairly. If you add new
  features, extend `SUITE` with new `BenchmarkGroup`s rather than changing the
  parameters of existing ones.
- The exact number of iterations each run takes depends on the optimizer's
  convergence path, so absolute timings are meaningful for comparing revisions
  of the *algorithm*, not as micro-optimization measurements.
