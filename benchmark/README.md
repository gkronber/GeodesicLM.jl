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

### Benchmark the current (dirty) state

```bash
benchpkg --path /path/to/GeodesicLM.jl --rev dirty -o ./benchresults \
         -s /path/to/GeodesicLM.jl/benchmark/benchmarks.jl
```

This benchmarks the current working tree (including uncommitted changes) and
writes `results_GeodesicLM@dirty.json` into `./benchresults`.

### Compare revisions

`benchpkg` runs benchmarks at any number of git revisions (tags, branches, or
commit hashes). For example, to compare the current state against the previous
commit:

```bash
benchpkg --path /path/to/GeodesicLM.jl --rev HEAD~1,dirty -o ./benchresults \
         -s /path/to/GeodesicLM.jl/benchmark/benchmarks.jl
```

AirspeedVelocity will check out each revision in turn, build the package, run
the frozen benchmark script, and print a markdown table with the median timings
(and a ratio column when exactly two revisions are compared).

Other useful flags:

- `--tune` — tune each benchmark with BenchmarkTools before timing.
- `-f case1,case2` — only run the named benchmarks (e.g. `-f rosenbrock`).
- `--dont-print` — skip printing the summary table.
- `--exeflags "..."` — extra flags passed to the Julia process running the
  benchmark (e.g. `-O3` for performance, since default is usually `-O2`/release
  in the runner).

See the [AirspeedVelocity README](https://github.com/MilesCranmer/AirspeedVelocity.jl)
for the full CLI reference and CI integration options.

In CI, the workflow `.github/workflows/Benchmark.yml` runs the same comparison for
every pull request that changes `src/`, `benchmark/` or `Project.toml`, and shows
the tables in the job summary.

### Without AirspeedVelocity: local revision comparison

If you do not have `benchpkg` installed, `compare_revisions.jl` provides the same
revision-vs-revision comparison using plain Julia, `git worktree`, and
BenchmarkTools (all already available in this environment):

```bash
julia benchmark/compare_revisions.jl HEAD~1 dirty
julia benchmark/compare_revisions.jl origin/main HEAD
```

It benchmarks each listed revision against the suite in
`benchmark/benchmarks.jl`, prints a markdown-style table of median timings with
a ratio column (first listed revision is the baseline), and reports memory
usage. Like `benchpkg`:

- `dirty` benchmarks the current working tree, uncommitted changes included.
- every other revision is benchmarked from a *temporary* `git worktree`, so
  your current checkout is never modified.
- each revision runs in its own Julia subprocess, guaranteeing the matching
  version of `GeodesicLM` is loaded.

Revisions that predate the benchmark suite (no `benchmark/` directory) or whose
`Project.toml` can't resolve the local `GeodesicLM` (e.g. before the `[sources]`
entry was added) are skipped with a warning.

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
