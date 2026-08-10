#!/usr/bin/env bash
# -*- mode: sh -*-
# ---------------------------------------------------------------------------
# run_benchmark.sh — run the GeodesicLM benchmark suite with AirspeedVelocity
# and print both the runtime and the memory/allocation tables.
#
# All arguments are forwarded unchanged to `benchpkg`, so anything you could
# pass to `benchpkg` works here too (see `benchpkg --help`). After the
# benchmarks finish, the script calls `benchpkgtable` on the saved JSON to
# print the results (runtime + memory by default).
#
# Examples (assume the package lives in the default location):
#   ./benchmark/run_benchmark.sh --rev dirty                     # current tree
#   ./benchmark/run_benchmark.sh --rev HEAD~1,dirty              # vs previous commit
#   ./benchmark/run_benchmark.sh --rev HEAD~1,dirty -o ./res     # custom output dir
#   ./benchmark/run_benchmark.sh --rev HEAD~1,dirty --tune -f rosenbrock
#
# Environment overrides:
#   BENCHPKG           path to the benchpkg binary (default: ~/.julia/bin/benchpkg)
#   BENCHPKGTABLE      path to the benchpkgtable binary (default: ~/.julia/bin/benchpkgtable)
#   BENCHPKG_TABLE_MODE  tables to print ("time", "memory", or "time,memory") default: time,memory
# ---------------------------------------------------------------------------
set -euo pipefail

BENCHPKG="${BENCHPKG:-$HOME/.julia/bin/benchpkg}"
BENCHPKGTABLE="${BENCHPKGTABLE:-$HOME/.julia/bin/benchpkgtable}"
TABLE_MODE="${BENCHPKG_TABLE_MODE:-time,memory}"

# Root of the GeodesicLM checkout (parent of this script's directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

orig_args=("$@")

# ---------------------------------------------------------------------------
# Pull out the values benchpkgtable also needs so we can re-print the tables
# from the same revision set / output directory.
# ---------------------------------------------------------------------------
pkg=""
rev=""
output_dir="."
path="$REPO_DIR"
path_given=false
script=""
script_given=false

i=0
while (( i < ${#orig_args[@]} )); do
    a="${orig_args[i]}"
    case "$a" in
        -r|--rev)
            rev="${orig_args[((i+1))]}"; i=$((i+2)) ;;
        --rev=*)
            rev="${a#*=}"; i=$((i+1)) ;;
        -o|--output-dir)
            output_dir="${orig_args[((i+1))]}"; i=$((i+2)) ;;
        --output-dir=*)
            output_dir="${a#*=}"; i=$((i+1)) ;;
        -s|--script)
            script="${orig_args[((i+1))]}"; script_given=true; i=$((i+2)) ;;
        --script=*)
            script="${a#*=}"; script_given=true; i=$((i+1)) ;;
        --path)
            path="${orig_args[((i+1))]}"; path_given=true; i=$((i+2)) ;;
        --path=*)
            path="${a#*=}"; path_given=true; i=$((i+1)) ;;
        *)
            if [[ "$a" != -* && -z "$pkg" ]]; then
                pkg="$a"
            fi
            i=$((i+1)) ;;
    esac
done

# Point benchpkg at this package's checkout unless the user gave a --path.
bench_args=("${orig_args[@]}")
if [[ "$path_given" == false ]]; then
    bench_args+=(--path "$path")
fi
# Use this repository's benchmark suite unless the user gave a --script.
if [[ "$script_given" == false ]]; then
    bench_args+=(--script "$SCRIPT_DIR/benchmarks.jl")
fi
# benchpkg does not create the output directory itself.
mkdir -p "$output_dir"

echo "[run_benchmark.sh] Running benchmarks (benchpkg) ..."
"$BENCHPKG" "${bench_args[@]}"

echo "[run_benchmark.sh] Printing results (benchpkgtable) ..."
table_args=()
[[ -n "$pkg" ]] && table_args+=("$pkg")
[[ -n "$rev" ]] && table_args+=(--rev "$rev")
# Always point the table tool at the package checkout, so it can resolve the
# package name from Project.toml (harmless even if a name was passed too).
table_args+=(--path "$path")
table_args+=(--input-dir "$output_dir")
table_args+=(--mode "$TABLE_MODE")
"$BENCHPKGTABLE" "${table_args[@]}"
