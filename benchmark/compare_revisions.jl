#!/usr/bin/env julia
# -*- julia -*-
# benchmark/compare_revisions.jl
# Compare GeodesicLM.jl benchmark results across git revisions, locally,
# without needing AirspeedVelocity.jl (benchpkg).
#
# Usage:
#   julia benchmark/compare_revisions.jl [rev1 [rev2 ...]]
#
# Examples:
#   julia benchmark/compare_revisions.jl HEAD~1 dirty        # prev. commit vs current tree
#   julia benchmark/compare_revisions.jl dirty HEAD          # current tree vs HEAD
#   julia benchmark/compare_revisions.jl origin/main HEAD~3
#   julia benchmark/compare_revisions.jl                     # defaults to "HEAD~1 dirty"
#
# Semantics:
#   * The special revision "dirty" benchmarks the current working tree,
#     including any uncommitted changes, exactly like `benchpkg --rev dirty`.
#   * Every other revision is benchmarked from a *temporary* `git worktree`,
#     so your current checkout, stash, and branch are never touched.
#   * Each revision is benchmarked in its own fresh Julia subprocess to
#     guarantee the matching version of GeodesicLM is loaded and compiled.
#   * Median timings (and memory) are collected and printed as a table where
#     the first listed revision is the baseline (ratio = rev / baseline).
#
# The benchmark suite itself is the same one used by AirspeedVelocity:
# benchmark/benchmarks.jl (defines `SUITE`).

using Printf

const SCRIPT          = @__FILE__
const REPO            = normpath(joinpath(dirname(SCRIPT), ".."))
const BENCHDIR        = joinpath(REPO, "benchmark")
const DEFAULT_REVS    = ["HEAD~1", "dirty"]

# ---------------------------------------------------------------------------
# Small runner program executed inside a subprocess. It loads the GeodesicLM
# that sits *above* the given benchmark dir (a worktree for a specific
# revision, or the repo itself for "dirty"), runs the suite, and writes one
# tab-separated row per benchmark case to the output file.
# ---------------------------------------------------------------------------
const RUNNER = raw"""
using Pkg
Pkg.instantiate()
using BenchmarkTools

dir = ARGS[1]
out = ARGS[2]

include(joinpath(dir, "benchmark", "benchmarks.jl"))  # defines SUITE

results = try
    run(SUITE; verbose=false)
catch err
    @warn "benchmark run failed" err = err
    nothing
end

open(out, "w") do io
    results === nothing && return
    med(v) = isempty(v) ? 0 : sort(v)[cld(length(v), 2)]
    for (g, grp) in results
        for (c, est) in grp
            # Leaf is either a Trial (raw samples) or TrialEstimate (medians).
            isest = est isa BenchmarkTools.TrialEstimate
            t = isest ? [est.time]   : est.times
            g_ = isest ? [est.gctime] : est.gctimes
            m = isest ? est.memory  : est.memory
            a = isest ? est.allocs  : est.allocs
            mt = isest ? Int(est.time)   : Int(med(t))
            mg = isest ? Int(est.gctime) : Int(med(g_))
            mm = isest ? Int(round(est.memory)) : (m isa AbstractVector ? Int(round(med(m))) : Int(round(m)))
            ma = isest ? Int(est.allocs) : (a isa AbstractVector ? Int(round(med(a))) : Int(round(a)))
            println(io, string(g), '\t', string(c), '\t',
                    mt, '\t', mg, '\t', mm, '\t', ma)
        end
    end
end
"""

# ---------------------------------------------------------------------------
# Resolve a revision string to path info. "dirty" maps to the current tree;
# anything else gets a fresh detached worktree.
function prepare_revision(rev::String)
    if rev == "dirty"
        return (dir = REPO, wt = nothing, label = rev)
    end
    wt = mktempdir()
    success(`git -C $REPO worktree add --detach $wt $rev`) ||
        error("could not check out revision '$rev'")
    return (dir = wt, wt = wt, label = rev)
end

function cleanup_revision(info)
    if info.wt !== nothing
        try
            run(`git -C $REPO worktree remove --force $(info.wt)`)
        catch
            @warn "could not remove worktree $(info.wt)"
        end
        try rm(info.wt; recursive=true, force=true) catch end
    end
    nothing
end

# Run one revision -> Vector of case Dicts.
function run_revision(info, outfile)
    dir = info.dir
    if !isdir(joinpath(dir, "benchmark"))
        @warn "skipping $(info.label): no benchmark/ directory in this revision"
        return Dict{String,Any}()
    end
    julia = Base.julia_cmd()
    proj = joinpath(dir, "benchmark")
    cmd = `$julia --startup-file=no --project=$proj -e $RUNNER $dir $outfile`
    ok = success(pipeline(cmd, stdout=devnull, stderr=devnull))
    ok || @warn "benchmarker failed for $(info.label)"
    isfile(outfile) || return Dict{String,Any}()
    results = Dict{String,Any}()
    for line in eachline(outfile)
        isempty(strip(line)) && continue
        fields = split(line, '\t')
        g, c = fields[1], fields[2]
        results["$g / $c"] = Dict(
            "group"   => g, "case"    => c,
            "time"    => Int(round(parse(Float64, fields[3]))),
            "gctime"  => Int(round(parse(Float64, fields[4]))),
            "memory"  => Int(round(parse(Float64, fields[5]))),
            "allocs"  => Int(parse(Float64, fields[6])),
        )
    end
    results
end

# ---------------------------------------------------------------------------
function fmt(v::Float64)
    v >= 1e9 && return @sprintf("%10.2f s", v / 1e9)
    v >= 1e6 && return @sprintf("%10.2f ms", v / 1e6)
    v >= 1e3 && return @sprintf("%10.2f µs", v / 1e3)
    return @sprintf("%10.2f ns", v)
end

function main()
    revs = isempty(ARGS) ? DEFAULT_REVS : collect(ARGS)

    isdir(BENCHDIR) || error("benchmark directory not found at $BENCHDIR")

    mktempdir() do tmp
        all_res = Dict{String,Any}()
        for (i, rev) in enumerate(revs)
            info = prepare_revision(rev)
            try
                @info "benchmarking revision" rev = rev i = i n = length(revs)
                outfile = joinpath(tmp, "rev_$(i)_$(replace(rev, "/" => "_")).json")
                all_res[rev] = run_revision(info, outfile)
            finally
                cleanup_revision(info)
            end
        end

        # ---- print comparison table ----
        baseline = revs[1]
        base_res = all_res[baseline]

        # All case keys seen across every revision, kept in a stable order.
        seen = Set{String}()
        for r in revs
            foreach(k -> push!(seen, k), keys(all_res[r]))
        end
        casekeys = sort!(collect(seen); by=s -> (split(s, " / ")[1], split(s, " / ")[2]))

        println()
        println("Benchmark comparison for GeodesicLM")
        println("  baseline: $(baseline)")
        for r in revs[2:end]
            println("  compared: $(r)   (ratio = $(r) / $(baseline))")
        end
        println()
        hdr = "case" * " "^36 * "baseline       "
        for r in revs[2:end]
            hdr *= @sprintf("%-20s", r)
        end
        println(hdr * "   memory")
        println("-" ^ (50 + 20 * length(revs)))

        for key in casekeys
            b    = get(base_res, key, nothing)
            bhas = b !== nothing && haskey(b, "time")
            btime = bhas ? Float64(b["time"]) : NaN
            bmem  = bhas ? Float64(get(b, "memory", 0)) : 0.0

            col = bhas ? fmt(btime) : rpad("n/a", 11)
            for r in revs[2:end]
                rr   = get(all_res[r], key, nothing)
                rhas = rr !== nothing && haskey(rr, "time")
                if !rhas
                    col *= rpad("n/a", 20)
                elseif !bhas
                    col *= @sprintf("%-20s", fmt(Float64(rr["time"])))
                else
                    ratio = Float64(rr["time"]) / btime
                    str = ratio >= 1000 ? ">" : string(round(ratio; digits=2))
                    col *= @sprintf("%-10s %-9s", fmt(Float64(rr["time"])), str)
                end
            end
            println(rpad(key, 48) * col * (@sprintf("%9.2f KiB", bmem / 2^10)))
        end
        println()
        println("Times are median run times. Ratio = revision/baseline.")
    end
end

main()
