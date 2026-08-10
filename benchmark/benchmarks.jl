# -*- julia -*-
# benchmark/benchmarks.jl
# Benchmark suite for GeodesicLM.jl.
#
# This file defines the `SUITE` BenchmarkGroup used by AirspeedVelocity
# (`benchpkg`) to compare the runtime of GeodesicLM across git revisions.
#
# The cases below are a set of *central* benchmark problems intended to
# exercise the optimizer through different regimes:
#
#   * `quadratic`          - over-determined linear least squares (many residuals)
#   * `rosenbrock`         - classic hard 2-parameter nonlinear problem
#   * `exponential_decay`  - nonlinear model, over-determined residual set
#   * `sloppy_sum_exps`    - near-sloppy sum-of-exponentials model (degenerate dirs)
#
# For each problem we benchmark the default solver configuration (finite
# difference Jacobian + finite difference geodesic acceleration) as well as
# a configuration using analytic Jacobian/acceleration where available, since
# those are the two main usage modes.

using GeodesicLM
using LinearAlgebra
using BenchmarkTools

# ============================================================================
# Problem 1: Over-determined linear least squares
#   y = a + b*t  fit to noisy synthetic data (n = 2 params, m = 30 residuals)
# ============================================================================
const _lsq_t = collect(range(0.0, 10.0; length=30))
const _lsq_a, _lsq_b = 2.0, -0.5
const _lsq_y = _lsq_a .+ _lsq_b .* _lsq_t

function lsq_resid!(x, fvec)
    a, b = x[1], x[2]
    @inbounds for i in eachindex(_lsq_t)
        fvec[i] = _lsq_y[i] - (a + b * _lsq_t[i])
    end
    nothing
end

function lsq_jac!(x, fjac)
    fill!(fjac, 0.0)
    @inbounds for i in eachindex(_lsq_t)
        fjac[i, 1] = -1.0
        fjac[i, 2] = -_lsq_t[i]
    end
    nothing
end

const _lsq_m = length(_lsq_y)
const _lsq_n = 2

# ============================================================================
# Problem 2: Rosenbrock function (n = 2, m = 2)
# ============================================================================
function rosenbrock!(x, fvec)
    fvec[1] = 10.0 * (x[2] - x[1]^2)
    fvec[2] = 1.0 - x[1]
    nothing
end

function rosenbrock_jac!(x, fjac)
    fjac[1, 1] = -20.0 * x[1]
    fjac[1, 2] = 10.0
    fjac[2, 1] = -1.0
    fjac[2, 2] = 0.0
    nothing
end

function rosenbrock_avv!(x, v, acc)
    acc[1] = -20.0 * v[1]^2
    acc[2] = 0.0
    nothing
end

const _rb_m = 2
const _rb_n = 2

# ============================================================================
# Problem 3: Exponential decay fit
#   y = a*exp(-t/τ)  (n = 2 params, m = 50 residuals)
# ============================================================================
const _exp_t = collect(range(0.0, 10.0; length=50))
const _exp_a_true, _exp_tau_true = 3.0, 2.0
const _exp_y = _exp_a_true .* exp.(-_exp_t ./ _exp_tau_true)

function exp_resid!(x, fvec)
    a, tau = x[1], x[2]
    @inbounds for i in eachindex(_exp_t)
        fvec[i] = _exp_y[i] - a * exp(-_exp_t[i] / tau)
    end
    nothing
end

function exp_jac!(x, fjac)
    a, tau = x[1], x[2]
    @inbounds for i in eachindex(_exp_t)
        t = _exp_t[i]
        e = exp(-t / tau)
        fjac[i, 1] = -e
        fjac[i, 2] = -(a * t / tau^2) * e
    end
    nothing
end

const _exp_m = length(_exp_y)
const _exp_n = 2

# ============================================================================
# Problem 4: Sum of two exponentials + offset (near-sloppy model)
#   y = a1*exp(-t/τ1) + a2*exp(-t/τ2) + c   (n = 5 params, m = 80 residuals)
# ============================================================================
const _sum_t = collect(range(0.0, 20.0; length=80))
const _sum_params = [2.0, 1.0, 1.5, 5.0, 0.1]   # a1, τ1, a2, τ2, c
const _sum_y = _sum_params[1] .* exp.(-_sum_t ./ _sum_params[2]) .+
                _sum_params[3] .* exp.(-_sum_t ./ _sum_params[4]) .+ _sum_params[5]

function sum_exp_resid!(x, fvec)
    a1, τ1, a2, τ2, c = x
    @inbounds for i in eachindex(_sum_t)
        t = _sum_t[i]
        fvec[i] = _sum_y[i] - (a1 * exp(-t / τ1) + a2 * exp(-t / τ2) + c)
    end
    nothing
end

function sum_exp_jac!(x, fjac)
    a1, τ1, a2, τ2, c = x
    @inbounds for i in eachindex(_sum_t)
        t = _sum_t[i]
        e1 = exp(-t / τ1)
        e2 = exp(-t / τ2)
        fjac[i, 1] = -e1
        fjac[i, 2] = -(a1 * t / τ1^2) * e1
        fjac[i, 3] = -e2
        fjac[i, 4] = -(a2 * t / τ2^2) * e2
        fjac[i, 5] = -1.0
    end
    nothing
end

const _sum_m = length(_sum_y)
const _sum_n = 5

# ============================================================================
# Benchmark suite
# ============================================================================

const SUITE = BenchmarkGroup()

# --- Problem 1: linear least squares --------------------------------------
SUITE["quadratic"] = BenchmarkGroup()
SUITE["quadratic"]["fd"] = @benchmarkable begin
    x = [0.0, 0.0]
    fvec = zeros($_lsq_m)
    geodesiclm(lsq_resid!, nothing, nothing;
               x=x, fvec=fvec, n=$_lsq_n, m=$_lsq_m, maxiter=50)
end samples=50 evals=1

SUITE["quadratic"]["analytic"] = @benchmarkable begin
    x = [0.0, 0.0]
    fvec = zeros($_lsq_m)
    geodesiclm(lsq_resid!, lsq_jac!, nothing;
               x=x, fvec=fvec, n=$_lsq_n, m=$_lsq_m,
               analytic_jac=true, maxiter=50)
end samples=50 evals=1

# --- Problem 2: Rosenbrock -------------------------------------------------
SUITE["rosenbrock"] = BenchmarkGroup()
SUITE["rosenbrock"]["fd"] = @benchmarkable begin
    x = [-1.2, 1.0]
    fvec = zeros($_rb_m)
    geodesiclm(rosenbrock!, nothing, nothing;
               x=x, fvec=fvec, n=$_rb_n, m=$_rb_m, maxiter=500)
end samples=30 evals=1

SUITE["rosenbrock"]["analytic"] = @benchmarkable begin
    x = [-1.2, 1.0]
    fvec = zeros($_rb_m)
    geodesiclm(rosenbrock!, rosenbrock_jac!, rosenbrock_avv!;
               x=x, fvec=fvec, n=$_rb_n, m=$_rb_m,
               analytic_jac=true, analytic_Avv=true, maxiter=500)
end samples=30 evals=1

SUITE["rosenbrock"]["no_accel"] = @benchmarkable begin
    x = [-1.2, 1.0]
    fvec = zeros($_rb_m)
    geodesiclm(rosenbrock!, nothing, nothing;
               x=x, fvec=fvec, n=$_rb_n, m=$_rb_m, iaccel=0, maxiter=500)
end samples=30 evals=1

# --- Problem 3: exponential decay ----------------------------------------
SUITE["exponential_decay"] = BenchmarkGroup()
SUITE["exponential_decay"]["fd"] = @benchmarkable begin
    x = [1.0, 1.0]
    fvec = zeros($_exp_m)
    geodesiclm(exp_resid!, nothing, nothing;
               x=x, fvec=fvec, n=$_exp_n, m=$_exp_m, maxiter=100)
end samples=50 evals=1

SUITE["exponential_decay"]["analytic"] = @benchmarkable begin
    x = [1.0, 1.0]
    fvec = zeros($_exp_m)
    geodesiclm(exp_resid!, exp_jac!, nothing;
               x=x, fvec=fvec, n=$_exp_n, m=$_exp_m,
               analytic_jac=true, maxiter=100)
end samples=50 evals=1

# --- Problem 4: sum of exponentials (near-sloppy) --------------------------
SUITE["sloppy_sum_exps"] = BenchmarkGroup()
SUITE["sloppy_sum_exps"]["fd"] = @benchmarkable begin
    x = [1.5, 1.0, 2.5, 4.0, 0.2]
    fvec = zeros($_sum_m)
    geodesiclm(sum_exp_resid!, nothing, nothing;
               x=x, fvec=fvec, n=$_sum_n, m=$_sum_m, maxiter=200)
end samples=30 evals=1

SUITE["sloppy_sum_exps"]["analytic"] = @benchmarkable begin
    x = [1.5, 1.0, 2.5, 4.0, 0.2]
    fvec = zeros($_sum_m)
    geodesiclm(sum_exp_resid!, sum_exp_jac!, nothing;
               x=x, fvec=fvec, n=$_sum_n, m=$_sum_m,
               analytic_jac=true, maxiter=200)
end samples=30 evals=1
