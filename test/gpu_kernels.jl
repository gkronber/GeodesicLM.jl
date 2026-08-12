# -*- julia -*-
# test/gpu_kernels.jl
# Unit tests for GeodesicLM.KAOps: the backend-agnostic GPU building-block
# kernels. These run on the CPU backend (KernelAbstractions' `CPU()`), so they
# require no GPU and validate the kernels that will later be executed on CUDA /
# Metal by simply passing those backends. See PLAN.md (milestones M0-M2).

using Test
using LinearAlgebra

using KernelAbstractions
using GeodesicLM.KAOps

# deterministic pseudo-random test data (no Random dependency)
_rnd(len) = [sin(0.131 * i) + 0.7 * cos(0.311 * i) for i in 1:len]

# ---------------------------------------------------------------------------
# Test front-end : run every kernel comparison for a list of backends.
# By default only the CPU backend (no GPU required). Extend `test_backends`
# to also include real GPU backends (e.g. `CUDA()`) when available in CI.
# ---------------------------------------------------------------------------
function test_backends()
    b = KAOps.backend(zeros(2))          # CPU() unless a GPU array is passed
    return b isa Vector{Any} ? b : [b]
end

@testset "KAOps building blocks" begin

    @testset "backend helpers" begin
        @test KAOps.backend(zeros(3)) isa KernelAbstractions.CPU
        @test KAOps.backend(CPU()) === CPU()
        @test KAOps.backend_of(Array{Float64,1}) === CPU()
        @test KAOps.alloc(Float64, CPU(), 4) isa Vector{Float64}
        @test length(KAOps.alloc(Float64, CPU(), 2, 3)) == 6
    end

    @testset "elementwise ops" begin
        m = 512
        x = _rnd(m)
        y = _rnd(m)
        # fill!
        z = zeros(m)
        KAOps.fill!(z, 3.0)
        @test all(z .== 3.0)
        # copyto!
        c = zeros(m)
        KAOps.copyto!(c, x)
        @test c == x
        # axpy!  y = y + a*x
        ya = copy(y)
        a = 2.5
        KAOps.axpy!(ya, a, x)
        @test isapprox(ya, y .+ a .* x; atol=1e-10)
        # scale!  y = a*x
        ys = zeros(m)
        KAOps.scale!(ys, a, x)
        @test isapprox(ys, a .* x; atol=1e-10)
    end

    @testset "matrix-vector" begin
        m, n = 200, 5
        A = reshape(_rnd(m*n), m, n)
        x = _rnd(n)
        y = zeros(m)
        KAOps.mul!(y, A, x)
        @test isapprox(y, A * x; atol=1e-10)

        xv = _rnd(m)
        yt = zeros(n)
        KAOps.mul!(yt, transpose(A), xv)
        @test isapprox(yt, transpose(A) * xv; atol=1e-10)
    end

    @testset "AtA! (J'J)" begin
        m, n = 150, 4
        A = reshape(_rnd(m*n), m, n)
        B = reshape(_rnd(m*n), m, n)
        C = zeros(n, n)
        KAOps.AtA!(C, A, B)
        @test isapprox(C, transpose(A) * B; atol=1e-10)
    end

    @testset "dot!/norm2 reductions" begin
        for len in (1, 2, 7, 63, 64, 100, 511, 1024, 10_000)
            a = _rnd(len)
            b = _rnd(len)
            o = zeros(1)
            KAOps.dot!(o, a, b)
            @test isapprox(o[1], dot(a, b); atol=1e-8 * sqrt(len))
            o2 = zeros(1)
            KAOps.norm2(o2, a)
            @test isapprox(o2[1], dot(a, a); atol=1e-8 * sqrt(len))
        end
    end

    @testset "nanflag" begin
        @test !KAOps.nanflag([1.0, 2.0])
        @test KAOps.nanflag([1.0, NaN])
        @test KAOps.nanflag([1.0, Inf])
    end

end
