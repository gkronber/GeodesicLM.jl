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

    @testset "cholesky! (M3)" begin
        # helper: symmetric PD matrix
        pd(n) = begin
            B = reshape(_rnd(n * n), n, n)
            X = transpose(B) * B + n * Matrix(I, n, n)
            (X .+ transpose(X)) ./ 2   # force numerical symmetry
        end
        for n in (2, 3, 5, 10, 20)
            G = pd(n)
            A = copy(G)                 # f x f upper triangle is what we factor
            info = zeros(Int, 1)
            ok = KAOps.cholesky!(A, info)
            @test ok && info[1] == 0

            # Reconstruct U'U == G from the stored upper triangle.
            U = zeros(n, n)
            for i in 1:n, j in i:n
                U[i, j] = A[i, j]
            end
            @test isapprox(transpose(U) * U, G; atol=1e-8 * n)

            # Solve g*x = b and check residuals.
            b = _rnd(n)
            x = zeros(n)
            KAOps.solve_chol!(x, A, b)
            @test isapprox(G * x, b; atol=1e-8 * n)
        end

        # Indefinite matrix must fail (return false, info[1] == 1).
        Gind = [1.0 2.0; 2.0 1.0]
        Aind = copy(Gind)
        info = zeros(Int, 1)
        ok = KAOps.cholesky!(Aind, info)
        @test !ok
        @test info[1] == 1
    end

    @testset "GPUWorkspace (M4)" begin
        # direct constructor allocates a fresh workspace
        w = GeodesicLM.GPUWorkspace(3, 50, Float64, CPU())
        @test w.n == 3 && w.m == 50
        @test length(w.x) == 3 && length(w.fvec) == 50
        @test size(w.fjac) == (50, 3) && size(w.jtj) == (3, 3)
        @test length(w.scalar) == 1 && length(w.info) == 1

        # OncePerTask reuse: same (n, m, T, backend) key -> same buffers
        wc = GeodesicLM._get_gpu_workspace(3, 50, Float64, CPU())
        wc2 = GeodesicLM._get_gpu_workspace(3, 50, Float64, CPU())
        @test pointer(wc2.x) == pointer(wc.x)
        # distinct size -> distinct buffers
        w3 = GeodesicLM._get_gpu_workspace(5, 50, Float64, CPU())
        @test pointer(w3.x) != pointer(wc.x)
        @test w3.n == 5
        # distinct eltype -> distinct buffers
        w4 = GeodesicLM._get_gpu_workspace(3, 50, Float32, CPU())
        @test pointer(w4.x) != pointer(wc.x)
        @test eltype(w4.x) == Float32
    end

    @testset "GPUObjective + fd Jacobian/acceleration (M5)" begin
        t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        y = 3.0 .* exp.(-t ./ 2.0)
        m, n = length(t), 2

        @kernel function exp_fun!(x, fvec, t, y)
            i = @index(Global, Linear)
            if i <= length(y)
                a = x[1]
                tau = x[2]
                fvec[i] = y[i] - a * exp(-t[i] / tau)
            end
        end
        fun!(be, xx, fv, data) = begin
            t, y = data
            ev = exp_fun!(be)(xx, fv, t, y; ndrange = length(y))
        end

        @kernel function exp_grad!(x, fjac, t, y)
            i = @index(Global, Linear)
            if i <= length(y)
                a = x[1]; tau = x[2]
                e = exp(-t[i] / tau)
                fjac[i, 1] = -e
                fjac[i, 2] = -(a * t[i] / tau^2) * e
            end
        end
        grad!(be, xx, fj, data) = begin
            tty, _yy = data
            ev = exp_grad!(be)(xx, fj, tty, _yy; ndrange = length(tty))
        end

        data = (t, y)
        obj = GeodesicLM.GPUObjective(fun!; grad! = grad!, data = data)

        # On the CPU backend the device arrays are plain `Array`, so we can
        # write scalars directly.
        w = GeodesicLM._get_gpu_workspace(n, m, Float64, CPU())
        w.x[1] = 1.0; w.x[2] = 2.0   # a=1.0, tau=2.0

        # residuals on-device: r_i = y_i - exp(-t_i/2)
        fun!(CPU(), w.x, w.fvec, data)
        eRef = exp.(-t ./ 2.0)
        @test isapprox(w.fvec, y .- eRef; atol = 1e-10)

        # analytic Jacobian: d/da = -e, d/dtau = -(a*t/tau^2)e
        jref = hcat(-eRef, -(1.0 .* t ./ 4.0) .* eRef)
        GeodesicLM.jac!(w, obj)          # uses obj.grad!
        @test isapprox(w.fjac, jref; atol = 1e-10)

        # finite-difference Jacobian (central) matches the analytic one
        objfd = GeodesicLM.GPUObjective(fun!; data = data)
        wfd = GeodesicLM._get_gpu_workspace(n, m, Float64, CPU())
        wfd.x[1] = 1.0; wfd.x[2] = 2.0
        fun!(CPU(), wfd.x, wfd.fvec, data)
        GeodesicLM.jac!(wfd, objfd, true, 1.0e-5)
        @test isapprox(wfd.fjac, jref; atol = 1e-9)

        # fd acceleration along v (jac_uptodate path) is finite and consistent
        v = [1.0, 0.2]
        GeodesicLM.avv!(wfd, objfd, v, true, 1.0e-4)
        @test all(isfinite, wfd.acc)
    end

    @testset "on-device LM step vs CPU reference (M6)" begin
        # Toy residual with analytic grad!/avv!:  r_array[i] = x1^2*t[i] + x2
        t = [0.2, 0.5, 1.0, 2.0, 4.0]
        n, m = 2, length(t)

        @kernel function toy_fun!(x, fvec, t)
            i = @index(Global, Linear)
            if i <= length(t)
                fvec[i] = x[1]^2 * t[i] + x[2]
            end
        end
        fun!(be, xx, fv, data) = begin
            td = data
            ev = toy_fun!(be)(xx, fv, td; ndrange = length(td))
        end

        @kernel function toy_grad!(x, fj, t)
            i = @index(Global, Linear)
            if i <= length(t)
                fj[i, 1] = 2 * x[1] * t[i]
                fj[i, 2] = 1.0
            end
        end
        grad!(be, xx, fj, data) = begin
            td = data
            ev = toy_grad!(be)(xx, fj, td; ndrange = length(td))
        end

        @kernel function toy_avv!(x, v, acc, t)
            i = @index(Global, Linear)
            if i <= length(t)
                acc[i] = 2 * v[1]^2 * t[i]
            end
        end
        avv!(be, xx, vv, acc, data) = begin
            td = data
            ev = toy_avv!(be)(xx, vv, acc, td; ndrange = length(td))
        end

        data = t
        obj = GeodesicLM.GPUObjective(fun!; grad! = grad!, avv! = avv!, data = data)

        # Build a self-consistent snapshot on-device: x, fvec=f(x), fjac=J(x).
        ws = GeodesicLM._get_gpu_workspace(n, m, Float64, CPU())
        ws.x[1] = 1.5; ws.x[2] = -0.5
        GD = GeodesicLM
        GD.jac!(ws, obj)                 # fills ws.fjac (J at x)
        fun!(CPU(), ws.x, ws.fvec, data)
        lam = 2.0
        # dtd = diag(1, 2)
        ws.dtd .= [1.0 0.0; 0.0 2.0]
        C = 0.5 * dot(ws.fvec, ws.fvec)

        r = GD.gpu_step!(ws, obj, lam, C, C, 0, 1, 10.0, true, 1.0e-5)
        @test r.ok
        v = collect(ws.v); a = collect(ws.a)

        # ---- pure-host reference, mirroring gpu_step! line-for-line ----
        x = [1.5, -0.5]
        fvec = [x[1]^2 * tt + x[2] for tt in t]
        J = hcat([2 * x[1] * tt for tt in t], ones(m))
        dtd = [1.0 0.0; 0.0 2.0]
        Cref = 0.5 * dot(fvec, fvec)
        jtj = transpose(J) * J
        g = jtj + lam * dtd
        L = cholesky(Hermitian(g, :U))
        vref = L \ (transpose(J) * (-fvec))
        tmp1 = jtj * vref; tmp2 = dtd * vref
        temp1r = 0.5 * dot(vref, tmp1) / Cref
        temp2r = 0.5 * lam * dot(vref, tmp2) / Cref
        pred_red_ref = temp1r + 2 * temp2r
        jv = J * vref
        cos_ref = abs(dot(fvec, jv)) / (sqrt(dot(fvec, fvec)) * sqrt(dot(jv, jv)))
        acc = [2 * vref[1]^2 * tt for tt in t]
        aref = L \ (transpose(J) * (-acc))
        av_ref = sqrt(dot(aref, dtd * aref) / dot(vref, tmp2))
        x_new_ref = x + vref + 0.5 * aref
        fnew_ref = [x_new_ref[1]^2 * tt + x_new_ref[2] for tt in t]
        Cnew_ref = 0.5 * dot(fnew_ref, fnew_ref)

        @test isapprox(v, vref; atol = 1e-9)
        @test isapprox(a, aref; atol = 1e-9)
        @test r.pred_red ≈ pred_red_ref atol = 1e-9
        @test r.cos_alpha ≈ cos_ref atol = 1e-9
        @test r.av ≈ av_ref atol = 1e-9
        @test r.Cnew ≈ Cnew_ref atol = 1e-9
        # consistent with the new residual evaluated at x_new
        @test isapprox(ws.fvec_new, fnew_ref; atol = 1e-9)
        # downhill step is accepted
        @test r.accepted == 1
        @test r.dirder < 0
    end

end
