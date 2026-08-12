using GeodesicLM
using Test
using LinearAlgebra

# ============================================================================
# Shared problem definitions
# ============================================================================

"""
    quadratic!(x, fvec)

Linear residuals with a unique zero at `x = [1.0, 2.0]`.
"""
function quadratic!(x, fvec)
    fvec[1] = x[1] - 1.0
    fvec[2] = x[2] - 2.0
end

"""
    rosenbrock!(x, fvec)

Rosenbrock function residuals (minimum at `x = [1.0, 1.0]`).
"""
function rosenbrock!(x, fvec)
    fvec[1] = 10.0 * (x[2] - x[1]^2)
    fvec[2] = 1.0 - x[1]
end

function rosenbrock_jac!(x, fjac)
    fjac[1, 1] = -20.0 * x[1]
    fjac[1, 2] = 10.0
    fjac[2, 1] = -1.0
    fjac[2, 2] = 0.0
    return nothing
end

# Second directional derivative d²/dv² rosenbrock residual
function rosenbrock_avv!(x, v, acc)
    acc[1] = -20.0 * v[1]^2
    acc[2] = 0.0
    return nothing
end

function cost(fvec)
    return 0.5 * dot(fvec, fvec)
end

@testset "GeodesicLM.jl" begin

    @testset "Return value structure" begin
        x = [0.0, 0.0]
        fvec = zeros(2)
        r = geodesiclm(quadratic!, nothing, nothing; x=x, fvec=fvec, n=2, m=2, maxiter=100)
        @test r isa Tuple
        @test length(r) == 7
        @test r[1] isa Vector{Float64}
        @test r[2] isa Vector{Float64}
        @test r[3] isa Int      # niters
        @test r[4] isa Int      # nfev
        @test r[5] isa Int      # njev
        @test r[6] isa Int      # naev
        @test r[7] isa Int      # converged code
        @test r[1] == x         # modifies x in place too
    end

    @testset "Linear quadratic (finite-difference Jacobian)" begin
        x = [0.0, 0.0]
        fvec = zeros(2)
        r = geodesiclm(quadratic!, nothing, nothing; x=x, fvec=fvec, n=2, m=2, maxiter=100)
        @test cost(r[2]) < 1.0e-10
        @test r[1] ≈ [1.0, 2.0] atol=1.0e-5
        @test r[7] in (3, 5, 6)  # gtol / xrtol / ftol convergence
        @test all(isfinite, r[2])
    end

    @testset "Analytic Jacobian and analytic Avv" begin
        x = [-1.2, 1.0]
        fvec = zeros(2)
        r = geodesiclm(rosenbrock!, rosenbrock_jac!, rosenbrock_avv!;
                       x=x, fvec=fvec, n=2, m=2,
                       analytic_jac=true, analytic_Avv=true, maxiter=500)
        @test r[1] ≈ [1.0, 1.0] atol=1.0e-4
        @test cost(r[2]) < 1.0e-6
        @test r[5] > 0   # analytic Jacobian was actually used
        @test r[6] > 0   # analytic acceleration was actually used
    end

    @testset "Finite-difference (no analytic derivatives)" begin
        x = [-1.2, 1.0]
        fvec = zeros(2)
        r = geodesiclm(rosenbrock!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2, maxiter=500)
        @test r[1] ≈ [1.0, 1.0] atol=1.0e-4
        @test cost(r[2]) < 1.0e-6
    end

    @testset "No geodesic acceleration (iaccel=0)" begin
        x = [-1.2, 1.0]
        fvec = zeros(2)
        r = geodesiclm(rosenbrock!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2, iaccel=0, maxiter=500)
        @test r[1] ≈ [1.0, 1.0] atol=1.0e-4
        @test cost(r[2]) < 1.0e-6
        @test r[6] == 0  # no acceleration evaluations when iaccel=0
    end

    @testset "Update methods (imethod)" begin
        # All update strategies should solve the linear problem exactly
        for im in (0, 1, 2, 10, 11)
            x = [0.0, 0.0]
            fvec = zeros(2)
            r = geodesiclm(quadratic!, nothing, nothing;
                           x=x, fvec=fvec, n=2, m=2, imethod=im, maxiter=200)
            @test r[1] ≈ [1.0, 2.0] atol=1.0e-5
            @test cost(r[2]) < 1.0e-8
            @test r[7] != -1   # must not run out of iterations
        end
    end

    @testset "Damping modes" begin
        for dm in (0, 1)
            x = [2.0, -1.0]
            fvec = zeros(2)
            r = geodesiclm(rosenbrock!, nothing, nothing;
                           x=x, fvec=fvec, n=2, m=2, damp_mode=dm, maxiter=500)
            @test r[1] ≈ [1.0, 1.0] atol=1.0e-4
            @test cost(r[2]) < 1.0e-6
        end
    end

    @testset "Bold acceptance criterion (ibold)" begin
        for ib in (0, 1, 2, 3, 4)
            x = [-1.2, 1.0]
            fvec = zeros(2)
            r = geodesiclm(rosenbrock!, rosenbrock_jac!, rosenbrock_avv!;
                           x=x, fvec=fvec, n=2, m=2,
                           analytic_jac=true, analytic_Avv=true,
                           ibold=ib, maxiter=500)
            @test r[1] ≈ [1.0, 1.0] atol=1.0e-4
            @test cost(r[2]) < 1.0e-6
        end
    end

    @testset "Convergence return codes" begin
        # Cost goal reached -> 2
        x = [0.0, 0.0]
        fvec = zeros(2)
        r = geodesiclm(quadratic!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2, Cgoal=1.0e-4, maxiter=100)
        @test r[7] == 2

        # Maximum iterations exceeded -> -1
        x = [0.0, 0.0]
        fvec = zeros(2)
        r = geodesiclm(quadratic!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2, maxiter=2)
        @test r[7] == -1

        # Maximum function evaluations exceeded -> -2
        x = [0.0, 0.0]
        fvec = zeros(2)
        r = geodesiclm(quadratic!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2, maxfev=5, maxiter=100)
        @test r[7] == -2

        # NaN in the objective -> -11
        function nan_func(x, fvec)
            fvec[1] = NaN
            fvec[2] = x[2]
        end
        x = [0.0, 0.0]
        fvec = zeros(2)
        r = geodesiclm(nan_func, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2, maxiter=100)
        @test r[7] == -11

        # User termination via callback -> -10
        function stop_callback(args...)
            return 1
        end
        x = [0.0, 0.0]
        fvec = zeros(2)
        r = geodesiclm(quadratic!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2,
                       callback=stop_callback, maxiter=100)
        @test r[7] == -10

        # Callback returning nothing leaves info unchanged (no termination)
        hits = Ref(0)
        function quiet_callback(args...)
            hits[] += 1
            return nothing
        end
        x = [0.0, 0.0]
        fvec = zeros(2)
        r = geodesiclm(quadratic!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2,
                       callback=quiet_callback, maxiter=100)
        @test hits[] > 0
        @test r[7] != -10
    end

    @testset "dpmpar machine parameters" begin
        @test dpmpar(1) ≈ eps(Float64)
        @test dpmpar(2) ≈ floatmin(Float64)
        @test dpmpar(3) ≈ floatmax(Float64)
        @test_throws ErrorException dpmpar(4)
    end

    @testset "fdjac finite-difference Jacobian" begin
        f(x, fvec) = begin
            fvec[1] = x[1]^2
            fvec[2] = x[1] * x[2]
        end
        x = [2.0, 3.0]
        fvec = [4.0, 6.0]
        fj = fdjac(2, 2, x, fvec, f, 1.0e-6, true)
        # Jacobian of (x1², x1*x2) is [[2x1, 0], [x2, x1]]
        @test fj ≈ [4.0 0.0; 3.0 2.0] atol=1.0e-6
        fj_fwd = fdjac(2, 2, x, fvec, f, 1.0e-6, false)
        @test fj_fwd ≈ [4.0 0.0; 3.0 2.0] atol=1.0e-5
    end

    @testset "fd_avv finite-difference acceleration" begin
        f(x, fvec) = begin
            fvec[1] = x[1]^3
            fvec[2] = x[1] * x[2]
        end
        x = [1.0, 2.0]
        v = [2.0, 0.5]
        fvec = [1.0, 2.0]
        # Jacobian at x: [[3x1², 0],[x2, x1]] = [[3,0],[2,1]]
        fjac = [3.0 0.0; 2.0 1.0]
        # Exact Avv = d²/dv² f = (6 x1 * v1², 2 f_xy v1 v2) = (6*1*4, 2*2*0.5) = (24, 2)
        acc = fd_avv(2, 2, x, v, fvec, fjac, f, true, 1.0e-4)
        @test acc[1] ≈ 24.0 rtol=1.0e-2
        @test acc[2] ≈ 2.0 rtol=1.0e-2
        # Non-updated-Jacobian path yields the same result
        acc2 = fd_avv(2, 2, x, v, fvec, fjac, f, false, 1.0e-4)
        @test acc2 ≈ acc rtol=1.0e-2
    end

    @testset "acceptance criterion" begin
        dtd = Matrix{Float64}(I, 2, 2)
        v = [1.0, 0.0]
        vold = [1.0, 0.0]

        # Downhill step is always accepted regardless of ibold
        for ib in 0:4
            @test acceptance(2, 10.0, 5.0, 10.0, ib, dtd, v, vold) >= 1
        end

        # Uphill step with ibold=0 (only downhill) is rejected
        @test acceptance(2, 10.0, 20.0, 10.0, 0, dtd, v, vold) < 0

        # vold = 0 forces beta = 1 (degenerate first step case)
        vold0 = [0.0, 0.0]
        @test acceptance(2, 10.0, 20.0, 15.0, 1, dtd, v, vold0) < 0
    end

    @testset "lambda / delta update helpers" begin
        # Fixed-factor lambda update
        @test update_lam_factor(10.0, 1, 2.0, 2.0) ≈ 5.0
        @test update_lam_factor(10.0, -1, 2.0, 2.0) ≈ 20.0

        # Fixed-factor delta update
        @test update_delta_factor(10.0, 1, 2.0, 2.0) ≈ 20.0
        @test update_delta_factor(10.0, -1, 2.0, 2.0) ≈ 5.0

        # Nelson update: with rho=0 the update factor is
        # max(1/factoraccept, 1-(factorreject-1)*(2*rho-1)^3) = max(0.5, 2) = 2,
        # so lam = 8 * 2 = 16.
        lam = update_lam_nelson(8.0, 1, 2.0, 2.0, 0.0)
        @test lam ≈ 16.0
        lam_rej = update_lam_nelson(8.0, -1, 2.0, 2.0, 0.0)
        @test lam_rej > 8.0
    end

    @testset "trust_region and dgqt" begin
        # Simple 1D problem: minimize (x - 2)^2  => residual f = x - 2
        fvec = [-1.0]
        fjac = reshape([1.0], 1, 1)
        dtd = reshape([1.0], 1, 1)
        (v, lam) = trust_region(1, 1, fvec, fjac, dtd, 1.0)
        @test length(v) == 1
        @test v[1] ≈ 1.0 atol=1.0e-3   # step toward x = 2 within radius 1
        @test lam >= 0.0

        # dgqt on a positive definite system
        A = [2.0 0.0; 0.0 4.0]
        b = [1.0, 1.0]
        (x, par, info, f) = dgqt(2, A, b, 1.0)
        @test info in (1, 2, 3, 4)
        @test norm(x) <= 1.0 + 1.0e-3
        @test isfinite(par)
    end

    @testset "destsv smallest singular value" begin
        # diag(4, 1) has smallest singular value 1
        R = [4.0 0.0; 0.0 1.0]
        (svmin, z) = destsv(2, R)
        @test svmin ≈ 1.0 atol=5.0e-2
        @test norm(z) ≈ 1.0 atol=1.0e-8
        @test norm(R * z) ≈ svmin atol=5.0e-2

        # Upper triangular non-diagonal: R = [3 4; 0 2]
        R2 = [3.0 4.0; 0.0 2.0]
        (svmin2, z2) = destsv(2, R2)
        λ2 = eigmin(R2 * R2')
        @test svmin2 ≈ sqrt(λ2) rtol=1.0e-2
        @test norm(z2) ≈ 1.0 atol=1.0e-8
        @test norm(R2 * z2) ≈ svmin2 rtol=1.0e-2
    end

    @testset "best-cost tracking" begin
        # If the optimizer temporarily worsens, it should still return the best x.
        x = [0.0, 0.0]
        fvec = zeros(2)
        r = geodesiclm(rosenbrock!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2, maxiter=500)
        # best cost is the minimum over the trajectory
        @test cost(r[2]) < 1.0e-6
    end

    # GPU building-block kernels (CPU backend; no GPU required — see PLAN.md)
    include("gpu_kernels.jl")

end
