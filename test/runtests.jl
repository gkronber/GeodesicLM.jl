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
        @test r[7] in (2, 3, 4, 6)  # Cgoal / gtol / xtol / ftol convergence
        @test all(isfinite, r[2])
    end

    @testset "Analytic Jacobian and analytic Avv" begin
        x = [-1.2, 1.0]
        fvec = zeros(2)
        r = geodesiclm(rosenbrock!, rosenbrock_jac!, rosenbrock_avv!;
                       x=x, fvec=fvec, n=2, m=2,
                       analytic_jac=true, analytic_Avv=true, maxiter=500)
        # the reference defaults stop at `Cgoal`, so on a zero-residual problem
        # the minimizer is only accurate to about sqrt(Cgoal)
        @test r[1] ≈ [1.0, 1.0] atol=1.0e-3
        @test cost(r[2]) < 1.0e-6
        @test r[5] > 0   # analytic Jacobian was actually used
        @test r[6] > 0   # analytic acceleration was actually used
    end

    @testset "Finite-difference (no analytic derivatives)" begin
        x = [-1.2, 1.0]
        fvec = zeros(2)
        r = geodesiclm(rosenbrock!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2, maxiter=500)
        # the reference defaults stop at `Cgoal`, so on a zero-residual problem
        # the minimizer is only accurate to about sqrt(Cgoal)
        @test r[1] ≈ [1.0, 1.0] atol=1.0e-3
        @test cost(r[2]) < 1.0e-6
    end

    @testset "No geodesic acceleration (iaccel=0)" begin
        x = [-1.2, 1.0]
        fvec = zeros(2)
        r = geodesiclm(rosenbrock!, nothing, nothing;
                       x=x, fvec=fvec, n=2, m=2, iaccel=0, maxiter=500)
        # the reference defaults stop at `Cgoal`, so on a zero-residual problem
        # the minimizer is only accurate to about sqrt(Cgoal)
        @test r[1] ≈ [1.0, 1.0] atol=1.0e-3
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
            @test r[1] ≈ [1.0, 1.0] atol=1.0e-3
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
            @test r[1] ≈ [1.0, 1.0] atol=1.0e-3
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

    @testset "Float32 (parametric element type)" begin
        x = Float32[-1.2, 1.0]
        fvec = zeros(Float32, 2)
        r = geodesiclm(rosenbrock!, rosenbrock_jac!, nothing;
                       x=x, fvec=fvec, n=2, m=2, analytic_jac=true, maxiter=500)
        @test r[1] isa Vector{Float32}
        @test r[2] isa Vector{Float32}
        # `Cgoal` also scales with the precision, so Float32 stops earlier still
        @test r[1] ≈ Float32[1.0, 1.0] atol=1.0f-2
        @test cost(r[2]) < default_tolerance(Float32)

        # finite differences too: the default step scales with the precision
        x = Float32[-1.2, 1.0]
        fvec = zeros(Float32, 2)
        r = geodesiclm(rosenbrock!, nothing, nothing; x=x, fvec=fvec, n=2, m=2, maxiter=500)
        @test r[1] ≈ Float32[1.0, 1.0] atol=1.0f-1
        @test cost(r[2]) < default_tolerance(Float32)
    end

    @testset "tight convergence with Cgoal disabled" begin
        # With the absolute cost goal switched off, a zero-residual problem is
        # driven to machine precision (gtol convergence).
        x = [-1.2, 1.0]
        fvec = zeros(2)
        r = geodesiclm(rosenbrock!, rosenbrock_jac!, nothing;
                       x=x, fvec=fvec, n=2, m=2, analytic_jac=true,
                       Cgoal=0.0, maxiter=500)
        @test r[1] ≈ [1.0, 1.0] atol=1.0e-7
        @test cost(r[2]) < 1.0e-15
        @test r[7] == 3   # gtol
    end

    @testset "precision-dependent defaults" begin
        # The reference interface hard-codes 1.49012e-8 == sqrt(eps(Float64))
        # for the forward-difference step and for the absolute tolerances.
        @test default_fd_step(Float64) ≈ 1.49012e-8 rtol=1e-5
        @test default_fd_step(Float64) == sqrt(eps(Float64))
        @test default_fd_step(Float64, true) == cbrt(eps(Float64))  # central diff
        @test default_fd_step(Float32) isa Float32
        @test default_fd_step(Float32) == sqrt(eps(Float32))
        @test default_fd_step(Float32) > default_fd_step(Float64)

        # h2 is a fraction of the proposed step, so it is precision-independent
        @test default_avv_step(Float64) == 0.1
        @test default_avv_step(Float32) === 0.1f0

        @test default_tolerance(Float64) ≈ 1.49012e-8 rtol=1e-5
        @test default_tolerance(Float32) == sqrt(eps(Float32))

        @test default_initialfactor(0) == 0.001    # direct-lambda methods
        @test default_initialfactor(1) == 0.001
        @test default_initialfactor(10) == 100.0   # trust-region methods
        @test default_initialfactor(11) == 100.0
    end

    @testset "keyword defaults match the reference interface" begin
        # Guards against drift from `original/pythonInterface/geodesiclm.py`.
        # `geodesiclm` is keyword-only, so the defaults are read off the method.
        m = only(methods(geodesiclm))
        kwnames = Base.kwarg_decl(m)
        for k in (:center_diff, :damp_mode, :maxiter, :artol, :xrtol, :frtol,
                  :imethod, :iaccel, :ibold, :ibroyden, :factoraccept,
                  :factorreject, :avmax, :maxfev, :maxjev, :maxaev,
                  :maxlam, :minlam)
            @test k in kwnames
        end
        # Behavioural check of the defaults that are cheap to observe:
        # `ibroyden = 0` means the Jacobian is recomputed every iteration.
        x = [-1.2, 1.0]; fvec = zeros(2)
        r = geodesiclm(rosenbrock!, rosenbrock_jac!, nothing;
                       x=x, fvec=fvec, n=2, m=2, analytic_jac=true, maxiter=500)
        @test r[5] == r[3]        # njev == niters (no Broyden updates)
        # ... and `ibroyden = 1` reuses it
        x = [-1.2, 1.0]; fvec = zeros(2)
        r = geodesiclm(rosenbrock!, rosenbrock_jac!, nothing;
                       x=x, fvec=fvec, n=2, m=2, analytic_jac=true,
                       ibroyden=1, maxiter=500)
        @test r[5] < r[3]
        # `maxiter` defaults to 200*(n+1); count the iterations with every
        # convergence criterion switched off (negative disables the ones that
        # are compared against a non-negative quantity)
        for np in (2, 3)
            hits = Ref(0)
            count_cb = (args...) -> (hits[] += 1; nothing)
            f! = (p, f) -> (for i in eachindex(f); f[i] = p[i] - i; end; nothing)
            x = zeros(np); fvec = zeros(np)
            r = geodesiclm(f!, nothing, nothing; x=x, fvec=fvec, n=np, m=np,
                           callback=count_cb, artol=0.0, Cgoal=0.0, gtol=-1.0,
                           xtol=-1.0, ftol=-1.0)
            @test hits[] == 200 * (np + 1)
            @test r[3] == 200 * (np + 1)
            @test r[7] == -1
        end
    end

    @testset "workspace reuse" begin
        # A reused workspace must give the same answer as a fresh one ...
        x1 = [-1.2, 1.0]; f1 = zeros(2)
        r1 = geodesiclm(rosenbrock!, rosenbrock_jac!, nothing;
                        x=x1, fvec=f1, n=2, m=2, analytic_jac=true, maxiter=200)
        ws = GeodesicLMWorkspace{Float64}(2, 2)
        local r2
        for _ in 1:3
            x2 = [-1.2, 1.0]; f2 = zeros(2)
            r2 = geodesiclm(rosenbrock!, rosenbrock_jac!, nothing;
                            x=x2, fvec=f2, n=2, m=2, analytic_jac=true,
                            workspace=ws, maxiter=200)
        end
        @test r2[1] == r1[1]
        @test r2[2] == r1[2]
        @test r2[7] == r1[7]

        # ... and must not allocate per iteration once it has grown
        run!(ws, x, fv) = geodesiclm(rosenbrock!, rosenbrock_jac!, nothing;
                                     x=x, fvec=fv, n=2, m=2, analytic_jac=true,
                                     workspace=ws, maxiter=200)
        x3 = [-1.2, 1.0]; f3 = zeros(2)
        run!(ws, x3, f3)
        x3 .= [-1.2, 1.0]; f3 .= 0
        short = @allocated run!(ws, x3, f3)
        @test short < 1024

        # a workspace grows to fit and is reusable across problem sizes
        wsg = GeodesicLMWorkspace{Float64}()
        for np in (2, 5, 3)
            xx = zeros(np); ff = zeros(6)
            g!(x, f) = (for i in eachindex(f); f[i] = x[min(i, length(x))] - i; end; nothing)
            r = geodesiclm(g!, nothing, nothing; x=xx, fvec=ff, n=np, m=6,
                           workspace=wsg, maxiter=50)
            @test length(r[1]) == np
            @test all(isfinite, r[1])
        end
        @test wsg.n >= 5
    end

    @testset "caller's dtd is not modified" begin
        dtd = Matrix{Float64}(I, 2, 2) .* 3.0
        dtd_copy = copy(dtd)
        x = [-1.2, 1.0]; fvec = zeros(2)
        geodesiclm(rosenbrock!, nothing, nothing;
                   x=x, fvec=fvec, n=2, m=2, dtd=dtd, damp_mode=1, maxiter=100)
        @test dtd == dtd_copy
    end

    @testset "dpmpar machine parameters" begin
        @test dpmpar(1) ≈ eps(Float64)
        @test dpmpar(2) ≈ floatmin(Float64)
        @test dpmpar(3) ≈ floatmax(Float64)
        @test_throws ErrorException dpmpar(4)
        @test dpmpar(Float32, 1) === eps(Float32)
        @test dpmpar(Float32, 2) === floatmin(Float32)
        @test dpmpar(Float32, 3) === floatmax(Float32)
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

    @testset "pure-Julia linear algebra kernels" begin
        # bounded entries, so that also Float32 products stay finite
        vals(T, m, n) = T[sin(3.0 * i + 7.0 * j) * exp(-0.2 * j) * (1 + i / m) for i in 1:m, j in 1:n]
        for T in (Float64, Float32), (m, n) in ((1, 1), (5, 3), (37, 4), (500, 7))
            A = vals(T, m, n)
            x = T[cos(1.3 * k) for k in 1:n]
            f = T[sin(0.7 * k) for k in 1:m]
            @test GeodesicLM._gemv!(zeros(T, m), A, x) ≈ A * x
            @test GeodesicLM._gemv_t!(zeros(T, n), A, f) ≈ A' * f
            C = GeodesicLM._syrk_t!(zeros(T, n, n), A)
            @test C ≈ A' * A
            @test issymmetric(C)
            @test GeodesicLM._dot(f, f) ≈ dot(f, f)
            @test GeodesicLM._nrm2(f) ≈ norm(f)
            @test GeodesicLM._axpy!(T(0.5), f, copy(f)) ≈ T(1.5) .* f
            @test GeodesicLM._scal!(copy(A), -T(2)) == -T(2) .* A
            # views, as the workspace hands out
            Aw = view(vals(T, m + 3, n + 2), 1:m, 1:n)
            @test GeodesicLM._gemv!(view(zeros(T, m + 1), 1:m), Aw, x) ≈ Aw * x
            @test GeodesicLM._syrk_t!(view(zeros(T, n + 1, n + 1), 1:n, 1:n), Aw) ≈ Aw' * Aw
        end

        # norm with over- and underflowing sums of squares, zeros, Inf and NaN
        for v in ([1.0e200, -3.0e200], [1.0e-200, 2.0e-200], Float32[1.0f30, -3.0f30],
                  Float32[1.0f-30, 2.0f-30])
            @test GeodesicLM._nrm2(v) ≈ norm(v)
        end
        for v in (zeros(3), Float64[], [1.0, Inf], [1.0, NaN])
            @test isequal(GeodesicLM._nrm2(v), norm(v))
        end

        # Cholesky and solves on the upper triangle; the lower triangle is untouched
        B = vals(Float64, 9, 4)
        S = B' * B + 0.1 * I
        F = copy(S)
        F[2, 1] = F[3, 1] = 42.0
        @test GeodesicLM._potrf_upper!(F) == 0
        @test UpperTriangular(F) ≈ cholesky(Symmetric(S)).U
        @test F[2, 1] == 42.0 && F[3, 1] == 42.0
        b = [1.0, -2.0, 0.5, 3.0]
        @test GeodesicLM._potrs_upper!(copy(b), F) ≈ S \ b
        @test GeodesicLM._trsv_ut!(copy(b), F) ≈ UpperTriangular(F)' \ b
        @test GeodesicLM._trsv_u!(copy(b), F) ≈ UpperTriangular(F) \ b

        # a failed factorization reports the first non-positive or NaN pivot
        # (like LAPACK potrf) instead of throwing
        @test GeodesicLM._potrf_upper!([1.0 2.0; 2.0 1.0]) == 2
        @test GeodesicLM._potrf_upper!([0.0 0.0; 0.0 1.0]) == 1
        @test GeodesicLM._potrf_upper!([1.0 0.0; 0.0 NaN]) == 2

        # NaN scan
        @test !GeodesicLM._hasnan(zeros(3, 2)) && GeodesicLM._hasnan([0.0, NaN, 1.0])
        @test !GeodesicLM._hasnan(Float32[Inf, -Inf]) && !GeodesicLM._hasnan(Float64[])
    end

    @testset "rank-deficient Jacobian" begin
        # The residuals depend on x[1] + x[2] only, so J'J is singular and the
        # undamped normal equations cannot be factored.
        function sum_residual!(x, fvec)
            for i in eachindex(fvec)
                fvec[i] = (x[1] + x[2]) * i - 3.0 * i
            end
        end
        function sum_jacobian!(x, fjac)
            for i in axes(fjac, 1)
                fjac[i, 1] = i
                fjac[i, 2] = i
            end
        end
        x = [0.0, 0.0]
        fvec = zeros(5)
        r = geodesiclm(sum_residual!, sum_jacobian!, nothing; x=x, fvec=fvec, n=2, m=5,
                       analytic_jac=true, iaccel=0, maxiter=200)
        @test cost(r[2]) < 1.0e-8
        @test r[1][1] + r[1][2] ≈ 3.0 atol=1.0e-4
    end

end
