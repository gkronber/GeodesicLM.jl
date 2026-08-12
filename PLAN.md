# GPU support for GeodesicLM — implementation plan

**Branch:** `GPU`
**Status:** foundation (M0–M2) implemented & unit-tested; see `src/gpu/KAOps.jl`
and `test/gpu_kernels.jl`. This document is the reviewable roadmap for the
remaining work.

---

## 1. Goal & assumptions

Make it possible to run GeodesicLM **entirely on a GPU** using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), so
the same package works on CUDA, Metal, AMDGPU and oneAPI by selecting a backend,
and so the optimizer never has to copy the large data/`f`/gradient arrays back to
the CPU.

Design assumptions (per the request):

- The **slow part is the user's evaluation** of `f(x,p)` and its gradient
  `∂f/∂p`. Both are evaluated on the GPU.
- Therefore the LM linear algebra only needs to be *correct and portable*, not
  highly tuned. Small, simple kernels are acceptable.
- We must avoid switching the *working set* back to the CPU. A **few scalars**
  (cost `C`, norms, `λ`, `δ`, convergence flags) still round-trip to the host
  each iteration for control flow — this is unavoidable and negligible.
- Reuse the CPU `GLMWorkspace` design (OncePerTask lazy cache) but backed by
  device arrays. Same `(n, m)` keying; buffers allocated once, reused.

## 2. Environment / backends

- Development machine: Apple M2 Pro (`arm64`) → realistic GPU backend is
  **Metal.jl**; CUDA is the primary target on CI/Linux.
- KernelAbstractions ships a **CPU backend** (`CPU()`) which we use for all unit
  tests so the port is testable **without a GPU**. The same kernels run on real
  GPUs unchanged — only the `Backend` passed at launch differs.
- KernelAbstractions 0.9 is already a dependency of `GeodesicLM`.

## 3. Architecture overview

```
                    HOST (control flow: λ, δ, accept/reject, convergence)
                     │  launch kernels; read a few scalars (C, norm, cos_alpha)
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │               GPU / device (KAOps + user kernels)        │
        │  x (n)         params            fvec (m)  residuals      │
        │  fjac (m×n)    Jacobian          jtj (n×n), g (n×n)      │
        │  v, vold, a (n) step/acc         dtd (n×n) damping       │
        │  user:  f_kernel!(x, fvec, data…)                        │
        │         grad_kernel!(x, fjac, data…)                     │
        └──────────────────────────────────────────────────────────┘
```

The optimizer is refactored into a host-side **orchestrator** that launches
KAOps kernels (and the user's `f`/gradient kernels) for every array operation,
and reads back only the handful of scalars needed for branching. No `m`- or
`n`-sized array ever needs to come back to the host.

## 4. Data layout & workspace

- `x` length `n` (small), `fvec` length `m` (large), `fjac` `m×n` (column-major),
  `jtj`, `g`, `dtd` `n×n` (small). All allocated on the device via
  `KAOps.alloc(T, backend, dims...)`.
- New `GPUWorkspace(n, m, backend)`: mirrors `GLMWorkspace` but holds device
  arrays. OncePerTask cache keyed by `(n, m, backend)`.
- **Scalar round-trips** use small length‑1 device arrays (`cost`, `norms`, …)
  copied to host; this is the only device→host traffic per iteration.

## 5. User interface for `f` and gradient on the GPU

The existing `func(x, fvec)` is a *host* callback. For GPU we need a device
protocol. Proposal — a small interface type:

```julia
struct GPUObjective{F,G}
    fun!::F        # kernel (or launcher): fun!(backend, x, fvec, data…) -> nothing
    grad!::Union{G,Nothing}
    data           # anything captured by the kernels (device arrays)
end
```

- `fun!` computes residuals into device `fvec` given device `x` (strided over
  `m` rows). `grad!` (optional) writes device `fjac`; if absent we provide a
  GPU **finite-difference** Jacobian using KAOps + `fun!` (as in the CPU code,
  `fdjac!` / `fd_avv!`), so analytic gradients are optional but recommended.
- Public entry point `geodesiclm_gpu(goal::GPUObjective; x, n, m, kwargs…)`.
  (Name TBD; an alternative is dispatching on the `GPUObjective` type.)

## 6. Kernel building blocks needed (KAOps)

Implemented & unit-tested (M0–M2):

| Kernel | Purpose | Notes |
|--------|---------|-------|
| `fill!`, `copyto!`, `axpy!`, `scale!` | elementwise setup | done |
| `mul!(y, A, x)` / `mul!(y, A', x)` | mat-vec (m·n, n·m) | done |
| `AtA!(C, A, B)` | `J'J` / `jtj` | done |
| `dot!`, `norm2` | reductions (two-stage: grid-stride + serial final) | done |
| `nanflag` | NaN guard on `x`/`fvec`/`fjac` | done |

Not yet implemented (M3+):

| Kernel | Purpose | Notes |
|--------|---------|-------|
| `cholesky!` (n×n, :U) | factor `g = jtj + λ·dtd` | single-thread block; tiny n; `PosDef` failure path |
| triangular solves `ldiv!` | `v = L\u`, `a = L\u` | small n; single thread |
| `axpy_mat!` (`g = jtj + λ·dtd`) | — | trivial, can be an elementwise kernel over n² |
| compose `x_new = x + v + ½a` | — | compose existing elementwise ops |
| GPU `fdjac!` / `fd_avv!` | finite-difference (optional) | built on `fun!` + elementwise ops |

## 7. Milestones

**M0 — Setup & harness** (done)
- Create `GPU` branch; add KernelAbstractions dependency.
- Establish `KAOps` submodule + CPU-backend test harness with a
  `test_backends()` hook so real GPU backends can be dropped in later.

**M1 — Elementwise & reductions** (done)
- `fill!`, `copyto!`, `axpy!`, `scale!`, `dot!`, `norm2`, `nanflag`.
- Unit tests validate every operation against `Base`/`LinearAlgebra` references
  on the CPU backend across many sizes (incl. non-multiples of the thread count).

**M2 — Mat-vec & matmul** (done)
- `mul!` (both orientations), `AtA!` (`J'J`). Tests vs. `*`/`'`.

**M3 — Cholesky & triangular solves (next priority)**
- Write `cholesky!` over the upper triangle of an in-place `n×n` buffer (mirror
  the CPU `cholesky!` already used), with a `PosDef` failure signal via `info`.
- Write single-thread `ldiv!` (forward/back substitution) using the factor.
- **Exploratory tests:** symmetric positive-definite `g` of sizes
  `n ∈ {2,3,5,10,20}` → factor, solve `g*L\u ≈ u`; indefinite matrices → failure
  path returns the same `info` as the CPU routine. Compare against
  `LinearAlgebra.cholesky` on the CPU backend.

**M4 — GPUWorkspace + OncePerTask (backend-aware)**
- Device-array workspace mirroring `GLMWorkspace`; `_get_workspace(n, m, backend)`.
- **Tests:** allocate on CPU backend, verify reuse (same pointer across calls),
  and that distinct `(n, m, backend)` keys do not collide.

**M5 — GPUObjective interface + finite differences**
- Define `GPUObjective`; wire `fun!`/`grad!` as KA kernels.
- Implement GPU `fdjac!`/`fd_avv!` (optional Jacobian) using KaOps + `fun!`.
- **Tests:** a toy GPU residual (`y = a·exp(−t/τ)`) as a KA kernel; verify
  analytic vs finite-difference Jacobians agree on-device; a toy `grad!` kernel
  is stably differentiable.

**M6 — Assemble one LM iteration on-device**
- Port the step computation using KAOps only: build `jtj`, `g = jtj+λ·dtd`,
  `cholesky!`, solve for `v`, compute `jv`, `cos_alpha`, `pred_red`, `av`;
  accept/reject using scalars.
- **Exploratory test:** reproduce, on the CPU backend, the exact `v`, `av`,
  `cos_alpha`, `pred_red` of the CPU routine for a fixed snapshot of
  `(x, fvec, fjac, λ, dtd)`. This is the single most important correctness gate.

**M7 — Full `geodesiclm_gpu` orchestrator**
- Host loop that calls the M6 step, user kernels, convergence check, λ/δ
  updates (scalars on host) — a faithful port of `geodesiclm_alg.jl` control
  flow, but every array op is a KAOps kernel.
- **Tests:** same toy problems as the CPU suite (quadratic, rosenbrock,
  exponential) — solutions and convergence codes must match the CPU results
  within tolerances, but with `x`/`fvec`/`fjac` remaining on device.

**M8 — Real GPU backends & hardening**
- Add Metal.jl (this machine) and CUDA.jl (CI) as *optional* test extras; run
  `test/gpu_kernels.jl` and the M3–M7 tests on them via `test_backends()`.
- Allocations/launch-overhead micro-tuning only if needed; document that the
  bottleneck is `f`/gradient and that small `(n, m)` may be slower than CPU.
- Optional: `isgpu()` helper and clear error when a GPU array type is seen but
  the backend package is not loaded.

## 8. Explorative steps & risk reduction

1. **Kernel-library unit tests first** (done for M0–M2). Each building block is
   validated on the CPU backend before being used in any orchestration — de-risks
   the whole port without needing a GPU.
2. **Snapshot regression test (M6)**: freeze one LM iteration's inputs and assert
   the GPU/CPU assembly produce identical `v`, `av`, `cos_alpha`, `pred_red`.
   Catches subtle kernel bugs (indexing, associativity) in isolation.
3. **Cholesky cross-check (M3)**: compare the custom `cholesky!`/`ldiv!` against
   `LinearAlgebra` for both PD and indefinite inputs.
4. Bump `test_backends()` to `[CPU()]` by default and add
   `using Metal` / `using CUDA` guarded by `Base.find_package` so CI without a
   GPU still passes and GPU runs are opt-in.

## 9. Open questions for review

1. **Public API shape** — separate `geodesiclm_gpu(goal, …)` vs. keyword-based
   dispatch on a `GPUObjective`. I propose the former (keeps the CPU API
   untouched).
2. **Reduction portability** — current `dot!`/`norm2` uses a simple grid-stride +
   serial-final reduction (correct everywhere, not tree-optimal). For very large
   `m` this could be a bottleneck; OK to leave for now given the "don't tune"
   premise, or swap in a workgroup-tree reduction later.
3. **`f`/gradient kernel signature** — should the kernel capture a closure over
   device `data`, or should we pass a fixed tuple of user buffers explicitly
   (better for GPU codegen/const-capture)? Recommend explicit buffers.
4. **Element type** — assume `Float64` device buffers (CUDA/Metal support it,
   Metal via float64 buffers). Keep `eltype`-generic kernels anyway.
5. **Batched use-case** — is the target a single problem on GPU, or *many
   independent problems* (batched LM)? If batched, the whole workspace scheme
   becomes arrays of `(n_batch × …)` and Cholesky becomes a batched factor.
   This changes M3/M4 design; please confirm the intended use-case.

## 10. Deliverables per review checkpoint

| Checkpoint | Deliverable |
|-----------|-------------|
| This branch | `PLAN.md`; `src/gpu/KAOps.jl` (M0–M2); `test/gpu_kernels.jl`; all 127 tests green |
| Next (M3–M5) | `cholesky!`/`ldiv!` kernels + tests; `GPUWorkspace`; `GPUObjective` + `fdjac!` |
| After (M6–M7) | on-device LM step + full `geodesiclm_gpu`, matching CPU results |
| Final (M8) | Metal/CUDA test runs, docs, optional batched mode |
