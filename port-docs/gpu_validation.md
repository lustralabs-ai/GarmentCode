# GPU validation of the ROCm/HIP GarmentCode sim (workstream 8)

Date: 2026-11-10. Hardware: AMD Radeon 8060S (gfx1151). Software: Warp 1.13.0+rocm.0 from the
`garmentcode-rocm` branch (warp-amd-probe), venv-gc. Q1-Q5 were run on numpy 1.26.4; the
"NumPy 2 re-validation" section below was run on numpy 2.5.4.

Labels used in this document:
- **[M]** means measured.
- **[I]** means inferred from measurements.

## Verdicts

| Q | Verdict |
|---|---|
| Q1 GPU residency / sync / errors | **Pass.** All sim work runs on the GPU inside the captured graph. The timings are synchronised. There are no swallowed HIP errors, and the refit is complete. Early frames are slow because of `create_self_edge_contacts` (O(k^2) serial work in long stitch-spring threads). Compilation and sync artefacts are not the cause. |
| Q2 kernel fidelity | **No miscompile.** With identical inputs, CPU and GPU produce identical contact sets for every kernel tested. All beyond-rounding differences come from FMA contraction (`fuse_fp=True`, which is Warp's default on both NVRTC and HIPRTC). The `integrate_particles` velocity clamp at \|v\| = 25 amplifies them. With FMA disabled, the contact and drag kernels are bit-exact, and everything else differs only at the 1e-7 level (atomic ordering). The BVH builder (lbvh vs median) has no effect per launch. |
| Q3 statistics | **Chaos explains the dress discrepancy, and every other discrepancy.** The spread of GPU runs matches the spread of CPU runs whose inputs were perturbed by 1e-5, both in final shape and in self-intersection count. All runs are geometrically sane. |
| Q4 performance | The GPU is **44x faster than the CPU in warm steady state** (7.3 vs 320 ms/frame, shirt) and **~10x faster end-to-end** (8.6 s vs 83.6 s). Graph capture gives no speed-up. Steady state is launch/latency bound [I]. Early frames are bound by the serial tail of `create_self_edge_contacts`. Cold compile is about 110 s, of which the collide module takes 97 s. |
| Q5 real `run_sim` | **Pass.** Default device selection picks `cuda:0`. Shirt and dress both simulate and render correctly. The dress "fail" in one run (341 > 300 self-collisions) is a chaotic outcome; the CPU reference has 434. |
| NumPy 2 | **Pass.** The kernel diff is identical. The CPU run is bit-exact against the numpy-1 reference. GPU runs fall within the numpy-1 spread. `run_sim` renders fine. No DeprecationWarnings were raised under `-W error::DeprecationWarning`. venv-gc stays on numpy 2.5.4. |

No Warp source change was needed. I did not create the `ws8-gpu` worktree.

## Setup and how to rerun

All scripts are in `reference/scripts/` (new files, prefixed `gpu_`, plus
`make_perturbed_inputs_seeded.py`). I ran them from a scratch dir (`/tmp/claude-1000/ws8`).
Environment:

```bash
export ROCM_PATH=/home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH=$ROCM_PATH/lib MPLCONFIGDIR=/tmp/claude-1000/mpl
export WARP_CACHE_PATH=/tmp/claude-1000/warp-cache-ws8
PY=/home/carl/lustra/GarmentCode_hack/venv-gc/bin/python
GC=/home/carl/lustra/GarmentCode_hack/GarmentCode
R=/home/carl/lustra/GarmentCode_hack/reference
S=$R/scripts; I=$R/data/inputs
```

| Script | Purpose |
|---|---|
| `gpu_common.py` | Shared setup. Builds a GarmentCode `Cloth` from harness inputs on a chosen device. `disable_fma()` sets `fuse_fp=False` on the sim kernel modules (diagnostic only). |
| `gpu_graph_audit.py` | Q1. Lists launches during graph capture against the nodes in the graph, does a HIP error check, compares graph vs eager vs graph, runs a one-substep-fewer control, and checks the refit. |
| `gpu_kernel_diff.py` | Q2. Isolated per-launch replay. It records every kernel launch of one substep on the CPU, replays each launch on the GPU with the CPU inputs, then diffs the outputs. Contact buffers are compared as sets. Flags: `--gpu-bvh`, `--warm-frames`, `--no-fma`. |
| `gpu_fma_clamp_repro.py` | Q2 minimal reproducer of the clamp-threshold flip. |
| `gpu_harness_variant.py` | Runs `dump_harness.py` unchanged, with the `--bvh` / `--no-fma` diagnostic variants. |
| `make_perturbed_inputs_seeded.py` | Q3. Perturbs the harness input vertices by 1e-5 cm with a seed. |
| `gpu_runs_table.py`, `gpu_geom_sanity.py`, `gpu_render_runs.py` | Q3. Spread table, geometric sanity, matplotlib renders. |
| `gpu_profile.py` | Q4. Per-frame wall time and contact counts. `--mode graph\|eager\|kernels` (kernels = Warp per-kernel timing), `--device cpu`. |
| `gpu_edge_query_load.py` | Q4. Per-spring candidate count k for `create_self_edge_contacts`. |
| `gpu_gc_run_default.py` | Q5 and NumPy 2. GarmentCode's `test_garment_sim` main path with no `set_device`. Runs in a private cwd. |

Commands, run sequentially on one GPU. Every GPU command was wrapped in `timeout 2400`-`3600`.

```bash
# Q1
$PY $S/gpu_graph_audit.py --garmentcode-dir $GC --inputs $I/shirt_mean --frames 12
# Q2 (also: --gpu-bvh lbvh; --warm-frames 40/150; dress_pencil --warm-frames 60; js_mean_all --warm-frames 40; --no-fma)
$PY $S/gpu_kernel_diff.py --garmentcode-dir $GC --inputs $I/shirt_mean --gpu-bvh median [--warm-frames N] [--no-fma] --out $PWD/kd/X.json
$PY $S/gpu_fma_clamp_repro.py [--no-fma]
# Q3: three GPU runs per garment (lbvh default), median-BVH and no-FMA variants
$PY $S/dump_harness.py run --garmentcode-dir $GC --inputs $I/$g --out runs/gpu_${g}_$k --device cuda:0 --save-every 2
$PY $S/gpu_harness_variant.py --bvh median -- run --garmentcode-dir $GC --inputs $I/shirt_mean --out runs/gpu_shirt_mean_median_$k --device cuda:0 --save-every 2
$PY $S/gpu_harness_variant.py --no-fma   -- run --garmentcode-dir $GC --inputs $I/$g --out runs/gpu_${g}_nofma_1 --device cuda:0 --save-every 2
# Q3: CPU perturbed runs (seeds 1-4 shirt, 1-5 dress, 1-2 js); one warm kernel cache copy per parallel job
$PY $S/make_perturbed_inputs_seeded.py $I/dress_pencil inputs/dress_p$s 1e-5 $s
WARP_CACHE_PATH=<per-job copy> $PY $S/dump_harness.py run --garmentcode-dir $GC --inputs inputs/dress_p$s --out runs/cpu_dress_p$s --device cpu --save-every 2
$PY $S/gpu_runs_table.py --ref $R/data/dress_pencil/cpu_run1 runs/gpu_dress_pencil_{1,2,3} runs/cpu_dress_p{1,2,3,4,5}
$PY $S/gpu_geom_sanity.py runs/* $R/data/*/cpu_run1 --json geom_sanity.json
$PY $S/gpu_render_runs.py renders/dress_pencil.png $R/data/dress_pencil/cpu_run1 runs/gpu_dress_pencil_{1,2,3} runs/cpu_dress_p{1,2}
# Q4 (quiet machine: no other GPU/CPU sim jobs)
$PY $S/gpu_profile.py --garmentcode-dir $GC --inputs $I/$g --mode graph|eager|kernels --out $PWD/prof/${g}_MODE.json
$PY $S/gpu_profile.py --garmentcode-dir $GC --inputs $I/shirt_mean --device cpu --mode eager --out $PWD/prof/shirt_mean_cpu.json
$PY $S/gpu_edge_query_load.py --garmentcode-dir $GC --inputs $I/shirt_mean --frames 0,1,5,12,20,39,60,150 --out $PWD/eql/shirt.json
# Q5
$PY $S/gpu_gc_run_default.py $GC $GC/assets/Patterns/${g}_specification.json $PWD/gc_out
```

## Q1: Is everything on the GPU?

[M] `gpu_graph_audit.py`, shirt, 12 frames:

- **Graph contents.** During GarmentCode's per-frame graph capture, 94 `wp.launch` calls are issued. Every one appears as a kernel node in the captured HIP graph, along with the BVH refit kernels. No launch targets the CPU. There are no host round-trips inside a frame apart from the graph launch and the end-of-frame readback of positions and contact counts.
- **Timing synchronisation.** Each frame ends with a device-to-host copy of `particle_q`, which synchronises. The per-frame wall times in `gpu_profile.py` and in GarmentCode's own timer therefore cover the GPU work, not just its enqueue. Kernels mode uses Warp's synchronised kernel timing.
- **Errors.** The eager run with `wp.config.verify_cuda=True` (a check after every launch) is clean. `hipDeviceSynchronize` and `hipGetLastError` after every graph launch returned 0. No contact-buffer "exceeded" warnings appear in any log. Peak edge-contact slot usage vs capacity: shirt 1.96M/4.55M, dress 3.9M/8.2M, js 8.7M/13.0M.
- **Refit.** After the refit, the mesh points equal `particle_q`. Each of the 15712 faces finds itself in a self-query. Mean candidates are 13.59 for both the graph run and the eager run.
- **Graph vs eager.** At frame 0 the maximum difference between graph and eager is of the same order as the difference between two graph runs (about 2e-3 cm; GPU atomics are non-deterministic). For comparison, the control of dropping one substep gives 0.042 cm max (0.017 rms). So graph replay executes the same work as eager.

**Why the early frames cost about 16x more.** On the shirt, GPU frame 0 takes 1068 ms against 7.3 ms in steady state, a ratio of about 150x. Frames 1-9 take about 276 ms, roughly 38x steady state. The "16x" figure corresponds to the frames 0-39 average.
- [M] Per-kernel timing shows that `create_self_edge_contacts` takes 1037 ms of shirt frame 0, then 115 ms at frame 20, 17 ms at frame 39 and 2.7 ms at frame 60.
- [M] `gpu_edge_query_load.py` explains why. The kernel runs one thread per spring, and each thread does an AABB query with half-width `len/2 + 3r`. That yields k candidate faces and therefore 3k edges, each of which goes through a linear `check_edge_exists` scan. Work per thread is O(k^2). Long stitch springs (up to 28.5 cm at frame 0) have huge k:

| shirt frame | max spring len (cm) | k_max | max k^2 | sum k^2 | share of sum k^2 from springs > 3x median length |
|---|---|---|---|---|---|
| 0 | 28.5 | 1567 | 2.46M | 428M | 98.6% |
| 39 | 9.1 | 230 | 53k | - | - |
| 60 | 2.3 | 43 | 1849 | - | 0% |

  js_mean_all frame 0 has k_max = 2384.
- [M] The total number of contact slots falls only 1.6x between these frames, while the kernel time falls about 400x. The time therefore tracks the slowest thread's k^2, not the total work.
- [I] The kernel costs roughly 0.3-0.4 us per unit of k^2 in the tail thread. This is not a precise model: frame 1 is an outlier.
- The early cost is a GPU serial tail. It is not compilation, not synchronisation, and not CPU fallback. The CPU (single-threaded) pays the sum of k^2 instead, which is spread more evenly over frames, so its early inflation is only about 1.6x.
- The cost disappears once the seams close (frames 40-60).
- A possible fix belongs on the GarmentCode side and is not done: exclude stitch springs from edge self-contact, cap them, or dedupe the candidate edges.

## Q2: Kernel-level CPU vs GPU fidelity

**Method.** `gpu_kernel_diff.py` runs one substep on the CPU and records every launch together with its inputs. It then replays each launch on the GPU starting from exactly the CPU inputs, so that errors do not compound. Contact outputs are compared as sets because slot order is atomic-dependent. Differences are reported relative to the scene scale.

| Case | Contact sets | Beyond-rounding kernels (max rel. diff) |
|---|---|---|
| shirt f0, median / lbvh | identical | integrate_particles 1.04e-2; reference_shape_drag 9.5e-4 |
| shirt warm 40, median / lbvh | identical | integrate_particles (clamp flips) |
| shirt warm 150 | identical | solve_edge_edge_self_contact 2.7e-4 |
| dress warm 60 | identical | integrate_particles |
| js warm 40 | identical | integrate_particles |
| shirt f0 and w40, `--no-fma` | identical | **none**. Contact and drag kernels are bit-exact. integrate is 7.6e-8-1.5e-7. Springs and bending are about 1e-7 (atomic order). apply_particle_deltas is 0. |

All other kernels differ by at most about 1e-5 of the scale.
- [M] The median and lbvh builders give identical per-launch results. Builder choice alone has no effect on kernel outputs; any later difference comes from traversal and atomic order inside chaotic dynamics.

**First beyond-rounding kernel: `integrate_particles`. Cause: the velocity clamp threshold.**
- How the kernel works: the damping step is `v *= exp(-0.25*|v|*dt)`, and then if `|v| > 25` the kernel sets `v = normalize(v)*25`.
- That clamp is discontinuous. The jump at the threshold is 25*(1 - exp(-0.25*25/600)) = **0.259**.
- Any particle clamped in the previous substep has |v| = 25 +/- 1 ulp. FMA contraction changes that last ulp, which flips the branch.
- [M] `gpu_fma_clamp_repro.py` feeds 200000 CPU-produced `normalize(d)*25` velocities to both devices:
  - With FMA, `normalize*25` differs in 21051 cases and the length in 28203, clamp decisions disagree in 13880, and the maximum \|v_new\| difference is **0.25907**, equal to the branch gap.
  - With `--no-fma`, there are 0 differences in every quantity.
- This is not a miscompile. `fuse_fp=True` (`-ffp-contract=fast`) is Warp's default, and NVRTC on NVIDIA behaves the same way. CPU Warp does not contract.

**Frame-0 difference.**
- [M] GPU vs CPU reference, max: 2.2e-3 cm.
- [M] GPU with no FMA vs CPU: 7.3e-4 cm. What remains is atomic ordering.
- [M] CPU with 1e-5 input perturbation vs CPU: 7e-4 to 3.5e-3 cm.
- So the frame-0 GPU difference is the same size as the CPU's own sensitivity to a 1e-5 input change.

## Q3: Statistics

The reference CPU runs are deterministic (shirt run2 is bit-identical to run1). Their results:

| Garment | Frames | Self-intersections |
|---|---|---|
| shirt | 247 | 13 |
| dress | 870 | 434 |
| js | 789 | 0 |

All runs use dump_harness.

| Group | Frames | Self-int | Max distance to CPU ref over run (cm) |
|---|---|---|---|
| GPU shirt, lbvh x3 | 224 / 244 / 239 | 4 / 3 / 10 | 2.1-2.4 |
| GPU shirt, median BVH x3 | 254 / 210 / 231 | 10 / 4 / 2 | similar |
| GPU shirt, no FMA | 231 | 0 | - |
| CPU shirt, perturbed 1e-5 x4 | 244 / 260 / 236 / 244 | 5 / 6 / 9 / 4 | 2.1-2.7 |
| CPU shirt, ref perturb1e-5 | 247 | 12 | - |
| GPU dress x3 | 604 / 1119 / 751 | 268 / 140 / 223 | 5.0-5.5 |
| GPU dress, no FMA | 753 | 405 | - |
| CPU dress, perturbed 1e-5 x5 | 544 / 608 / 559 / 496 / 552 | 104 / 275 / 589 / 260 / 191 | 4.4-5.4 |
| GPU js x3 | 753 / 494 / 521 | 0 / 0 / 0 | 2.5-2.8 |
| CPU js, perturbed 1e-5 x2 | 449 / 563 | 0 / 0 | 2.65 |

**Chaos.** [M] For every garment, GPU runs and perturbed CPU runs differ from the reference by the same amount. The dress self-intersection count ranges from 104 to 589 on the CPU from a 1e-5 cm input change alone. The GPU counts (140-268) and the reference's 434 both fall inside that range. Frame counts (the equilibrium stop) scatter the same way. The verdict is that the dress discrepancy is explained by chaotic sensitivity, not by a GPU defect. Builder choice is indistinguishable from noise.

**Geometric sanity.** [M] `gpu_geom_sanity.py` computes signed distance with the Warp winding number. Over all GPU and CPU runs: no NaN, inside-body fraction 0, max penetration 0, body intersections 0, median cloth-body distance about 1 cm. Max edge stretch ratios match the CPU references:

| Garment | Runs | CPU reference |
|---|---|---|
| shirt | ~1.85 | 1.84 |
| dress | 1.09-1.21 | 1.19 |
| js | 3.84-3.87 | 3.87 |

The js value of 3.87 is a seam/stitch edge and is present on the CPU too.

**Renders.** [M] I inspected `renders/{shirt_mean,dress_pencil,js_mean_all}.png` (front and side, coloured by distance to the reference). The garments sit on the body and look plausible. The dress differs only in the fold pattern around the waist.

## Q4: Performance

[M] `gpu_profile.py`, quiet machine, warm kernel cache, times in ms per frame:

| Run | f0 | median f1-9 | median f10-39 | median f40+ | Total |
|---|---|---|---|---|---|
| shirt GPU graph | 1068 | 276 | 82 | 7.27 (p90 8.25) | 8.58 s / 237 fr |
| shirt GPU eager | 1075 | 274 | 81 | 6.54 | 8.52 s |
| shirt CPU eager | 1134 | 490 | 373 | 320 | 83.6 s / 247 fr |
| dress GPU graph | 1749 | 88 | 36 | 15.0 | - |
| dress GPU eager | - | - | - | 14.5 | - |
| js GPU graph | 4991 | 391 | 106 | 17.0 | - |
| js GPU eager | - | - | - | 15.4 | - |

- **GPU vs CPU.** Shirt steady state is 44x faster (7.3 vs 320 ms). End to end it is 9.7x faster (8.6 s vs 83.6 s), because the early serial tail limits the gain.
- **Graph vs no graph.** There is no benefit; eager is equal or slightly faster.
- **Per-kernel breakdown.** [M] `create_self_edge_contacts` takes this share of total GPU kernel time:

  | Garment | Share |
  |---|---|
  | shirt | 89.7% |
  | dress | 74.7% |
  | js | 79.1% |

- **Steady-state boundedness.** In steady state the kernel time sums to about 6 ms out of 7.2 ms wall, spread over about 150 small launches per frame. [I] Steady state is therefore launch/latency bound, not arithmetic bound. Early frames are bound by traversal and the serial tail (see Q1).
- **Cold compile** [M] (HIPRTC, empty cache):

  | Module | Time |
  |---|---|
  | `warp.sim.collide` | 96.7 s |
  | integrator_xpbd | 6.5 s |
  | integrator_euler | 3.8 s |
  | inertia | 1.6 s |
  | panel_assignment | 1.35 s |

  The warm-cache load is 1.4-3 s. HIPRTC ignores the PCH option.
- **Compile options** (notes only, nothing changed):
  1. Keep a persistent `WARP_CACHE_PATH`, or ship a pre-warmed cache.
  2. Run a warm-up or precompile step (`wp.load_module`) at install time.
  3. Use a lower `optimization_level` for the collide module (trades runtime speed).
  4. Split `warp.sim.collide` into smaller or `module="unique"` kernel modules, so the sim compiles only what it uses.
- The previously reported 0.49 s/frame on the GTX1070 is **not verified** here. [I] It is probably compile-dominated, like ws6's cold-cache 0.53 s/frame.

## Q5: Real `run_sim` path, default device

[M] `gpu_gc_run_default.py` (no `set_device`). Warp printed `default device cuda:0`.

| Garment | Wall | sim_time | Frames | Self-collisions | Body | Outcome |
|---|---|---|---|---|---|---|
| shirt_mean | 11.8 s | 9.54 s | 247 | 10 | 0 | ok, spf 0.0386 |
| dress_pencil | 16.3 s | 13.5 s | 582 | 341 | 0 | fail `cloth_self_intersection` (>300) |

The dress failure is a chaotic outcome: the CPU reference has 434, perturbed CPU runs range 104-589, and the numpy-2 run below passed with 169. I inspected the front and back renders, and they look correct.

## NumPy 2 re-validation

- **Versions.** [M] numpy was 1.26.4 before the upgrade. It was upgraded with `VIRTUAL_ENV=.../venv-gc UV_CACHE_DIR=/tmp/claude-1000/uvcache /home/carl/py/bin/uv pip install -U 'numpy>=2'` and is now numpy 2.5.4 (warp 1.13.0+rocm.0). venv-gc **stays on numpy 2**.
- **Commands.** These are the same commands as above, with `$PY -W error::DeprecationWarning` (except one non-`-W` `run_sim` per garment). Outputs are in `/tmp/claude-1000/ws8/np2/`.

| Check | Result |
|---|---|
| Kernel diff, shirt f0 and warm 40 (median BVH) | Summary **identical** to numpy 1: same kernels, same maxima, identical contact sets. |
| CPU shirt 10 frames vs numpy-1 CPU reference | **Bit-exact**: init verts and frames 0, 2, 4, 6, 8 (the reference saves every 2nd frame), max diff 0.0. |
| GPU shirt, 1 run | 241 frames, 0 self-int, max distance over run 2.26 cm, spf 0.0358. Within the numpy-1 spread. |
| GPU dress, 1 run | 652 frames, 211 self-int, 4.72 cm. Within the spread. |
| GPU js, 1 run | 748 frames, 0 self-int, 3.11 cm. Slightly above the 3-run GPU range (2.5-2.8). One sample of a chaotic quantity, frames and self-int are in range, and geometric sanity is normal. I judge it consistent, but note it. |
| `run_sim` GPU shirt | 235 frames, 10 self-collisions, wall 11.6 s. The renders look correct. |
| `run_sim` GPU dress | 691 frames, 169 self-collisions, no fail, wall 17.8 s. The renders look correct. |
| `run_sim` GPU shirt under `-W error::DeprecationWarning` | 233 frames, 17 self-collisions, ok. |
| Deprecations | **None raised** anywhere on the sim/run_sim path. |
| Box-mesh vertex counts | Unchanged: shirt 7990 v / 15712 f, dress 14425 v / 28506 f. The possible pattern-gen count change (e.g. pencil skirt 11576 to 11578) did not occur for these two garments. |

**Verdict.** NumPy 2 changes nothing measurable in the GPU or CPU sim path.

**Not covered.** The GarmentCode GUI and pattern-fitter paths under numpy 2. The `-W error` check covers only the code that ran here.

## Defects found

- **Warp / ROCm port:** none. The only behavioural difference between CPU and GPU is FMA contraction, which is Warp's documented default and matches NVIDIA.
- **GarmentCode (performance, not correctness; not changed):** `create_self_edge_contacts` does O(k^2) per-thread work, and long stitch springs dominate the first ~40 frames.

## Unverified

- The GTX1070 0.49 s/frame claim.
- Any direct comparison against an NVIDIA/CUDA run of this branch.
- An exact timing model for `create_self_edge_contacts`; k^2 is only an approximate fit.
- Statistical power. With n = 3-5 runs per group, the evidence says the GPU and perturbed-CPU spreads overlap. It cannot rule out small distribution shifts.
- The GUI and pattern-fitter paths under numpy 2.
