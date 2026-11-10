# Porting GarmentCode to AMD Radeon 8060S (ROCm/HIP)

Written 2026-11-10. Status is tracked at the bottom.

## Goal

Run GarmentCode's cloth draping (XPBD simulation in NVIDIA Warp) on this machine's
Radeon 8060S APU (gfx1151, ROCm 10.1), with results matching the NVIDIA reference
on `alcyone` (GTX 1070).

## Findings that shape the plan

- AMD's ROCm port of Warp is public: `github.com/AMD-Ecosystem/warp`, branch
  `amd-integration` (also `release/1.13.0.amd0`), 57 commits on Warp 1.13.0.
  Git remote `amd` in `NvidiaWarp-GarmentCode`.
- It builds for gfx1151 against the ROCm 10.1 pip SDK in `/home/carl/py` in ~45 s.
  One fix was needed: hiprtc in ROCm 10.1 lacks `uint8_t..uint64_t` at JIT time
  (typedefs added in `warp/native/builtin.h`).
- Smoke test passes on the 8060S: kernel launch with `atomic_add`, mesh ray/point
  queries, graph capture and replay. Device shows as `cuda:0`, `is_cuda` True,
  `is_hip` True.
- The AMD fork has no `warp.sim` (removed upstream in 1.10; last tag with it: v1.9.1).
- GarmentCode's Warp fork (`garmentcode_hack` branch) is 2 commits on Warp
  1.0.0-beta.6 (`97aba17a1`): ~100 lines native (`mesh_query_edge` in
  `warp/native/mesh.h`, registration in `warp/builtins.py`, `exports.h`), and
  ~2,500 lines of Python in `warp/sim/{collide,integrator_xpbd,integrator_euler,model,utils,particles}.py`
  plus new `warp/collision/panel_assignment.py`.

## Strategy

Branch `garmentcode-rocm` from `amd/amd-integration`. Add the fork's `warp/sim` and
`warp/collision` back into the tree and fix API drift from beta.6 to 1.13, so
`import warp.sim` keeps working and GarmentCode needs little or no change. Lift the
fork's own sim code (not v1.9.1 sim + re-applied changes) so the physics stays
identical to the reference; use upstream's beta.6 -> v1.9.1 sim history as the guide
for each API change.

Rejected: backporting HIP to the beta.6 fork (native layer diverged too far).

Validation: get the port running on the CPU backend first and compare with the
NVIDIA fork's CPU backend on alcyone (should match almost exactly). That separates
porting mistakes from HIP bugs. GPU runs are compared statistically against the
alcyone CUDA baseline (shirt: 222 frames, 0 body intersections, 10 self-intersections,
~0.49 s/frame; 40-garment batch: 10/39 flagged, 0 crashes).

## Layout

`$WS` in these docs is the workspace directory that holds the checkouts side by side
(originally `/home/carl/lustra/GarmentCode_hack`).

| Path under `$WS` | Purpose |
|---|---|
| `GarmentCode/` | Application; our fork, branch `amd-rocm`. Holds these docs in `port-docs/` |
| `GarmentCode/port-docs/reference/` | Comparison/validation harness (scripts, README) |
| `NvidiaWarp-GarmentCode/` | Warp checkout, branch `garmentcode-rocm` (the port). Earlier docs call this `warp-amd-probe/` |
| `venv-ref/` | venv of the CPU-only beta.6 reference build (the build itself, `warp-ref-cpu/`, was removed) |
| `venv-gc/` | GarmentCode environment (see `SETUP.md`, `scripts/setup_amd.sh`); removed, the runs since use `/home/carl/py` |
| `reference/data/` | Reference inputs and CPU runs (414 MB, not in git) |
| `port-docs`, `reference/scripts` | Symlinks into `GarmentCode/port-docs` for older commands |

Build / run environment for Warp on this machine:

```
export PATH=/home/carl/py/bin:$PATH
export ROCM_PATH=/home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH=$ROCM_PATH/lib
python build_lib.py --rocm-path=$ROCM_PATH --hip-arch=gfx1151
```

Never install into `/home/carl/py` (user's venv). Each agent uses its own
`WARP_CACHE_PATH` and its own git worktree when editing Warp.

## Workstreams

| # | Workstream | Model | Depends on |
|---|---|---|---|
| 1 | Base bring-up: run Warp's test suite on gfx1151, classify failures by whether the sim touches them | Sonnet | - |
| 2 | API-drift cheat sheet, beta.6 -> 1.13 | Sonnet | - |
| 3 | Reference harness on alcyone: per-frame vertex dumps (CPU and CUDA), comparison script | Sonnet | - |
| 4 | GarmentCode environment on this machine; pattern + meshing stages | Sonnet | - |
| 5 | Native `mesh_query_edge` on the 1.13 BVH, builtin registration, test | Opus | 1 |
| 6a | Sim port: `model.py`, `collide.py`, `panel_assignment.py` | Opus | 2 |
| 6b | Sim port: `integrator_xpbd.py`, `integrator_euler.py` | Opus | 2 |
| 6c | Sim port: `render.py`, `utils.py`, `particles.py`, `__init__.py`; drop unused importers | Sonnet | 2 |
| 7 | CPU integration: GarmentCode end to end on CPU, match alcyone CPU reference | Opus | 3, 4, 5, 6 |
| 8 | HIP integration: run on the 8060S, fix HIP failures, graph capture in the sim loop, performance | Opus | 7 |
| 9 | Batch validation (40 garments), build script, docs | Sonnet | 8 |

## Risks

- gfx1151 is not a target AMD tests (they claim gfx942, ROCm 7.x). Fallback: CPU
  backend on 32 cores from stream 7.
- Licensing: the fork's sim code dates from Warp beta.6 (pre-Apache-2.0, to be
  confirmed); the AMD fork is Apache 2.0. Open question for the user. Alternative is
  re-applying GarmentCode's changes on the v1.9.1 sim.
- API drift across ~16k lines of sim code; mostly mechanical.

## Status

- 2026-11-10: feasibility done; branch `garmentcode-rocm` created with typedef fix; streams 1-4 started.
- 2026-11-10: streams 1-6 done and merged into `garmentcode-rocm` (see `test_triage.md`,
  `api_drift.md`, `local_env.md`, `mesh_query_edge.md`, `sim_port.md`).
  alcyone was off, so stream 3 built the beta.6 fork CPU-only on this machine
  (`warp-ref-cpu/`, `venv-ref/`) as the reference; CUDA reference runs on alcyone are deferred.
- 2026-11-10: stream 7 done. Port is bit-identical to the beta.6 fork on CPU for
  shirt_mean (247 frames), dress_pencil (870), js_mean_all (789); re-checked from the
  integration branch (shirt, 40 frames, max diff 0.0).
- 2026-11-10: stream 8 done (`gpu_validation.md`). GPU path is correct: all 94 launches of a
  frame run on the GPU inside the captured graph, timings are synchronised, contact sets match
  the CPU kernel by kernel; the only beyond-rounding difference is FMA contraction flipping the
  velocity clamp at |v| = 25 (`reference/scripts/gpu_fma_clamp_repro.py`). GPU-vs-CPU drift over
  full runs is the same size as CPU-vs-perturbed-CPU, including the dress. Warm steady state:
  shirt 7.3 ms/frame (CPU 320), dress 15, jumpsuit 17; shirt end to end 8.6 s vs 83.6 s on CPU.
  Early frames are slow because of O(k^2) work on long stitch springs in
  `create_self_edge_contacts`. Cold compile ~110 s. No Warp changes were needed.
- 2026-11-10: GarmentCode patched for NumPy 2 (uncommitted): `cross_2d` replaces 2-D `np.cross`,
  `np.infty` shim for pyrender, pin dropped. `venv-gc` is on numpy 2.5.4; GPU re-validation passed.
- 2026-11-10: stream 9b done (`batch_validation.md`). 40 random garments on the GPU, seed 20261110:
  31 pass / 9 flagged after GarmentCode's resim (3 are meshing failures that also fail on the
  NumPy 1 reference env), 0 crashes. Per garment, first attempt, warm cache: median 17.4 s,
  p90 58.7 s; steady-state median 10.4 ms/frame. `create_self_edge_contacts` dominates GPU time
  in both the early frames (93-98 %) and steady state on large garments (54-80 %).
- 2026-11-10: stream 9a done: `SETUP.md` + `scripts/setup_amd.sh`, clean from-scratch run passed.
  `local_env.md` sections on missing `warp.sim` are now stale (see `SETUP.md` section 8).
  GarmentCode `data_config.py` patched (uncommitted) to call `warp.init()` directly.
- 2026-11-10: `create_self_edge_contacts` hotspot fixed on Warp branch `edge-contact-opt`: the
  duplicate-edge scan now runs only for accepted contacts (was O(k^2) per thread). Output is
  unchanged: CPU bit-identical to the reference over the full shirt drape. GPU kernel total over
  120 frames: shirt 7116 -> 191 ms, jumpsuit 13206 -> 490 ms; jumpsuit frame 0 5171 -> 60 ms,
  frames 1-39 median 149 -> 17 ms, steady 18.7 -> 14.7 ms. Full GPU drapes (sim only, warm):
  shirt 231 frames at 6.3 ms, dress 784 at 8.8 ms, jumpsuit 464 at 13.2 ms.
- Decided by user: internal use only, so licensing and commit sign-off are not concerns for now.
- Open:
  `ray_hit` panel-assignment strategy unported (unused by GarmentCode).
