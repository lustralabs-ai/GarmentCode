# Warp (AMD ROCm port, 1.13.0+rocm.0) unit test triage on gfx1151

Date 2026-11-10. Worktree `warp-amd-probe` (branch `garmentcode-rocm`). Device `cuda:0` = AMD Radeon 8060S (gfx1151, ROCm 7.15 runtime reported by Warp).

## How tests were run

AGENTS.md says: `uv run --extra dev -m warp.tests -s autodetect` (unittest-based, `warp/_src/thirdparty/unittest_parallel.py`). No CI workflow runs the suite (only security scans; `docker/rocm/Dockerfile` has no test command).
I instead ran **one `python -m unittest -v warp/tests/<file>` process per test file** (isolates hangs/crashes, 900 s timeout each, 4 files in parallel), so every test ran on both `cpu` and `cuda:0` (test names end `_cpu` / `_cuda_0`). Scripts: `/tmp/claude-1000/ws1/runone.sh`, `order.txt`, `order2.txt`; logs `/tmp/claude-1000/ws1/logs{,2}/`.

```
source /tmp/claude-1000/ws1/env.sh   # PATH, ROCM_PATH, LD_LIBRARY_PATH, PYTHONPATH
export WARP_CACHE_PATH=/tmp/claude-1000/warp-cache-ws1
cd warp-amd-probe && timeout 900 .venv/bin/python -m unittest -v warp/tests/<file>.py
```
Pass 1: all top-level `warp/tests/test_*.py`, `geometry/`, `cuda/` (120 files). Pass 2 (after `uv pip install usd-core pillow matplotlib` into the probe `.venv`): geometry files and test_coloring/test_examples that had skipped for usd-core, plus `matrix/`, `tile/`, `fem/`, `interop/`. Pass-2 logs override pass-1 for the same file. Order was prioritised (mesh, bvh, mesh_query_*, hash grid, atomics, array, copy, graph, launch, math...) but everything finished; **no hangs, no timeouts, no crashes, no GPU resets visible** (dmesg is not readable without root; `journalctl -k` showed no amdgpu/page-fault lines).

Note: `test_mat.py` at top level does not exist (it is `matrix/test_mat.py`); the first-pass ERROR for it is my mistake and was rerun. `interop/test_{jax,paddle,torch}.py` and `tile/test_tile_mlp.py` ran 0 tests (packages not installed / skipped at import) -- not run.

## Totals (parsed from verbose output, `parse2.py`; 4 of these files were rerun, counts use the rerun)

| Device | pass | fail | error | skip | total |
|---|---|---|---|---|---|
| cpu | 2064 | 0 | 0 | 4 | 2068 |
| cuda:0 (HIP) | 2115 | 34 | 1 | 50 | 2200 |
| device-agnostic names (run on default device = cuda:0, no suffix) | 1497 | 3 | 1 | 22 | 1523 |

(The "device-agnostic" bucket mostly are tests that internally loop over devices or run examples; the 4 failures there are fem tests and one import error, see table. Parsing is regex-based; treat totals as +/- a few.)
CPU: zero failures. So every failure below is (a) HIP-specific unless stated.

## Classified failures

| Test(s) | Class | Detail | Sim-relevant? |
|---|---|---|---|
| tile/test_tile_reduce: 16 tests (sum, min, max, argmin/argmax, scans, axis_tier2/3, custom_struct, matrix, simt, ones, sum_to_shared) | (a) | **silently wrong numerics**, e.g. `tile_reduce_sum 24.65 != 49.00`, scan results wrong for 97% of elements. Likely wave64/wave32 assumption in tile reduce/scan | No (tiles) |
| tile/test_tile_fused_ops: test_tile_dot_basic, test_tile_dot_shared_shared | (a) | `192 != 384`, `-1056 != -64` | No |
| tile/test_tile_sort: test_tile_sort | (a) | key sorting mismatch int32 | No |
| test_examples: tile_convolution, tile_filtering, tile_fft | (a) | wrong output vs numpy (convolution `0.3047 vs 0.0024`) | No |
| test_examples: fluid_checkpoint | (a) | ERROR, subprocess run of example failed (not dug into; it is an autodiff example) | No |
| test_linear_solvers: test_cg, test_cr, test_bicgstab, test_gmres (cuda_0) | (a) | solve residual `7.25e6 > 3e-7` (garbage result). Warp `array_inner`/`array_sum` themselves are correct on GPU (checked, `repro/reduce.py`), so suspect sparse matvec | No (sparse) |
| test_sparse: test_bsr_multiply_deep_2_2, _30_30 (cuda_0) | (a) | BSR matmul wrong (`104.99 vs 200.84`) | No |
| fem/test_fem_integrate: tensor/vector_divergence_theorem_cuda_0 | (a) | `-0.220 != -0.070` | No |
| fem/test_fem_quadrature: gimp / nodal / particle quadrature | (a)? | `0.03235 != 0.0625`; also "Unexpected output" (stdout noise). Device not in name; probably GPU run, not verified on CPU | No |
| fem/test_fem_shape: test_cube_shape_functions | (a)? | "Unexpected output" assertion (stdout/warning text) | No |
| test_map: test_array_ops_cuda_0 | (a) | float32 `a**b` 1-2e-6 relative error (e.g. 3^19: 3486786560 vs 3486784512), test tolerance 1.5e-7. Fast `powf` on HIP | Marginal: only if sim uses `wp.pow`/`**` on large exponents; ULP-level |
| matrix/test_mat: test_inverse_float16_cuda_0 | (a) | fp16 inverse off by 0.06 (`-31.375 vs -31.3125`) | No (fp16) |
| cuda/test_ipc: test_ipc_multiprocess_write, test_ipc_event_missing_interprocess_flag | (a) | IPC unsupported/different behaviour | No |
| test_grad_debug.py | (c) | first pass: `ModuleNotFoundError: matplotlib`; after installing it the script runs but prints `NaN` gradient check failing for `vec_length_kernel` (at vec(0,0,0)-like input, `nan` AD) and still exits 0 / runs 0 unittest tests. Not triaged (autodiff) | No |
| test_examples: particle_repulsion, tile_mlp (pass 1) | (c) | missing Pillow; passed/rerun after install | - |
| 27 tests (mesh_query_*, coloring, examples) | (c) | skipped "Requires usd-core"; **rerun with usd-core: all geometry/mesh_query_* tests passed** | - |

## Skips (d, expected) by reason
usd-core 27 (resolved on rerun), "No suitable devices" 26, "Conditional graph nodes not supported" 16 (cuda/test_conditional_captures), MathDx 8+1, multi-GPU/peer 11, ml_dtypes 5, arch<90/<100 3, rmm 1, pillow 1.
HIP-specific skips written by AMD that are worth knowing about:
- `test_bvh_ray_query... (issue #288)`: skipped, "Clang optimization bug with axis-aligned single-AABB ray query (1/0=inf reciprocal) when several BVH kernels share an HIPRTC module". Potentially relevant to `mesh_query_ray` with axis-aligned rays; mesh_query_ray tests themselves passed.
- `test_bvh` / `test_grouped_bvh` "BVH rebuild inside a captured graph": skipped on HIP (radix sort not graph-capturable).
- event timing / cross-graph event sync inside captured graphs: not supported on HIP.

## Sim-relevant shortlist

**No failing test touches an API the cloth sim uses.** All of the following passed on cuda:0 with 0 failures (first pass, no usd needed unless noted):
mesh (13 tests, 0 fail), bvh (14, 2 HIP skips above), mesh_query_point (9, after usd-core), mesh_query_ray (9), mesh_query_aabb (15), hash_grid (19), atomic (161), array (138), copy (14, 3 skips), graph (12), launch (32), closest_point_edge_edge (20), vec (89), mat (matrix/ 583 tests; one fp16 failure only), quat (167), spatial (201), codegen (116), conditional (50), print/types etc.

Extra direct checks of graph-capture paths the sim could use (`/tmp/claude-1000/ws1/repro/capture_build.py`, verified by running):
- `mesh.refit()` in capture + replay x2: OK
- `HashGrid.build()` in capture + replay x2: OK
- `Bvh.refit()` in capture: OK
- `Bvh.rebuild()` in capture: **FAIL**, `CUDA graph capture failed. Warp CUDA error 901 ... (wp_free_device_async)` (matches AMD's known limitation; sim should use refit, not rebuild, inside graphs). Only matters if GarmentCode rebuilds meshes inside captured graphs.
Caveat: capture_build.py checks that capture/launch does not error; correctness of resulting grid/BVH inside the graph was not compared against CPU.

## Surprises
- CPU is perfectly clean (2068 tests, 0 fail), so GPU failures are all genuine HIP/gfx1151 issues, not port-wide bugs.
- Tile reduce/scan/dot/sort and BSR/CG failures are silent wrong answers (no error raised) -- worth remembering if anyone later enables tile or sparse features.
- test_examples takes ~13 min on GPU (fem examples ~9 min); fem files ~20 min total.

## Not run / unverified
- jax/torch/paddle interop (packages absent), tile_mlp (needs extra deps), MathDx tests, multi-GPU.
- fem_quadrature/fem_shape CPU behaviour not separately checked.
- dmesg inaccessible; no GPU reset evidence found in journalctl (`-k`, last 3 h).
- Root causes of the failing GPU tests were not investigated (task scope).
- The probe `.venv` now additionally contains usd-core, pillow, matplotlib (and deps).
