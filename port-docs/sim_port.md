# warp.sim / warp.collision port: GarmentCode fork (1.0.0-beta.6) -> Warp 1.13.0+rocm.0

Workstream 6. Branch `ws6-sim-port` in worktree `$WS/warp-ws6`,
based on `garmentcode-rocm`, with `ws5-mesh-query-edge` merged in.
Fork source: NvidiaWarp-GarmentCode `garmentcode_hack` (base `97aba17a1`), read-only.

**Result in one line.** On CPU, the ported sim is **bit-identical to the original beta.6 fork**. That covers every
saved frame of the full-length shirt_mean, dress_pencil and js_mean_all reference runs, a body-smoothing run, and the
final `*_sim.obj` written by GarmentCode's own `run_sim`. GarmentCode itself needed no change.

## Commits

| Commit | What |
|---|---|
| `37d4b1c8d` | Fork's `warp/sim/*` and `warp/collision/panel_assignment.py` added verbatim (pristine baseline for review) |
| `a346e9306` | Mechanical port: private imports, kernel `mat33(vec,vec,vec)` -> `matrix_from_cols`, drop importers |
| `392db3bbf` | Merge `ws5-mesh-query-edge` (`wp.mesh_query_edge`, fork's out-param signature) |
| `13e696373` | Python-scope builtins on tuples/ndarrays (`+ wp.vec3(pos)`, `wp.vec3d` in `add_triangles_stitching`) |
| `b1bc87b1c` | Build `wp.Mesh` with beta.6's BVH layout (median/LBVH, 1 face per leaf) |
| `887b779ee` | Tuple `scale` in `compute_shape_mass` |

Review the port with `git diff 37d4b1c8d HEAD -- warp/sim warp/collision`.
That diff is 13 files, +44/-2093, and 2066 of the deleted lines are the four dropped importer files.

Native rebuild after the merge used the standard command (see Verification). The kernel cache was cleared, because it does
not notice `native/*.h` edits.

## Change categories, with counts

| # | Category | Count | Kind |
|---|---|---|---|
| A | Private-module imports (removed public re-exports) | 13 sites, 4 files | mechanical |
| B | Kernel `wp.mat33(v0,v1,v2)` -> `wp.matrix_from_cols(v0,v1,v2)` | 9 sites, 5 files | mechanical, semantics checked |
| C | Dropped importers (MJCF/URDF/SNU/USD) | 4 files deleted, 4 import lines in `__init__.py` | removal |
| D | Python-scope builtins given explicit Warp types (tuple/ndarray rejected in 1.13) | 4 statements (6 call sites) in 2 functions + 1 | **non-mechanical** |
| E | `wp.Mesh(..., bvh_constructor=..., bvh_leaf_size=1)` | 3 sites, 2 files | **non-mechanical** |
| F | `wp.mesh_query_edge` | 0 call-site changes (ws5 native builtin, merged) | n/a |
| - | Physics / numerics changes | **0** | |

### A. Private imports (13 sites)
- `optimizer.py:8`: `from warp.context import synchronize` -> `from warp import synchronize`.
- `model.py:19-23`: added `from warp._src.types import int_types` and `from warp._src.utils import MeshAdjacency, warn`.
  Uses follow from that:
  - `wp.utils.warn(` -> `warn(` x5 (now 319, 955, 978, 988, 1003);
  - `wp.utils.MeshAdjacency(` -> `MeshAdjacency(` x4 (now 3610, 3712, 3843, 3960);
  - `wp.types.int_types` -> `int_types` x1 (75).
  
  The fork's `MeshAdjacency` is byte-identical to 1.13's `warp._src.utils.MeshAdjacency`. 1.13's `warn` only added a
  keyword-only `once=False` argument.
- `render.py:16`: `from warp.render.utils import ...` -> `from warp._src.render.utils import solidify_mesh, tab10_color_map`.

### B. Kernel matrix-from-vectors (9 sites)
`utils.py:48` (quat_decompose), `utils.py:179` (transform_inertia), `articulation.py:25,55,101,131`,
`integrator_euler.py:657,672`, `integrator_xpbd.py:695`.

In beta.6, the native constructor `mat_t(vec c0, vec c1, vec c2)` fills **columns** (I checked the fork's `native/mat.h`).
So `matrix_from_cols` is the exact replacement; `matrix_from_rows` would compile and silently transpose. The model.py and
inertia.py `wp.mat33(...)` calls take 9 scalars, nested lists or `np.eye(3)`, still work in 1.13, and were left alone.

### C. Dropped importers
`import_mjcf.py`, `import_urdf.py`, `import_snu.py` and `import_usd.py` were removed with `git rm`, and their four lines in
`sim/__init__.py` replaced by a comment. GarmentCode does not use them, and they depended on removed APIs (`wp.vec(...)`,
`wp.utils.*`). `articulation.py` is kept because `__init__` re-exports `eval_fk` / `eval_ik`. Its four Category B sites were
patched and it compiles.

### D. Python-scope builtins: the non-mechanical changes
1.13 raises `TypeError: Built-in functions cannot be called with non-Warp array types` for tuple, list and ndarray
arguments. I checked what beta.6 actually did with these arguments by **running beta.6** (`venv-ref`, the
`warp-ref-cpu` build), not by reading its source.

Beta.6 `Function.__call__` tries each non-generic overload. Its `call_builtin` packs the arguments only when the
element types match exactly: Python float -> float32, a numpy dtype -> the matching Warp scalar.

1. **`model.py:3901`** (`add_cloth_mesh_sewing_spring`, the GarmentCode path) and **`model.py:3684`** (`add_cloth_mesh`):
   ```python
   # before
   p = wp.quat_rotate(rot, wp.vec3(v * scale)) + pos
   # after
   p = wp.quat_rotate(rot, wp.vec3(v * scale)) + wp.vec3(pos)
   ```
   GarmentCode passes `pos=(0.0, 0.0, 0.0)`, a tuple of Python floats. Beta.6 packed it as float32 (verified: the
   result is a `vec3f`), and `wp.vec3(pos)` gives the same float32 values. If a `vec3` is passed, `wp.vec3(vec3)` is
   the identity.

2. **`model.py:3261-3263` and `3285-3288`** (`add_triangles_stitching`, both branches; GarmentCode takes the `orig_lens`
   branch):
   ```python
   # before
   n = wp.normalize(wp.cross(qp, rp)); e1 = wp.normalize(qp)
   # after
   n = wp.normalize(wp.cross(wp.vec3d(qp), wp.vec3d(rp))); e1 = wp.normalize(wp.vec3d(qp))
   ```
   `qp` and `rp` are **float64** ndarrays (from `create_triangle_with_edge_lengths`; `p` is an int64 zero vector).
   Beta.6 matched them to the **`vec3d` (float64) overloads**: running it gives `type(wp.cross(qp, rp)) == vec3d`.
   The scratch port's `wp.vec3(*(q - p))` would therefore have silently downgraded the rest-pose computation to
   float32. With `wp.vec3d`, the output of `(e1, e2)` matches beta.6 **bitwise on 2000 random triangles**
   (`/tmp/claude-1000/ws6/b6/t2.py`). `R = np.array((e1, e2))` stays float64 as before.

3. **`model.py:387`** (`compute_shape_mass`, GEO_MESH/SDF with precomputed inertia):
   `wp.cw_mul(c, scale)` -> `wp.cw_mul(c, wp.vec3(scale))`. Beta.6 accepted a float tuple for `scale` and packed it as
   float32; `wp.vec3(scale)` gives the same values. GarmentCode passes `wp.vec3(1,1,1)` here, so this site was not
   failing on its path; it is fixed for compatibility. Verified bitwise against beta.6 for tuple and vec3 scale
   (`/tmp/claude-1000/ws6/b6/inertia_t.py`).

I audited the other Python-scope builtin calls on the GarmentCode path. They receive Warp types, or forms that still
work in 1.13: `wp.vec3(ndarray)`, `wp.transform(tuple, quat)`, `wp.mat33(np.eye(3))`, `wp.quat_from_axis_angle(vec3, float)`.
All of them ran in the end-to-end runs below.

### E. BVH construction (3 sites): the source of the only numerical divergence

| Site | Mesh |
|---|---|
| `model.py:264` | `sim.Mesh.finalize`: the body collider and the cloth-reference body-part meshes |
| `model.py:4482` | `particle_shape`, the cloth mesh used for self-collision |
| `panel_assignment.py:171` | body mesh for panel assignment |

```python
# before
wp.Mesh(points=..., indices=...)
# after
wp.Mesh(points=..., indices=..., bvh_constructor="median" if <device>.is_cpu else "lbvh", bvh_leaf_size=1)
```
In each case `<device>` is the device of the mesh's own points array or scoped device; nothing is hard-coded.

**Why.** After categories A-D, the CPU port ran but differed from beta.6 already at frame 0 (max 4.0e-4 cm, 1.5e-2 cm by
frame 2).
- Beta.6 built mesh BVHs with `MedianBVHBuilder` on CPU (LBVH on GPU), one primitive per leaf.
- 1.13 defaults to SAH on CPU with 4 primitives per leaf.
- Every mesh query keeps the first minimum it finds (strict `<`), so traversal order decides exact ties: shared
  edge/vertex closest points in `mesh_query_point*`, equal-`t` ray hits, and the summation order of `sign_normal`'s
  angle-weighted normal.

With the beta.6 layout, the CPU port is bit-identical to beta.6 for whole runs, so no other divergence source remains on
this path. I first tested this as a diagnostic, by monkeypatching `wp.Mesh.__init__` in a wrapper
(`/tmp/claude-1000/ws6/diag_bvh.py`), and then committed it at the three sites.

GPU note: beta.6 used LBVH on CUDA, so the port asks for `"lbvh"` with leaf size 1 there. Whether 1.13's LBVH builds the
same tree as beta.6's is not verified, since no beta.6 GPU build is available here.

## Short-circuit `and`/`or` audit (C11)
Beta.6 `emit_BoolOp` evaluated every operand eagerly (`adj.add_bool_op(func, [adj.eval(expr) for expr in node.values])`);
1.13 short-circuits. An AST scan of every `@wp.kernel` / `@wp.func` in `warp/sim/*` and `panel_assignment.py`
(`/tmp/claude-1000/ws6/sc_audit.py`, output in `sc_audit.txt`) found:
- **87 BoolOps, 0 IfExp.**
- 77 have operands that make no calls.
- The 10 with calls are all side-effect free:
  - `wp.abs`: collide.py 208, 1060, 1090, 1098, 1316, 1318;
  - `check_edge_exists`, a read-only loop over `edge_contact_pairs`: collide.py 1693, 1717, 1741;
  - `is_zero`: integrator_xpbd.py 2153.

No atomic, array write or out-parameter appears in any operand. **Short-circuiting cannot change results.** Eager
evaluation in beta.6 could perform reads that 1.13 now skips, but the boolean value is the same.

Related codegen checks:
- **`arr[i] += x`.** Beta.6 lowered this to a non-atomic load-add-store; 1.13 emits `atomic_add`. An AST scan found
  **0** augmented assignments to subscripts in sim kernels/funcs (`/tmp/claude-1000/ws6/aug_audit.py`). All accumulation
  goes through explicit `wp.atomic_add`, so this is not relevant.
- **Name-to-name value copy (`a = b` then `a[i] = ...`).** The scan found **0** component writes to non-parameter
  locals, so this is not relevant.

## Native builtin semantics (beta.6 vs 1.13; summary of a header-by-header diff)
Changes that matter, and how they affect GarmentCode:
1. **`mesh_query_point` (out-param form) sign rule.**
   - Beta.6 `mesh_query_inside` returned -1 only if all 3 axis rays hit back faces (`vote == 3`). 1.13
     `mesh_query_inside_ray_tracing` uses a majority vote (`vote >= 2`). This is confirmed in both headers.
   - **No effect on GarmentCode.** Every site that uses the sign overwrites it with the fork's own Python-level copy
     of the old rule, `warp.sim.utils.mesh_query_inside_` (`vote == 3`). Those sites are
     `reference_shape_drag_constraint` (xpbd:2261) and `eval_reference_shape_drag` (euler:1567).
   - The other two sites ignore the sign: `closest_point_on_reference_shape` (collide:1372) and
     `panel_assignment_closest_point_test` (panel_assignment:317).
   - Anyone writing new code against 1.13's `mesh_query_point` sign should know it now uses a majority vote.
2. **BVH builder and leaf size.** Fixed by Category E.
3. **`hash_grid` binning** uses `floor` instead of truncation. It is used only by particle-particle collisions, which are
   disabled in the default config (`enable_particle_particle_collisions: false`). This is unverified against beta.6
   with them enabled.
4. **`mat * mat` uses `fmaf` (muladd) in 1.13.**
   - The kernels on the GarmentCode path are unaffected; whole runs are bit-identical.
   - It is visible at **Python scope**: `warp.sim.inertia.transform_inertia` (`R @ I @ R^T`) differs from beta.6 in the
     last bit of some entries (e.g. -43.52703094 vs -43.52702713).
   - `transform_inertia` is only reached for shapes attached to dynamic bodies (`_update_body_mass` returns for
     `body == -1`). GarmentCode's body is static, so it never runs on the GarmentCode path. Left unchanged and listed
     under Open issues.
5. **`mesh_query_ray` on HIP** has a separate traversal (nearest child first, early exit). Same closest hit; ties may
   differ. GPU only.

Unchanged (the comparison found the code identical apart from formatting and templating):
- `closest_point_to_triangle`, `intersect_ray_tri_woop`;
- `mesh_query_point_sign_normal` apart from leaf looping;
- `mesh_eval_position` / `_velocity`;
- `mesh_query_aabb` overlap test;
- `closest_point_edge_edge`;
- scalar helpers (`sign(0)=+1`, `step`, `nonzero`, `clamp`, ...);
- `normalize` (`kEps = 0`), `length`, `cross`, `dot`;
- mat33 `inverse` and `determinant`, quaternion / transform helpers;
- the `spatial_vector` layout, CPU atomics, and query stack depth 32.

## Undetermined items from the brief: now determined

| Item | Result |
|---|---|
| `Mesh.refit()` / `replace_mesh_points` | Exercised by the body-smoothing run (below). Bit-identical to beta.6. `refit()` only updates bounds in both versions; the tree topology is fixed at build time. |
| `implicit_laplacian_smoothing` (igl + scipy) | Works in venv-gc (scipy 1.17.1; venv-ref has scipy 1.15.3). The smoothing run is bit-identical to beta.6, cloth frames included. |
| `capture_begin()` with no args | Works on cuda:0 (HIP). It captures on the current device, and the GarmentCode shirt ran 243 frames through graph replay (see GPU section). |
| numpy-argument calls in `inertia.py` | `compute_mesh_inertia` (float64 vertices, `wp.vec3(np.mean(...))`) is bitwise equal to beta.6, solid and hollow. `transform_inertia` works, but see native item 4 (ULP differences, not on the GarmentCode path). |
| kernel `and`/`or` short-circuit | Audited above: no effect. |
| int/float strictness in launches | No failures on any exercised path. All launches in the runs below succeeded. |
| Default device (cuda:0 on this box) | The port follows the fork exactly; nothing is hard-coded. GarmentCode's `Cloth` uses `wp.get_device()` (or `'cpu'`), and `finalize(device=)` scopes arrays. A few fork launches in `inertia.py` / `compute_mesh_inertia` use the current default device, as in the fork, so CPU runs need `wp.set_device('cpu')` before `Cloth()`, as the harness and my driver do. |
| `wp.sim` kernel-scope constants (`wp.sim.GEO_*`) | Resolve once `warp.sim` is imported. **No `warp/__init__.py` change is needed**, and the fork's `warp/__init__.py` did not import sim either. GarmentCode imports `warp.sim` explicitly. Packaging `include = ["warp*"]` already picks up `warp.sim` and `warp.collision`. |

## Corrections to `api_drift.md`
- **C1:** "the real port `warp/__init__.py` must `from . import sim`" is wrong. It is not needed (see above), and the
  fork did not do it either.
- **C5:**
  - "beta.6 auto-converted" needs qualifying. Beta.6 matched overloads by **exact element type**.
  - Float64 ndarrays went to the **float64 (`vec3d`) overloads**. So the suggested fix for `add_triangles_stitching`,
    `wp.vec3(*(q-p))`, is **wrong**: it silently drops to float32. The correct fix is `wp.vec3d(...)`, verified bitwise
    by running beta.6.
  - "Not run: beta.6 itself (no build available)" is out of date. `venv-ref` / `warp-ref-cpu` exists now, and I used
    it for these checks.
- **C5 inertia.py:313-316 [U]:** now determined (see native item 4). `compute_shape_mass` tuple scale also failed in
  1.13 (fixed, Category D3).
- **C13** "Default `bvh_constructor` unchanged [U]" and "Mesh bvh_leaf_size default 4 (already 4 before)" are **wrong**.
  Beta.6 used median/LBVH with **one face per leaf**; 1.13 uses SAH (CPU) with 4. This caused all of the observed
  divergence (Category E).
- **C11:** audited. No behavioural effect.
- **C14:** igl and scipy are installed in venv-gc, and `implicit_laplacian_smoothing` was run and verified. No
  lazy-import change was needed.
- **Part 2, sim/__init__ / C2 table:** `warp.context.runtime` in GarmentCode `data_config.py:301` is **not** wrapped
  safely. It is caught only by `except ImportError`, and 1.13 raises `AttributeError: module 'warp' has no attribute
  'context'` (verified). See Open issues.
- **"Not determined" list:** `Mesh.refit`, `replace_mesh_points`, `implicit_laplacian_smoothing`, the short-circuit audit
  and the data_config question are all resolved above.

## GarmentCode diffs
**None.** GarmentCode is unmodified (`git -C GarmentCode status` is clean apart from pre-existing state).

The CPU runs use a driver script outside the repo, `/tmp/claude-1000/ws6/gc_run.py`. It mirrors
`test_garment_sim.py`'s `__main__` and differs in only two ways:
- it calls `wp.init(); wp.set_device(<dev>)` first;
- it writes output under /tmp instead of `./Logs/`.

It also has an optional, environment-gated `GC_STATIC_COPY=1` shim, described below.

## Verification

Environment (all commands):
```bash
export ROCM_PATH=/home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel LD_LIBRARY_PATH=$ROCM_PATH/lib
export PYTHONPATH=$WS/warp-ws6 WARP_CACHE_PATH=/tmp/claude-1000/warp-cache-ws6 \
       MPLCONFIGDIR=/tmp/claude-1000/mpl PYTHONDONTWRITEBYTECODE=1
PY=$WS/venv-gc/bin/python     # (= /tmp/claude-1000/ws6/env.sh)
R=$WS/reference
```
Native build (after merging ws5):
```bash
cd warp-ws6; export PATH=/home/carl/py/bin:$PATH; python build_lib.py --rocm-path=$ROCM_PATH --hip-arch=gfx1151
```
Then clear the kernel cache.

### 1. Import and compile every kernel on CPU
`WARP_CACHE_PATH=/tmp/claude-1000/warp-cache-ws6-fresh $PY /tmp/claude-1000/ws6/compile_all.py cpu`. This runs a
forced `wp.load_module` per module into an empty cache.

```
OK   warp.sim.utils: kernels=2 funcs=13         OK   warp.sim.integrator_euler: kernels=17 funcs=2
OK   warp.sim.inertia: kernels=2 funcs=1        OK   warp.sim.integrator_xpbd: kernels=20 funcs=7
OK   warp.sim.particles: kernels=1 funcs=1      OK   warp.sim.articulation: kernels=2 funcs=5
OK   warp.sim.model: kernels=0 funcs=0          OK   warp.sim.render: kernels=1 funcs=0
OK   warp.sim.collide: kernels=11 funcs=29      OK   warp.collision.panel_assignment: kernels=2 funcs=0
OK   warp.sim.optimizer: kernels=3 funcs=0      ALL OK
```
That is 61 kernels and 58 functions, matching the `@wp.kernel` / `@wp.func` counts per file. There are **zero Warp
warnings**.

The only other output is one Python `ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/wp_pch_...'>`.
It comes from Warp's own precompiled-header temp dir being cleaned up at interpreter exit, and is shown only because
the script sets `warnings.simplefilter("always")`; ResourceWarning is hidden by default. It does not come from sim code.

### 2. Reference comparison (ws3 harness), CPU
For each garment:
```bash
$PY $R/scripts/dump_harness.py run --garmentcode-dir .../GarmentCode --inputs $R/data/inputs/<g> --device cpu \
    --out /tmp/claude-1000/ws6/runs/<g>_cpu1 --save-every 2 --static-mode copy
```
Bitwise check: `$PY /tmp/claude-1000/ws6/bitcmp.py $R/data/<g>/cpu_run1 runs/<g>_cpu1`.

Tolerance check:
```bash
$PY $R/scripts/compare_runs.py views/<g>_ref views/<g>_port -q --tol-max 0 --tol-frames 0 \
    --tol-self-int 0 --tol-body-int 0 --require-same-stop
```
The `views/*` directories are symlink copies without `f000000_init.npy`; see the compare_runs bug under Open issues.

| Garment | Frames | Stop | Self / body intersections | Port s/frame (ref) | Max vertex diff, every saved frame |
|---|---|---|---|---|---|
| shirt_mean | 247 = 247 | static_equilibrium | 13 / 0 = 13 / 0 | 0.34 (0.44) | **0.0** (125 frames + init + final_verts) |
| dress_pencil | 870 = 870 | static_equilibrium | 434 / 0 = 434 / 0 | 0.63 (0.80) | **0.0** (437 frames) |
| js_mean_all | 789 = 789 | static_equilibrium | 0 / 0 = 0 / 0 | 0.98 (1.24) | **0.0** (395 frames + init + final_verts) |

All three runs are bit-identical to the beta.6 fork. `compare_runs.py` exits with OK at **zero** tolerance.

Body smoothing is off in the default config, so I tested it separately. I used a copy of the shirt inputs with
`enable_body_smoothing: true`, `smoothing_recover_start_frame: 20` and `smoothing_num_steps: 10`
(`/tmp/claude-1000/ws6/inputs/shirt_smooth`), with `--max-frames 60 --save-every 1`, on **both** beta.6 (venv-ref)
and the port.
- Both give 60 frames, 25 self / 27 body intersections.
- **All 61 saved frames are bit-identical.**
- The smoothed run differs from the unsmoothed one by 0.15 to 0.8 cm, so smoothing really is active. This exercises
  `implicit_laplacian_smoothing`, `replace_mesh_points` and `Mesh.refit()`.

### 3. GarmentCode end to end, real `run_sim` path, CPU
Run with cwd = GarmentCode: `$PY /tmp/claude-1000/ws6/gc_run.py ./assets/Patterns/<g>_specification.json <out> cpu`.

- **As-is, shirt_mean.**
  - 11 frames, stopped by "Static with 0 non-static vertices", 0.74 s/frame, 44 body / 92 self intersections. Outputs
    were saved and rendered, and there was no "crashed with".
  - Beta.6 run through the same driver gives exactly the same result: 11 frames, 44 / 92, and a **byte-identical
    `shirt_mean_sim.obj`**.
  - The early stop is GarmentCode's CPU aliasing issue, not a port problem. On CPU, `wp.array.numpy()` returns a view,
    so `last_verts` and `current_verts` alias and `is_static()` is trivially true at the first check (frame 10).
  - The render shows the undraped, box-like initial shirt, as expected for 11 frames.
- **With `GC_STATIC_COPY=1`.** The driver wraps `Cloth.run_frame` to copy `current_verts` after each frame, the same as
  the harness's `--static-mode copy`; GarmentCode is not modified.
  - **shirt_mean:** 247 frames, static equilibrium, 0.36 s/frame, **13 self / 0 body** intersections, no fails, outputs
    and renders saved.
  - The saved `shirt_mean_sim.obj` vertices equal the beta.6 reference `final_verts.npy` exactly (max diff 0.0).
  - The rendered front image shows the shirt draped correctly on the body, with sleeves on the arms and no visible
    artifacts.
  - **dress_pencil:**
    - 870 frames, stopped by "Static with 215 non-static vertices out of 14425".
    - 0.62 s/frame (sim 540 s, pipeline 9 min 0 s), render 0.66 s, no "crashed".
    - 0 body / 434 self intersections. GarmentCode marks this as a fail (`cloth_self_intersection`), and the beta.6
      reference has exactly the same 434. So this is the garment's own behaviour, not a port regression.
    - `dress_pencil_sim.obj` vertices equal the reference `final_verts.npy` exactly (max diff 0.0).
    - The front render shows the dress draped on the body: bodice on the torso, pencil skirt over the legs, held at
      the waist by attachment.
    - Output: `gc_out_copy/dress_pencil_261110-13-07-21`.
- Outputs are in `/tmp/claude-1000/ws6/gc_out*/<garment>_<timestamp>/`: `*_sim.obj`, `*_render_front.png`,
  `*_render_back.png`, `sim_props.yaml` with stats, textures, and so on.

**Feature exercise** was checked with `/tmp/claude-1000/ws6/feature_probe.py`, which builds the real `Cloth`, counts
`wp.launch` per kernel, and inspects model arrays over 40 frames.

| Feature | shirt_mean | dress_pencil |
|---|---|---|
| Panel assignment (`closest`) | labels body 4840 / left_arm 1383 / right_arm 1422 / unassigned 345 | body 6087 / legs 7864 / unassigned 474 |
| Attachment | off: the shirt has no attachment labels (GarmentCode warning, same in beta.6) | **on**: 113 vertices, `attachment_constraint` launched 400 times |
| Body-part (reference) drag | 2 drag label pairs; up to 188 cloth-cloth and 23 body particles flagged per frame | up to 20 body particles flagged |
| Body collision filters | `face_filter` 14506 non-zero entries; particle filter id 1 on 4978 particles | (same mechanism) |
| Self-collision | up to 52058 point-triangle and 1.96 M edge contact candidates per frame; `find_intersecting_particles` (uses `mesh_query_edge`) flagged up to 241 particles | kernels launched every frame/substep |
| Body collision | up to 1316 soft contacts per frame | `create_soft_contacts` / `solve_particle_shape_contacts` launched |

Kernels launched in 40 dress frames (10 substeps per frame):
- 400 each: `integrate_particles`, `solve_springs`, `bending_constraint`, `solve_particle_triangle_self_contacts`,
  `solve_edge_edge_self_contact`, `solve_particle_shape_contacts`, `reference_shape_drag_constraint`,
  `attachment_constraint`, `replace_mesh_points`, `apply_particle_deltas`;
- 40 each: `create_soft_contacts`, `create_self_point_triangle_contacts`, `create_self_edge_contacts`,
  `find_intersecting_particles`;
- plus `panel_assignment_closest_point_test`, `compute_solid_mesh_inertia` and the counting kernels.

### 4. GPU stretch: shirt_mean on cuda:0 (graph capture)
`WARP_CACHE_PATH=/tmp/claude-1000/warp-cache-ws6-gpu timeout 1200 $PY gc_run.py ./assets/Patterns/shirt_mean_specification.json /tmp/claude-1000/ws6/gc_out_gpu cuda:0`

**It works.** It completed with rc=0, and graph capture was used. GarmentCode sets `sim_use_graph = wp.get_device().is_cuda`,
then calls `wp.capture_begin()` with no arguments, `capture_end()`, and `capture_launch(graph)` every frame.

| | |
|---|---|
| Frames | 243 |
| Stop reason | "Static with 119 non-static vertices out of 7990" (GPU arrays: no aliasing, so this is the real static check) |
| Intersections | 0 body / 19 self, `is fail: False`, no crash |
| Speed | 0.53 s/frame (sim 127 s, pipeline 2 min 8 s) |
| Render | 0.61 s |

Output is in `/tmp/claude-1000/ws6/gc_out_gpu/shirt_mean_261110-13-09-35/`. The front render shows the shirt draped
correctly on the body.

**First-run JIT cost.** With a cold cache the first run takes about 2 min of HIPRTC compiles, of which
`warp.sim.collide` alone is 102 s. HIPRTC also prints one "does not support precompiled headers" warning per module;
this is a Warp/ROCm build message, not a sim issue.

**Comparison with the CPU result.** The final vertices differ from the CPU/beta.6 reference `final_verts`:
- max 2.22 cm, mean 0.37 cm, p99 1.28 cm (script `/tmp/claude-1000/ws6/gpucmp.py`);
- intersections are 19 self against 13 on CPU.

Over 243 frames of contact-rich cloth this is expected for a different device: different float ordering, atomics, and
LBVH instead of median. It is **not** a fidelity validation of the GPU path. Frame-by-frame GPU-vs-reference
comparison is ws3/ws-other territory and was not done here.

## Open issues
1. **GarmentCode `data_config.py:301` `add_sys_info()`.** It uses `warp.context.runtime`, guarded only by
   `except ImportError`. On 1.13 this raises `AttributeError`, which would crash `pattern_data_sim.py:105` and
   `pattern_fitter.py:227`; it is not on the `test_garment_sim.py` path. There are two minimal fixes:
   - GarmentCode: change the guard to `except (ImportError, AttributeError)`;
   - Warp: re-expose `warp.context`.
   
   Neither was made: no GarmentCode change was needed for the task paths, and re-exposing `warp.context` would be a
   change outside `warp/sim`.
2. **CPU `is_static()` aliasing in GarmentCode.** On CPU, GarmentCode's own run stops at frame 10. This is identical
   in beta.6, so it is pre-existing and not a port issue. Left as is, as instructed.
3. **`transform_inertia` at Python scope** differs from beta.6 by 1 ULP in some entries, because 1.13 native mat*mat
   uses FMA. This is not on the GarmentCode path (static body). Fixing it would need either a native change or
   rewriting the numerics in Python, and I did neither.
4. **`panel_assignment` `ray_hit` strategy** (panel_assignment.py:256) calls `wp.cross` / `wp.normalize` on ndarrays.
   It fails in 1.13 and was not patched. GarmentCode uses `strategy='closest'`, so it is not on the path. A fix would
   have to establish the beta.6 overload dtype first, as in D2.
5. **`hash_grid` floor-vs-truncation** (native item 3) is unverified against beta.6 with
   `enable_particle_particle_collisions: true`.
6. **ws3 `compare_runs.py` bug.** `load_frames` keys files by `re.match(r"f(\d+)")`, so `f000000_init.npy` and
   `f000000.npy` collide on key 0. Glob order decides which one is used, so it can compare the init frame of one run
   with frame 0 of the other. On shirt_mean this gave a spurious 0.417 cm "frame 0" difference between bit-identical
   runs. Workaround: compare symlink views without `_init` files (`/tmp/claude-1000/ws6/noinit_view.sh`). Suggested fix:
   regex `f(\d+)\.npy$`. The script belongs to ws3, so I did not change it.
7. **Beta.6 flake.** One beta.6 `run_sim` launch segfaulted at frame 0 and passed on rerun. This matches the known
   beta.6 kernel-cache race in reference/README.md. It is not a port issue.
8. **Commit trailers.** Commits carry the brief's `Co-Authored-By: Claude Fable 5.2` trailer. Warp's AGENTS.md asks for
   `--signoff` (DCO); that was not added, because the brief specified the trailer format.
