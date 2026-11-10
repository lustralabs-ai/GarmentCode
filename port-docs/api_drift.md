# Warp API drift: 1.0.0-beta.6 (fork, 97aba17a1) -> 1.13.0+rocm.0

Scope: fork `warp/sim/*` and `warp/collision/panel_assignment.py`. Research only.
Legend: [V] verified (ran it on 1.13 CPU, or changelog line / commit cited), [U] unverified.
Classes: HARD = error, SILENT = compiles but behaves differently, DEPR = warning only.
Sources: 1.13 changelog extract (/tmp/claude-1000/cl.md, line refs "cl:N"), upstream v1.9.1 (last tag with warp.sim),
scratch port in /tmp/claude-1000/ws2/run/fsim (patched copy that runs 5 XPBD frames of cloth on a sphere, CPU, no NaN) [V].
Not run: beta.6 itself (no build available), so "beta.6 behaviour" is from source/changelog [U where noted].
Default device on this box is cuda:0; CPU tests need `wp.set_device('cpu')`. Fork launches do not pass `device=`.

## Part 1. Change catalogue

### C1. warp.sim removed (1.10; deprecated 1.8/1.9) -- HARD [V cl:~651, import failed]
- beta.6: `import warp.sim`, `wp.sim.JOINT_*`, `wp.sim.GEO_*` inside kernels.
- 1.13: no `warp.sim`. Kernels referencing `wp.sim.X` fail with "`sim` is not an attribute of warp".
- Fix used in scratch: package lives at `warp/sim` again and is registered: `sys.modules['warp.sim']=pkg; wp.sim=pkg`
  (in the real port `warp/__init__.py` must `from . import sim`). Kernel-scope `wp.sim.*` constants resolve via the attribute. [V]
```python
# before/after: nothing changes in the sim source IF warp/sim is shipped as a package inside warp and imported in warp/__init__.py
```

### C2. Private-API shim layer removed (1.11 added shims; 1.13 removed, GH-1352) -- HARD [V]
Public sources moved to `warp/_src/`. Hits:
| beta.6 form | 1.13 replacement |
|---|---|
| `warp.context.synchronize` (optimizer.py:8) | `wp.synchronize` |
| `warp.context.runtime` (GarmentCode data_config.py:301-307) | no public equivalent; use `wp.is_cuda_available()` / just `wp.init()` [U whether app wraps in try/except] |
| `warp.render.utils` (sim/render.py:16: solidify_mesh, tab10_color_map) | `warp._src.render.utils` (no public replacement) [V] |
| `wp.utils.warn` (model.py x5) | `warp._src.utils.warn` or `warnings.warn` [V AttributeError] |
| `wp.utils.MeshAdjacency` (model.py:3601,3703,3834,3951) | `warp._src.utils.MeshAdjacency` [V] |
| `wp.types.int_types` (model.py:70) | `warp._src.types.int_types` or local tuple of wp.int8..uint64 [V] |
| `warp.torch`, `warp.mat`/`warp.vec` proxies | removed; unused here |
Public `warp.utils` now exports only: AllocatorRmm, GraphColoringAlgorithm, array_cast, array_inner, array_scan, array_sum,
create_warp_function, graph_coloring_*, radix_sort_pairs, runlength_encode, segmented_sort_pairs, (+ few). `wp.utils.array_inner` still OK. [V]

### C3. Kernel `wp.mat33(vec,vec,vec)` / `wp.matrix(vectors)` removed (GH-1179, 1.12, cl:~312) -- HARD at kernel scope; SILENT trap in the fix [V]
- beta.6 kernel scope: `wp.mat33(v0, v1, v2)` built a matrix with the vectors as COLUMNS. Upstream replaced with `wp.matrix_from_cols` (commit 0b3e4c7bb). [V]
- 1.13 kernel: error "Passing vectors to `wp.matrix()` isn't supported, use `wp.matrix_from_rows()` or `wp.matrix_from_cols()` instead." [V]
- Python scope in 1.13: `wp.mat33(vec3,vec3,vec3)` still works and gives ROWS (m[0,1]==2.0 for first vec (1,2,3)); deprecated GH-981. [V]
- DANGER: the correct kernel replacement is `matrix_from_cols`. `matrix_from_rows` compiles and silently transposes. `matrix_from_cols(vec3(1,2,3),..)[0,1]==4.0`, `matrix_from_rows[0,1]==2.0`. [V]
- Sites (all kernel scope; patched in scratch with sed): utils.py:48,179; articulation.py:25,55,101,131; integrator_euler.py:657,672; integrator_xpbd.py:695. (8 sites + any inertia/model; model/inertia sites are Python scalars/numpy forms and still work [V]).
```python
# before (kernel)   R = wp.mat33(a, b, c)            # columns
# after  (kernel)   R = wp.matrix_from_cols(a, b, c) # columns, NOT matrix_from_rows
```
Scalar forms `wp.mat33(9 floats)`, `wp.mat33(np.eye(3))`, nested lists, `wp.mat22(np)` still fine at Python scope. [V]

### C4. `wp.mesh_query_edge` missing -- HARD [V]
GarmentCode-only native builtin (fork warp/builtins.py ~1829, native/mesh.h +107 lines, exports.h). In 1.13: "Could not find function wp.mesh_query_edge" at collide.py:1420 and :1481. Workstream 5 re-implements it. See Part 3 for signature.

### C5. Python-scope builtins reject tuple / list / ndarray (1.10) -- HARD [V ran]
- beta.6: `wp.cross(np_a, np_b)`, `wp.vec3 + tuple`, `wp.transform_point(X, tuple)`, etc. auto-converted.
- 1.13: raise TypeError/RuntimeError ("Couldn't find a function 'cross' compatible with the arguments 'ndarray, ndarray'"). Verified failing: `vec3 + tuple|ndarray`, `cross`, `normalize`, `quat_from_axis_angle` on ndarray, `transform_point(X,tuple)`, `quat_rotate(q,tuple)`, `cw_mul(v,tuple)`.
- Still working: `np + vec3` (numpy side), `wp.vec3(tuple|np|list|*list)`, `wp.quat(tuple)`, `wp.transform(tuple,tuple)`, `wp.transform(np,np)`, `wp.transform(7 floats)`, `wp.spatial_vector(6 floats)`. [V]
- FAILS: `wp.spatial_vector(vec3, vec3)` at Python scope ("expected 6 elements, got 2"). Use 6 floats. [V]
- Sites (Python scope, from AST scan, t_ast.py):
  - model.py: cross 3094,3096,3255,3257,3278,3280,3408,3409; normalize x13; cw_mul 381; inverse 1507,4349; mul 2615,2617,2682,2684,2749,2751; quat_from_axis_angle 2434; quat_identity 4336; quat_rotate 3550,3675,3773,3892,4061,4142,4246; spatial_vector 1512; transform_inverse 2197.
  - Confirmed actual failures needing patch: model.py:3892 (`quat_rotate(...) + pos` with tuple pos), model.py:3253-3257 and 3276-3280 (add_triangles_stitching, ndarray cross/normalize). Other sites received wp types and ran. Vectorised `add_triangles` already uses numpy; do NOT wrap it.
  - panel_assignment.py:256 `ray_hit` strategy (cross/normalize on ndarray) FAILS; 'closest' (what GarmentCode uses) works. 174 transform_inverse/transform_multiply fine.
  - inertia.py:313-316 dot/outer/quat_to_matrix/transpose with numpy args in `transform_inertia` [U: not exercised].
  - render.py:151,332 `transform_expand` (exists as Python helper warp/_src/utils.py:77), :334 transform_point.
- GarmentCode app side: passes tuples for `pos`/`vel` into add_particle/add_shape_mesh paths -> hits model.py:3892.
```python
# before: pos = wp.quat_rotate(q, p_tuple) + pos_tuple
# after : pos = wp.quat_rotate(q, wp.vec3(*p)) + wp.vec3(*pos_tuple)
```

### C6. `wp.select(cond, f, t)` removed (deprecated 1.7, removed 1.10, cl:~651) -- HARD but SILENT-on-rename [V]
- 1.13: `wp.select` raises "wp.select() has been removed. Use wp.where(cond, value_if_true, value_if_false)". Argument order is SWAPPED vs old select(cond, if_false, if_true).
- Fork has ZERO uses (grep). Warning for anyone adding one.
```python
# before: wp.select(c, a, b)  # a if c is False
# after : wp.where(c, b, a)
```

### C7. `wp.array(length=, owner=)` removed (1.8.0, cl:~942) -- HARD [V changelog]
Fork has no array uses of them (`length=` grep hits are plane-width params). `wp.array(np, dtype, device, copy=False)` as used by GarmentCode works. [V on CPU: array creation in e2e]

### C8. Matrix from vectors at host: `wp.matrix(pos,quat,scale)` removed (1.10, cl:652) -- HARD, unused here [V changelog]; use `wp.transform_compose`.

### C9. Kernel launch argument checking -- mixed
- vec3 param accepts np.array, tuple, list, wp.vec3. [V]
- Scalar passed to a vec3 param: DeprecationWarning (deprecated 1.12). SILENT-ish. [V]
- float passed to an int param: HARD error. int to float OK. Wrong arg count: error. [V]
- Numpy array to a `wp.array` param works via implicit conversion. [V]
- Strict arg matching for user-function calls from Python scope (1.9.1). [U: not triggered]

### C10. Float16 / float64 return types (1.12, `wp.config.legacy_scalar_return_types`, cl:~329) -- SILENT, not hit [U]
Indexing vec/mat and builtins now return Warp scalar types for non-native scalars. Fork is float32-only. Matters if anyone uses float64 arrays.

### C11. Short-circuit `and`/`or` in kernels (1.13, GH-1329) -- SILENT [V changelog, behaviour U]
Side effects (atomic_add, function calls with writes) on the right of `and`/`or` may now be skipped. Audit kernels with `a and f(x)`/`atomic_*` on RHS. [U not audited]

### C12. Integer semantics -- UNCHANGED [V on 1.13, beta.6 U]
Kernel `-7/2=-3`, `-7%2=-1`, `-7//2=-3` (C truncation); op mapping identical in both codegens. Fork uses `//`,`%` only on non-negative ints.

### C13. Other 1.10-1.13 changes (no hit, or minor)
- 1.13 `isfinite/isnan/isinf` no longer accept ints [V changelog]; 1.13 `wp.load_module` raises when module has no kernels; 1.13 `tape.backward` zeroing (sim does not use tape).
- 1.10 `wp.Mesh(bvh_leaf_size)`: default 4 for Mesh (already 4 before) -> unchanged; `wp.Bvh` default 1. 1.13 adds cubql constructor (experimental, ray only). Default `bvh_constructor` unchanged [U].
- 1.10 `wp.transform()` q defaults to identity (kernel scope). 1.7 `wp.matmul`/CUTLASS removed (unused).
- 1.12 any int type can index vec/mat; 1.12 typed composite constructors allowed.
- 1.13 `UsdRenderer.update_body_transforms` deprecated; render signatures otherwise compatible; new params `render_arrow`, `smooth_shading`, `visible`, `as_spheres`, `plane` in render_ground. pxr import lazy. [V]
- 1.9: query types lost `_t` suffix: `wp.MeshQueryPoint/Ray/AABB`, `wp.HashGridQuery`. Fork does not name them (uses out-param/struct returns). [V grep not needed; compiled]
- `wp.is_tensor` does not exist in 1.13 (nor, by grep, in beta.6): `Model.tensors()` at model.py:943 dead/latent [U].
- `wp.leaky_min` only in commented code. `wp.vec(len,dtype)` (import_mjcf.py:159,161) gone from `wp` namespace; file is droppable.
- No changelog entry found for `wp.printf`, `ScopedTimer`, `capture_*`, `get_device` breaks; all exist: `wp.ScopedTimer`, `wp.capture_begin/end/launch`, `wp.ScopedCapture`, `wp.get_device`, `wp.printf(fmt,*args)`. [V exist; `capture_begin()` no-arg and graph path NOT run, GPU left alone: U]
- Fork file headers carry the pre-Apache NVIDIA licence notice ("strictly prohibited"): see PLAN.md licensing question.

### C14. External dependencies imported at module top
`warp/sim/utils.py` imports `igl` and `scipy` (implicit_laplacian_smoothing: `igl.cotmatrix`, `igl.massmatrix`, `scipy.sparse.linalg.spsolve`). Probe venv has neither; scratch used empty stubs. Port must either add the deps or lazy-import them. `implicit_laplacian_smoothing` itself was NOT run [U].

## Part 2. Per-file inventory

Counts: grep over fork pristine copy (/tmp/claude-1000/ws2/fork/warp). H = hard-error sites needing a patch for GarmentCode path.

| File (lines) | Entries hit | Detail / counts |
|---|---|---|
| sim/__init__.py (50) | C1, C2 | Imports 4 importers (mjcf, urdf, snu, usd) -> drop those lines. Must be importable as `warp.sim`. |
| sim/model.py (4726) | C2 x10, C5 x ~7 sites patched, C7 (none) | `wp.utils.warn` x5, `wp.utils.MeshAdjacency` x4 (3601,3703,3834,3951), `wp.types.int_types` x1 (70). Patch 3892 (`+ wp.vec3(*pos)`), 3253,3254,3276,3277 (`wp.vec3(*(q-p))`, `wp.vec3(*(r-p))`). `wp.Mesh(points=,velocities=,indices=)` at 259 and 4472; `wp.HashGrid(128,128,128)` 4468; `wp.array(` 103; `wp.zeros` 45; `wp.constant` 23; `get_device` x2; `wp.mat33(` x6 are scalar/numpy forms (OK). `wp.is_tensor` at 943 latent. |
| sim/collide.py (2113) | C1 (`wp.sim.` x51, kernel constants), C4 x2 | `mesh_query_edge` 1420,1481 (stub in scratch); `mesh_query_point` x9, `mesh_query_ray` x2, `mesh_get` x16, `mesh_eval` x10, `wp.printf` x2, `wp.launch` x7, `wp.array(` x79. All compile on 1.13 except mesh_query_edge. |
| sim/integrator_xpbd.py (2990) | C1 (`wp.sim.` x18), C3 x1 (695), kernel mat33 (10 mat33-family uses; 1 vector-form) | `wp.launch` x23, `wp.array(` x203, `wp.empty` x2, `wp.printf` x2, `ScopedTimer` x1, `mesh_query_point` x1, `mesh_get` x2, `mesh_eval` x2. `replace_mesh_points` kernel: not run on 1.13 [U]. |
| sim/integrator_euler.py (2118) | C1 (`wp.sim.` x7), C2 (imports optimizer -> warp.context), C3 x2 (657,672) | `wp.launch` x18, `wp.array(` x156, `ScopedTimer` x2, `mesh_query_point` x1, `mesh_eval` x2. `integrate_particles`/`integrate_bodies` imported by xpbd, keep file. |
| sim/optimizer.py (138) | C2 | line 8 `warp.context.synchronize` -> `from warp import synchronize`. Imported at module top by euler. |
| sim/utils.py (289) | C3 x2 (48,179), C14 | `igl`, `scipy` top-level imports; `wp.array(` x3, `wp.constant` x2, `mesh_get` x1, `mesh_query_ray` x1. |
| sim/particles.py (113) | none | `wp.array(` x5. Compiled on 1.13. [V] |
| sim/render.py (381) | C2 (line 16), `warp.render` | `wp.array(` x10, `wp.constant` x1; imports `warp.sim`, `warp.render`; `transform_expand` 151,332 exists; `transform_point` 334. Uses `UsdRenderer`; signatures compatible. Not run end-to-end [U]. |
| sim/inertia.py (316) | C5 (313-316, numpy args, U) | `wp.mat33(` x9 (scalar/numpy forms, OK), `wp.array(` x18. Kernels compile. [V compile] |
| sim/articulation.py (686) | C1 (`wp.sim.` x15), C3 x4 (25,55,101,131) | `wp.array(` x32. Only `eval_fk/eval_ik` exported; not used by GarmentCode. |
| sim/import_mjcf.py (486) | `wp.vec(...)` 159,161; `wp.mat33` x? | Droppable. |
| sim/import_urdf.py (485) | `wp.utils.` x1, imports `warp.sim.model.Mesh` | Droppable. |
| sim/import_usd.py (866), import_snu.py (221) | n/a | Droppable. |
| collision/panel_assignment.py (326) | C5 (256, ray_hit only) | `mesh_query_point` x2 (out-param form, OK), `mesh_query_ray` x1, `wp.Mesh(points=,indices=)` 169, `wp.array(` x7, transform_inverse/transform_multiply 174. 'closest' strategy ran on CPU OK [V]. |

Private imports that moved (all files): see C2 table.
Totals: `wp.launch` 63; `wp.printf` 4; `wp.types.` 1; `wp.utils.` 10; `wp.HashGrid` 1; `wp.sim.` 96 across kernels.

## Part 3. Builtins check (kernel scope, verified by t_bi.py dump of 1.13 signatures + module compile on CPU)

All present with compatible signatures in 1.13 unless noted [V]:
dot, cross, normalize, length, length_sq, abs, min, max, clamp, sqrt, sign, step, exp, acos, asin, atan2, nonzero,
cw_mul, cw_div, mul, transpose, inverse, determinant, outer, skew,
transform_point, transform_vector, transform_identity(dtype), transform_get_translation/rotation, transform_inverse, transform_multiply,
quat_rotate, quat_rotate_inv, quat_inverse, quat_identity(dtype), quat_from_axis_angle, quat_from_matrix, quat_to_matrix, quat_rpy, quat_from_euler(e:vec3f,i,j,k),
spatial_vector(w, v) (top=angular, bottom=linear, UNCHANGED [V]), spatial_top, spatial_bottom, spatial_adjoint(r:mat_t, s:mat_t),
atomic_add, atomic_sub, printf(fmt,*args),
closest_point_edge_edge(p1,q1,p2,q2,epsilon),
volume_sample_f, volume_sample_grad_f, volume_world_to_index, wp.Volume.LINEAR,
hash_grid_query / _next / _point_id,
mesh_get(id:uint64), mesh_eval_position(id,face,u,v), mesh_eval_velocity(id,face,u,v),
mesh_query_point(id,point,max_dist,inside,face,u,v) [out-param form still exists and compiles; struct form (id,point,max_dist) also],
mesh_query_point_sign_normal(id,point,max_dist,inside,face,u,v,epsilon) [out-param] and struct form (id,point,max_dist,epsilon),
mesh_query_ray out-param (id,start,dir,max_t,t,u,v,sign,normal,face) [+ optional trailing `root:int32` in 1.13 signature; out-param form compiles and runs, V],
mesh_query_aabb / mesh_query_aabb_next,
array2d/array3d, zeros_like, empty_like, clone, copy, vec2i, vec4i, vec3ub (host-side/constructors).
Python helpers (not builtins) present: `wp.transform_expand`, `wp.quat_between_vectors`.
Absent: `wp.matrix`, `wp.vec`, `wp.mat` names on `wp`; `wp.select` raises; `wp.is_tensor` absent; `wp.leaky_min` (commented only).

New 1.13 mesh builtins that could help workstream 5 [V names via changelog cl:455-470]: `mesh_query_ray_anyhit`, `mesh_query_ray_count_intersections`, `mesh_query_point_sign_parity`, `mesh_get_group_root`, tiled AABB queries.

### mesh_query_edge (fork-only, MISSING in 1.13) [V from fork source]
```
mesh_query_edge(id:uint64, v1:int, v2:int,
                t:float, bary_u:float, bary_v:float, sign:float,
                normal:vec3, face:int) -> bool
```
All out-params (no `require_original_output_arg`). Registered at fork warp/builtins.py ~1829; implementation in warp/native/mesh.h (+107 lines) and exports.h. Call sites: collide.py:1420, :1481. The scratch stub returns False (drops edge-edge-ish contact), which still ran e2e but is NOT behaviour-equivalent.

## Part 4. What GarmentCode needs

GarmentCode imports (garment.py/simulation.py): `warp.sim.render`, `warp.sim.utils.implicit_laplacian_smoothing`,
`warp.collision.panel_assignment`, `warp.sim.collide.count_self_intersections/count_body_cloth_intersections`,
`warp.sim.integrator_xpbd.replace_mesh_points`; uses `ModelBuilder`, `add_cloth_mesh_sewing_spring`, `add_shape_mesh`,
`add_cloth_reference_shape_mesh`, `add_cloth_reference_labels`, `finalize(device=)`, `wp.sim.Mesh`, `XPBDIntegrator`, `wp.sim.collide`,
`SimRenderer(model, path, scaling=1.0)`, `render_mesh(..., is_template=True)`, `model.particle_grid.build`, capture_begin/end/launch, `ScopedTimer("simulate", print=False, active=True)`, `wp.array(..., device='cpu', copy=False)`, `wp.copy`, `Mesh.refit()`, `model.state()`, `wp.get_device().is_cuda`.

| File | Verdict |
|---|---|
| sim/model.py | KEEP (patch C2, C5) |
| sim/collide.py | KEEP (needs mesh_query_edge) |
| sim/integrator_xpbd.py | KEEP (patch C3) |
| sim/integrator_euler.py | KEEP: xpbd imports `integrate_bodies`/`integrate_particles` and its module-level imports; could strip `SemiImplicitIntegrator`/Variational classes but not the file [U runtime refs beyond grep] |
| sim/optimizer.py | STUB or fix line 8 (imported at top of euler) |
| sim/utils.py | KEEP (igl/scipy: lazy-import) |
| sim/inertia.py, sim/particles.py | KEEP |
| sim/render.py | KEEP (fix import) |
| collision/panel_assignment.py | KEEP ('closest' only; 'ray_hit' needs numpy-safe patch) |
| sim/articulation.py | DROP-able: only `__init__` re-exports `eval_fk/eval_ik`; no GarmentCode call [V by grep]; needs C3 patches if kept |
| sim/import_mjcf/urdf/usd/snu.py | DROP (remove from `__init__`) [V: scratch `__init__` without them ran] |

## Not determined
- GPU behaviour, `wp.capture_begin()` no-arg and graph path (GPU left alone).
- `Mesh.refit()` and `replace_mesh_points` on 1.13; `implicit_laplacian_smoothing` (needs igl/scipy).
- beta.6 runtime behaviour (no build): all "beta.6" semantics from source/changelog.
- Whether data_config.py `warp.context.runtime` access is wrapped in try/except.
- `Model.tensors()` was dead already in beta.6.
- Short-circuit `and/or` audit of kernels (C11).
