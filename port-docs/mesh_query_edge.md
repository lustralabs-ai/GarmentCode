# `wp.mesh_query_edge` on Warp 1.13 (AMD fork), workstream 5

Branch `ws5-mesh-query-edge` in `warp-ws5/`, two commits on `garmentcode-rocm` (49010b944):

- `a1b408526`: Add mesh_query_edge builtin for GarmentCode (native, registration, `__init__.pyi`, docs list)
- `08b97b69b`: Test mesh_query_edge against a brute-force reference

## Call-site changes the sim port needs

None. The fork's call pattern compiles and behaves the same:

```python
t = float(0.)
face_index = int(0)
face_u = float(0.0)
face_v = float(0.0)
sign = float(0.0)
normal = wp.vec3()
if wp.mesh_query_edge(mesh_id, v1, v2, t, face_u, face_v, sign, normal, face_index):
    ...
```

The out-parameter overload of `wp.mesh_query_ray` (`mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face)`),
which the fork's `collide.py`, `utils.py` and `panel_assignment.py` also use, still exists in 1.13: it is registered as
`hidden=True` (left out of the docs), but codegen accepts it and the existing `test_mesh_query_ray` tests run it.

## Python signatures available

```
wp.mesh_query_edge(id: uint64, v1: int32, v2: int32,
                   t: float32, bary_u: float32, bary_v: float32,
                   sign: float32, normal: vec3f, face: int32) -> bool       # fork signature, out-parameters
wp.mesh_query_edge(id: uint64, v1: int32, v2: int32) -> wp.MeshQueryRay    # 1.x style: .result .t .u .v .sign .normal .face
```

Both are kernel-only (`export=False`), like `mesh_query_ray`, and `is_differentiable=False`. The fork's adjoint was an
empty stub, so gradients are zero in both cases.

## Design

- `warp/native/mesh.h`: `mesh_query_edge(id, v1, v2, t, u, v, sign, normal, face)` plus the struct-returning wrapper
  and empty adjoints. Placed after `mesh_query_ray_t`.
- Traversal on the Warp BVH (sah / median / lbvh): written from the target's `mesh_query_ray`, adjusted to the 1.13
  layout: leaves are ranges `[lower.i, upper.i)` into `bvh.primitive_indices` (leaf size can be above 1), nodes are
  read with `bvh_load_node`, root is `*mesh.bvh.root`. It uses the nearer-child-first, prune-by-entry-distance scheme
  of AMD's HIP `mesh_query_ray` path on all backends (CPU and GPU), since it is plain C++ and prunes better than the
  generic stack traversal. Node AABBs are padded by 1e-3 as in the fork and in `mesh_query_ray`. Stack size is
  `BVH_QUERY_STACK_SIZE` (32), the same as `mesh_query_ray`.
- The vertex-exclusion test is applied per primitive before the ray/triangle test:
  `i,j,k != v1 && i,j,k != v2`, as in the fork.
- cuBQL meshes (`bvh_constructor="cubql"`, experimental) are also supported: `cubql_ray_traversal` gained two optional
  arguments `exclude_vertex_a/b` (default -1, which excludes nothing, so existing callers behave as before). cuBQL
  nodes have no 1e-3 padding; that is the same difference `mesh_query_ray` already has for cuBQL.
- Ray/triangle test: `intersect_ray_tri_woop`, which is identical between beta.6 and 1.13 except for formatting
  (beta.6 swapped `kx/ky` through a `float` temporary, 1.13 through an `int`; same result). So `u`, `v` are the
  barycentric weights of the face's first and second vertex (`indices[3f+0]`, `indices[3f+1]`), matching
  `mesh_eval_position(id, face, u, v) = p*u + q*v + r*(1-u-v)` in both versions. `sign` is the raw Woop
  determinant (negative when the ray travels along the face normal), `normal` is `normalize(cross(q-p, r-p))`.

## Semantic differences from the fork

1. Ties. When two triangles are hit at exactly the same `t` (ray through a shared triangle edge), the first one
   visited wins (strict `<`). Visit order depends on the BVH (1.13 builder, leaf packing, near-first order), so the
   reported `face` can differ from the fork in exact ties. It was already BVH-dependent in the fork.
2. Empty-mesh guard. The fork returned `false` when `mesh.bvh.num_nodes == 0`. In 1.13, `num_nodes` is not set for
   device-built (LBVH) trees, and that check made every GPU query return `false` during development. The port checks
   `mesh.num_tris == 0 || !mesh.bvh.root` instead.
3. Degenerate edge (`v1 == v2` or coincident points). The port returns `false` immediately. The fork also always
   returned `false` here, but only after a NaN traversal.
4. Outputs on a miss are left untouched, as in the fork. In the struct overload they keep the `mesh_query_ray_t`
   defaults (`face = 0`, `t = 0`, ...).
5. Not exported to Python scope (`exports.h`). The fork exported it, but calling it outside a kernel with
   out-parameters had no use.

Unchanged: `0 <= t < |x(v2) - x(v1)|` (strict upper bound), ray origin `x(v1)`, direction `normalize(x(v2) - x(v1))`,
closest hit wins, normal normalized, `u`/`v` convention, `sign` convention.

## Tests

`warp/tests/geometry/test_mesh_query_edge.py` (added to `default_suite`). The reference is a vectorized float64 NumPy
port of `intersect_ray_tri_woop` over all triangles, with the same exclusion and acceptance rules. Queries are
skipped as numerically ambiguous (float32 vs float64) only if the ray passes within 1e-4 of a triangle edge, crosses
a triangle within 1e-4 of either end point, or has two hits within 1e-4 of each other. That leaves out 13 of 8,950
queries in the main scene and 1 of 4,187 in the axis-aligned scene. Each test runs for every BVH constructor
available on the device (CPU: sah, median, cubql; GPU: sah, median, lbvh, cubql) and leaf sizes 1 and 4:

- `brute_force`: a jittered 20x20 grid, a tilted jittered 16x16 grid and a jittered UV sphere, all interpenetrating
  (1,936 triangles). Queries: every real edge in both directions plus about 3,000 random vertex pairs (8,950 total;
  2,931 hits, 1,448 of them with 2 or more candidate triangles; 648 hits on real edges). Checks hit/no-hit, `face`
  (exact), `t`, `u`, `v`, `sign`, `normal`, and that `mesh_eval_position(face, u, v)` equals `x(v1) + t*dir`. Run
  for both overloads. Without the exclusion rule the reference would report 8,946 hits, so exclusion is exercised
  everywhere.
- `axis_aligned`: three unjittered, axis-aligned, mutually perpendicular grids (688 triangles, 4,187 queries, of which
  1,621 have direction vectors with two zero components, i.e. infinite `rcp_dir` in the slab test; 166 of those hit).
  Added because AMD skips `test_bvh_ray_query_inside_and_outside_bounds` on HIP for a Clang bug with axis-aligned
  rays. No problem showed up here.
- `refit`: the same kind of scene, compared before and after moving the points (`mesh.points.assign(...)`,
  `mesh.refit()`), for all constructors including cuBQL. Asserts that over 50 queries change outcome.
- `adjacent_excluded`: a fan around vertex 0. A ray from the fan centre hits the adjacent triangles at `t = 0` through
  `mesh_query_ray`, while `mesh_query_edge` reports no hit. A separate triangle crossed at `t = 0.5` is reported with
  the right face, normal `(0,0,1)` and `sign < 0`. Also covers the reversed edge and a real edge with no hit.
- `degenerate`: `v1 == v2` returns `false`.

Mutation check: with exclusion of `v2` removed, the brute-force tests fail on both devices (hundreds of hit mismatches).

### Commands

```
cd $WS/warp-ws5
export PATH=/home/carl/py/bin:$PATH
export ROCM_PATH=/home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH=$ROCM_PATH/lib WARP_CACHE_PATH=/tmp/claude-1000/warp-cache-ws5
.venv/bin/python build_lib.py --rocm-path=$ROCM_PATH --hip-arch=gfx1151
PYTHONPATH=$PWD .venv/bin/python warp/tests/geometry/test_mesh_query_edge.py
for t in test_mesh_query_ray test_mesh_query_point test_mesh_query_aabb test_mesh test_bvh test_grouped_bvh; do
  PYTHONPATH=$PWD .venv/bin/python warp/tests/geometry/$t.py; done
```

(`.venv` contains numpy, usd-core, ruff and clang-format. Without usd-core most mesh_query_ray/point tests are skipped.)

### Results (2026-11-10, devices `cpu` and `cuda:0` = Radeon 8060S gfx1151, ROCm 7.15 / HIP)

| Test file | Result | Notes |
|---|---|---|
| test_mesh_query_edge | 10/10 OK | 5 tests, each run on cpu **and** cuda:0 |
| test_mesh_query_ray | 9/9 OK | |
| test_mesh_query_point | 9/9 OK | |
| test_mesh_query_aabb | 15/15 OK | |
| test_mesh | 13/13 OK | |
| test_bvh | 14 OK, 2 skipped | skips are AMD's own on HIP (axis-aligned single-AABB Clang bug; BVH rebuild in graph capture) |
| test_grouped_bvh | 12 OK, 1 skipped | HIP graph-capture rebuild skip (AMD's) |
| test_context, test_builtins_resolution | OK | |

The native library builds cleanly with `-Werror`.

## Gotcha for anyone editing `warp/native/*.h`

The kernel cache key does not include native header contents. After editing a header without running `build_lib.py`
(which clears the cache), old kernels keep loading from `WARP_CACHE_PATH` ("(cached)" in the load log). Delete the
module directories from the cache by hand. Modules run as scripts are cached as `wp___main___*`.

## Not done

- No CHANGELOG entry, to avoid conflicts on the shared integration branch.
- No performance measurement (the sim calls this once per edge per step).
