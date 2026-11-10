# GarmentCode local environment (workstream 4)

Written 2026-11-10. alcyone was unreachable (No route to host) the whole time, so nothing here was
cross-checked against the reference install (no `pip freeze`, no `~/gch/data`, no sample sim outputs).
Versions are simply what resolved today from PyPI against GarmentCode's `setup.cfg`.

## Recreate venv-gc

```
cd /home/carl/lustra/GarmentCode_hack
export UV_HTTP_TIMEOUT=600
/home/carl/py/bin/uv venv --python 3.12 venv-gc
/home/carl/py/bin/uv pip install --python venv-gc/bin/python "numpy<2" scipy pyyaml svgwrite svgpathtools \
    psutil matplotlib CairoSVG nicegui trimesh libigl pyrender cgal pillow
/home/carl/py/bin/uv pip install --python venv-gc/bin/python "pyopengl==3.1.9"     # see "Differences"
# path files instead of editable installs (nothing written into the checkouts):
SP=venv-gc/lib/python3.12/site-packages
echo /home/carl/lustra/GarmentCode_hack/warp-amd-probe > $SP/warp-amd-probe.pth
echo /home/carl/lustra/GarmentCode_hack/GarmentCode    > $SP/garmentcode-root.pth
```
Full resolved list: `port-docs/venv-gc-freeze.txt`. Python 3.12 worked for every dependency (incl. cgal 6.0.1 and
libigl 2.6.3 wheels), so no fallback to 3.10 was needed.

Why `.pth` rather than `uv pip install -e`: warp-amd-probe's `setup.py` is a custom build backend that would
try to build/package native libs; GarmentCode's setup.cfg has an odd `package_dir=pygarment` layout. The `.pth` for
warp-amd-probe makes `import warp` use the already-built `warp/bin/*.so`. Warp's own dependencies beyond numpy were
not needed. (Python will drop `__pycache__` dirs in the checkouts unless `PYTHONDONTWRITEBYTECODE=1`.)
pygarment is *not* pip-installed; `GarmentCode/` on sys.path (via .pth) provides `pygarment`, `assets`, `gui`.
Scripts must be run with cwd = `GarmentCode/` because `./system.json` and `./assets/...` are relative.

## Differences from alcyone / upstream pins
- pyopengl: pyrender 0.1.45 pins `pyopengl==3.1.0`, which fails on this stack
  (`ctypes.ArgumentError: No array-type handler ... glGenTextures`). Upgraded to 3.1.9 (pip warns about the
  conflict; harmless). alcyone's pin is unknown (unreachable), so whether it had the same fix is unverified.
- Python 3.12 vs alcyone 3.10; numpy 1.26.4 (<2 as required).
- torch: NOT needed. No pygarment/gui/test script imports torch (grepped). Not installed.
- Warp: `warp-amd-probe` 1.13.0+rocm.0 vs the beta.6 fork on alcyone.

## Environment variables
```
export ROCM_PATH=/home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH=$ROCM_PATH/lib
export WARP_CACHE_PATH=/tmp/claude-1000/warp-cache-ws4      # your own
export MPLCONFIGDIR=/tmp/claude-1000/mpl                     # ~/.config/matplotlib is not writable here
export PYTHONDONTWRITEBYTECODE=1                             # optional, keeps checkouts clean
```
pythonrender.py sets `PYOPENGL_PLATFORM=egl` itself on Linux; no need to export it.
Run with `/home/carl/lustra/GarmentCode_hack/venv-gc/bin/python`, cwd `GarmentCode/`.
(Note: a GUI-stub run before I exported WARP_CACHE_PATH created `~/.cache/warp/`; safe to delete.)

## system.json (GarmentCode/system.json, git-ignored; workstream 3 may read it)
```
{
  "output": "./Logs/",
  "datasets_path": "",
  "datasets_sim": "",
  "sim_configs_path": "./assets/Sim_props",
  "bodies_default_path": "./assets/bodies",
  "body_samples_path": ""
}
```
(= `system.template.json` unchanged; the in-repo `assets/bodies` and `assets/Sim_props` are enough for the default body `mean_all`.
No data had to be copied from alcyone.)

## Verification results
Scripts are in `port-docs/scripts/`.

1. Imports (`import_all.py`): 21 pygarment modules import OK. Fail: `pygarment.mayaqltools` (needs Maya, irrelevant)
   and `meshgen.garment`, `meshgen.simulation`, `meshgen.datasim_utils` (all `No module named 'warp.sim'`).
   `cgal`, `igl`, `trimesh`, `pyrender`, `cairosvg` all import fine.
2. Pattern generation (`gen_and_mesh.py`): 9 designs (t-shirt, fitted shirt, shirt+pants, circle skirt+WB,
   fitted shirt+pencil skirt+FittedWB, levels skirt, godet, Skirt2, many-panels) generated, serialized (svg/png/json) with
   no errors, ~0.1 s each. Designs are built from `default.yaml` by setting `meta.upper/wb/bottom`.
3. Box meshing (same script; reproduces `test_garment_sim.py` up to `run_sim`: `BoxMesh(...).load(); serialize(...)`
   at resolution_scale from default_sim_props): all 9 OK, 1-3 s each, V=3.8k-12k, F=7k-24k, output under
   `/tmp/claude-1000/ws4-out/mesh/<name>/` (`*_boxmesh.obj`, segmentation, orig_lens pickle, ...). Only a harmless
   warning `Path does not exist: .../design_params.yaml` (my script does not copy the design file).
   `import cgal` works on 3.12, nothing else was required.
4. Rendering (`render_test.py`): `pygarment.meshgen.render.pythonrender.render_images` works with EGL on the Radeon:
   GL_RENDERER = "AMD Radeon 8060S (radeonsi, strix_halo, ACO)", Mesa 26.0.3, OpenGL 4.6 core. No osmesa lib installed
   (not needed). Rendered the t-shirt box mesh (stand-in for `_sim.obj`, copied to that name) + mean_all body at 800x800
   front/back; images at `/tmp/claude-1000/ws4-render/t-shirt/`, ~57k opaque non-white pixels each, visually inspected: textured
   flat-panel box garment, correct. A real simulated garment was not available (alcyone down), so the sim-output path
   (g_sim with draped vertices) is untested, but it is just a trimesh load of an OBJ.
5. GUI: `python gui.py` fails at import: `gui/gui_pattern.py:16 -> pygarment/meshgen/simulation.py:29 -> garment.py:9
   import warp.sim.render`. With `warp.sim*` stubbed by MagicMock (`gui_stubbed.py`; scratch check only) NiceGUI binds
   0.0.0.0:8080 and `GET /` returns HTTP 200; stopped afterward. Also: `ui.run` opens a browser by default; here it
   tried to launch Google Chrome, which aborted on its SUID sandbox (harmless, server unaffected). No GUI interaction tested.

## Where the pipeline stops without warp.sim
`python test_garment_sim.py` (and gui.py, and importing simulation/garment/datasim_utils):
```
File "GarmentCode/test_garment_sim.py", line 6: from pygarment.meshgen.simulation import run_sim
File "pygarment/meshgen/simulation.py", line 29: from pygarment.meshgen.garment import Cloth
File "pygarment/meshgen/garment.py", line 9: import warp.sim.render
ModuleNotFoundError: No module named 'warp.sim'
```
Further warp imports garment.py needs next: `warp.sim.utils.implicit_laplacian_smoothing`,
`warp.collision.panel_assignment`, `warp.sim.collide.{count_self_intersections,count_body_cloth_intersections}`,
`warp.sim.integrator_xpbd.replace_mesh_points`. (`import warp; warp.init()` itself works: devices cpu and
cuda:0 "AMD Radeon 8060S Graphics", gfx1151, Warp 1.13.0+rocm.0.)
Note `test_garment_sim.py` imports `run_sim` before the meshing part, so it can't even reach meshing; use `gen_and_mesh.py`.

## GarmentCode patches
None. `git -C GarmentCode status` is clean (system.json is ignored).

## System packages
None needed (libcairo2, libEGL/Mesa, libgbm already present).

## Not obtained / unverified (alcyone down)
- alcyone pip freeze / version pins, `system.json`, `~/gch/data`, `~/gch/scripts/all_designs.py`; pyopengl 3.1.0 situation there.
- Real simulated garment meshes for the render test; body sample datasets (`body_samples_path` left empty).
- Anything that runs the sim (`run_sim`, `Cloth`, datasim); pattern_data_sim.py, pattern_fitter.py, pattern_sampler.py not run.
