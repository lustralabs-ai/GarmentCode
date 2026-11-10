# Setting up GarmentCode cloth simulation on an AMD GPU (Warp ROCm port)

Workstream 9a, 2026-11-10. Goal: from "ROCm SDK installed, repos cloned" to "GarmentCode drapes a garment on the AMD GPU".
Everything below was run for real from scratch (fresh Warp worktree, fresh venv, empty kernel cache) on the Radeon 8060S
(gfx1151, ROCm 7.15 runtime as reported by Warp).

## 1. Prerequisites

| Need | Notes |
|---|---|
| ROCm SDK with `bin/hipcc` and `lib/` | Here: the pip SDK in `/home/carl/py`, root `/home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel`. The script only reads it; it never installs into it. |
| AMD GPU + amdgpu driver, user in `render`/`video` group | `/dev/kfd` must be accessible. |
| Warp source on branch `garmentcode-rocm` | Warp 1.13 AMD fork plus the GarmentCode `warp.sim`/`warp.collision` port. Plain AMD or upstream Warp has no `warp.sim` (removed in 1.10), so GarmentCode cannot import (PLAN.md, local_env.md). |
| GarmentCode checkout | Unmodified is fine. |
| `g++` and `git`, network access | `build_lib.py` downloads LLVM headers through packman into `<warp>/_build`; pip installs from PyPI. |
| `uv` (preferred) or a Python 3.12 with `venv` | `uv venv --python 3.12` can fetch an interpreter. Without uv the script uses `python -m venv` with whatever `--python` resolves to. In my run uv was used (`/home/carl/py/bin/uv`, which is on PATH on this machine; only the uv binary is used, nothing is installed in that venv). |

Python 3.12 works for every dependency (local_env.md). Alcyone's reference install used 3.10; not needed here.

## 2. One-command setup

```
cd $WS
port-docs/scripts/setup_amd.sh \
    --rocm-path /home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel \
    --warp-dir  $PWD/warp-amd-probe \
    --garmentcode-dir $PWD/GarmentCode \
    --venv $PWD/venv-gc \
    --cache-dir /path/for/kernel-cache        # optional
```
Options: `--hip-arch` (default: first non-`gfx000` entry of `rocm_agent_enumerator`, else `gfx1151`), `--python`
(default `3.12`; version for uv, or interpreter name/path without uv), `--env-file` (default `<venv>.env.sh`),
`--skip-build`, `--force-build`, `--skip-smoke`. The script is idempotent: an existing venv is reused, `pip`/`uv` installs are
no-ops when satisfied, Warp is not rebuilt if `warp/bin/warp.so` and `warp-clang.so` exist (use `--force-build`), an existing
`system.json` is never overwritten. It stops at the first failure with an `ERROR:` message.

What it does:
1. Checks prerequisites (hipcc, Warp tree has `warp/sim`, GarmentCode tree, g++, git), detects the HIP arch, picks uv or venv.
2. Creates the venv.
3. Installs dependencies (section 4 explains the pins).
4. Builds Warp (about 45 s, CPU-only work, 4 workers) with the venv's python.
5. Writes `warp-amd.pth` and `garmentcode-root.pth` into the venv's site-packages.
6. Copies `system.template.json` to `system.json` if missing.
7. Writes `env.sh` and runs the smoke test: import warp, list devices, require a HIP GPU, one tiny kernel (1000 `atomic_add`s, so the
   only GPU use is a ~1 s JIT compile plus one launch), `import warp.sim`, `import warp.collision.panel_assignment`, `import pygarment`.

Evidence from the clean run (venv, cache, and `warp/bin/*.so` all absent beforehand; total 51 s):
```
autodetected HIP arch: gfx1151        venv tool: uv
devices: ['cpu', 'cuda:0']
gpu: AMD Radeon 8060S Graphics is_hip True
Module __main__ 7ef8bcf load on device 'cuda:0' took 1042.32 ms  (compiled)
kernel result: 1000.0
import warp.sim OK / import pygarment OK / SMOKE TEST PASSED
```

## 3. Runtime environment (`env.sh`)

The script writes `<venv>.env.sh`; source it in every new shell:
```
export ROCM_PATH=<sdk root>
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # Warp's HIP runtime loads libamdhip64 etc. from here
export WARP_CACHE_PATH=<dir>            # only if --cache-dir was given; default is ~/.cache/warp/<version>
export PYTHONDONTWRITEBYTECODE=1        # keeps __pycache__ out of the checkouts (local_env.md)
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-$USER}   # ~/.config/matplotlib may not be writable (local_env.md)
source <venv>/bin/activate
```
Run GarmentCode scripts with cwd = the GarmentCode directory: `./system.json` and `./assets/...` are relative (local_env.md).
`pythonrender.py` sets `PYOPENGL_PLATFORM=egl` itself; do not export it.

First-run kernel compile: with an empty kernel cache the first simulation spends about 2 minutes in HIPRTC compiles
(`warp.sim.collide` alone is ~102 s). Everything is cached afterwards under `WARP_CACHE_PATH`; use one cache per user and warm it with a single process
before launching parallel runs (sim_port.md section 4; reference/README.md gotcha 3: concurrent first-time JIT into one cache segfaulted 3 of 6 runs on
CPU). The CPU backend compiles faster (about 15 s for the sim modules, measured).
HIPRTC prints "does not support precompiled headers" once per module; harmless (sim_port.md).

## 4. Manual equivalent of each step

```
S=$WS
ROCM=/home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel

# 2. venv
uv venv --python 3.12 $S/venv-gc            # or: python3.12 -m venv $S/venv-gc

# 3. deps (uv shown; use "$S/venv-gc/bin/python -m pip install" otherwise)
uv pip install --python $S/venv-gc/bin/python numpy scipy pyyaml svgwrite svgpathtools psutil matplotlib \
    CairoSVG nicegui trimesh libigl pyrender cgal pillow
uv pip install --python $S/venv-gc/bin/python "pyopengl==3.1.9"          # second step on purpose

# 4. Warp build
cd $S/warp-amd-probe
chmod +x tools/packman/packman tools/packman/python.sh
export ROCM_PATH=$ROCM LD_LIBRARY_PATH=$ROCM/lib
$S/venv-gc/bin/python build_lib.py --rocm-path=$ROCM --hip-arch=gfx1151

# 5. link (no editable installs: warp's setup.py is a custom build backend; GarmentCode's setup.cfg has an odd package_dir layout)
SP=$S/venv-gc/lib/python3.12/site-packages
echo $S/warp-amd-probe > $SP/warp-amd.pth
echo $S/GarmentCode    > $SP/garmentcode-root.pth

# 6. config
cp $S/GarmentCode/system.template.json $S/GarmentCode/system.json
```
The hand recipe in the docs used `/home/carl/py/bin` first on PATH for the build; this is not needed, the venv's python is enough
(`build_lib.py` only needs numpy).

### Dependency pins: which matter and why
Only two pins are deliberate; everything else floats (the resolved set that is known to work is `port-docs/venv-gc-freeze.txt`).
- **numpy**: no pin needed any more. Upstream GarmentCode requires `numpy<2` because it calls `np.cross` on 2-D vectors
  (removed in NumPy 2: `ValueError: Both input arrays must be (arrays of) 3-dimensional vectors`). The local GarmentCode tree
  replaces those five calls with a `cross_2d` helper and drops the pin in `setup.cfg`. pyrender 0.1.45 still uses `np.infty`
  (`pyrender/mesh.py`, bounds of an empty mesh); `pygarment/meshgen/render/pythonrender.py` restores that alias before importing pyrender. Checked with numpy 2.5.4: pattern
  generation + meshing (9 designs), EGL render, and a 10-frame CPU sim bit-identical to the reference. Not yet checked on
  numpy 2: GPU sim, GUI, `pattern_fitter.py`. `venv-gc` itself is still on 1.26.4.
- **`pyopengl==3.1.9`** installed after pyrender. pyrender 0.1.45 pins `pyopengl==3.1.0` (uv visibly downgrades to it), which fails under EGL here with
  `ctypes.ArgumentError: No array-type handler ... glGenTextures` (local_env.md; not re-run by me because rendering uses the GPU's GL stack and the GPU
  was reserved). pip prints a conflict warning; ignore it.
- Not pins, but needed: `libigl` (used by `warp/sim/utils.py` implicit smoothing and pygarment), `scipy`, `cgal` (box meshing). torch is not needed anywhere (local_env.md).
  Observed versions: libigl 2.6.3, cgal 6.0.1, trimesh 5.2.0, scipy 1.17.1, nicegui 4.0.0, pyrender 0.1.45.

## 5. Verifying a fresh environment

```
source <venv>.env.sh
cd GarmentCode
python ../port-docs/scripts/gen_and_mesh.py /tmp/out        # pattern generation + box meshing for 9 designs (18 s)
export WARP_CACHE_PATH=/tmp/somewhere
python ../reference/scripts/dump_harness.py run --garmentcode-dir $PWD \
    --inputs ../reference/data/inputs/shirt_mean --device cpu --max-frames 10 --out /tmp/cpu_run
```
Results in the fresh env: all 9 designs OK (V 3.8k-12k, F 7k-24k; the "design_params.yaml does not exist" warning is harmless);
the 10-frame CPU sim took 25 s wall including ~14 s of kernel compile, ended with `stop_reason: frame_cap`, no NaNs, and frame 8 is bit-identical
(max diff 0.0) to `reference/data/shirt_mean/cpu_run1/frames/f000008.npy`.
To run on the GPU use `--device cuda:0` (or `python test_garment_sim.py`); remember the 2-minute first compile. GPU results are not bit-comparable to CPU
(shirt: max 2.2 cm final-vertex difference, 19 vs 13 self-intersections, sim_port.md section 4).

## 6. Known limitations and gotchas

- GarmentCode needs `warp.sim`, which only exists on `garmentcode-rocm`; the port is bit-identical to the beta.6 fork on CPU, the GPU path is statistically (not bitwise) equivalent (sim_port.md, PLAN.md, reference/README.md).
- `gui.py` imports `warp.sim` through `pygarment.meshgen.simulation`; the GUI server started with a stubbed sim in the older check (local_env.md item 5). With the real port it has not been re-tested. `ui.run` tries to open a browser (Chrome aborts on its SUID sandbox; harmless).
- `pygarment/data_config.py` `add_sys_info()` used `warp.context.runtime`, which breaks on Warp >= 1.10 (sim_port.md open issue 1). The GarmentCode checkout currently carries an uncommitted fix for it (`warp.init()` unconditionally); with an unpatched checkout `pattern_data_sim.py`/`pattern_fitter.py` would crash, `test_garment_sim.py` would not.
- CPU backend: GarmentCode's `is_static()` check aliases buffers on CPU, so GarmentCode's own run stops at the first check (frame 10) (reference/README.md gotcha 2, sim_port.md open issue 2). Use the harness's default `--static-mode copy` to get realistic CPU runs. GPU is unaffected.
- `panel_assignment` strategy `ray_hit` is unported (GarmentCode uses `closest`); `transform_inertia` differs by 1 ulp at Python scope; `hash_grid` floor-vs-truncation unverified (sim_port.md open issues 3-5).
- gfx1151 is not an AMD-tested target (PLAN.md risks). Warp's own test suite on this GPU: CPU 0 failures, cuda:0 34 failures + 1 error, classified in test_triage.md.
- Native build quirks: ROCm 10.1's hiprtc lacks `uint8_t..uint64_t` at JIT time (fixed by typedefs in `warp/native/builtin.h` on the branch, PLAN.md). HIP builds disable libmathdx automatically.
- Do not install anything into `/home/carl/py` (it is the ROCm SDK provider); `uv pip install` always with `--python <venv>/bin/python`.
- Several processes JIT-compiling into one `WARP_CACHE_PATH` at the same time can segfault (reference/README.md); give each process its own cache or warm one first.
- `reference/scripts/compare_runs.py` has a known `f000000_init.npy` vs `f000000.npy` key collision (sim_port.md open issue 6).
- Python drops `__pycache__` into the checkouts unless `PYTHONDONTWRITEBYTECODE=1` (set by env.sh).

## 7. Troubleshooting (failures hit while writing this)

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: Directly evaluating Warp code defined as a string using exec() is not supported` | You defined a `@wp.kernel` from stdin or `python -c`. Warp needs the kernel source in a file. (Hit in my first smoke-test draft; the script now writes a temp file.) |
| `ld.lld: error: cannot open output file .../warp/bin/warp.so: No such file or directory`, then "Warp build failed" | `warp/bin/` does not exist. It is normally present because `warp/bin/.gitignore` is tracked; I had deleted the directory myself. Restore it with `git checkout -- warp/bin`; the script also `mkdir -p`s it. Never `rm -rf warp/bin`, delete `warp/bin/*.so` instead. |
| `ValueError: Both input arrays must be (arrays of) 3-dimensional vectors` in pattern generation | Unpatched GarmentCode on numpy 2; use the patched tree or install `numpy<2`. |
| `ctypes.ArgumentError: No array-type handler ... glGenTextures` when rendering | pyopengl 3.1.0 (pyrender's pin); `pip install pyopengl==3.1.9`. |
| `SMOKE FAIL: no GPU device visible to Warp` | `LD_LIBRARY_PATH` lacks `$ROCM_PATH/lib`, wrong ROCm path, or no access to `/dev/kfd`. Source env.sh. |
| `ModuleNotFoundError: No module named 'warp.sim'` | Warp tree is not the `garmentcode-rocm` branch (script prefix check catches this) or the `.pth` points at the wrong tree. |
| Build fails with a download error | packman needs network access to fetch LLVM headers. |
| Warp prints `Warp warning: HIPRTC does not support precompiled headers` | Expected. |
| Segfault at frame 0 on first launch with parallel runs | Shared kernel cache race; see section 6. |
| `Path does not exist: .../design_params.yaml` | Harmless warning from box meshing when the design file was not copied next to the pattern. |

## 8. Corrections to the earlier docs
- `local_env.md` "Where the pipeline stops without warp.sim" and the 21/4 import-failure list are stale: `warp.sim` now exists; `meshgen.garment/simulation/datasim_utils` import with the ported Warp.
- `local_env.md` names the path file `warp-amd-probe.pth`; the script writes `warp-amd.pth` (any name works).
- `sim_port.md` open issue 1 (`data_config.py:301`) has since been patched in the working copy of GarmentCode, so the statement "GarmentCode is unmodified" in the task brief is no longer exactly true (`git -C GarmentCode status` shows `pygarment/data_config.py` modified).
- `PLAN.md` says `warp-amd-probe/` has its "own `.venv`"; that venv is only for Warp's tests (test_triage.md), not for GarmentCode (use `venv-gc`).
