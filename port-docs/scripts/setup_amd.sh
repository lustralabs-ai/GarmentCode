#!/usr/bin/env bash
# Reproducible setup: GarmentCode cloth simulation on an AMD GPU (Warp ROCm port).
# See port-docs/SETUP.md for the manual equivalent of every step.
#
# Usage:
#   setup_amd.sh --rocm-path DIR --warp-dir DIR --garmentcode-dir DIR --venv DIR [options]
#
# Required:
#   --rocm-path DIR        ROCm SDK root (the dir containing bin/hipcc and lib/), e.g.
#                          /home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel
#   --warp-dir DIR         Warp source (branch garmentcode-rocm checkout / worktree)
#   --garmentcode-dir DIR  GarmentCode checkout
#   --venv DIR             venv to create/update (never point this at the ROCm SDK's own venv)
# Options:
#   --hip-arch ARCH        default: autodetect via rocm_agent_enumerator, else gfx1151
#   --python SPEC          interpreter or version for the venv (default 3.12; uv can fetch it)
#   --cache-dir DIR        if given, written to env.sh as WARP_CACHE_PATH (kernel cache)
#   --skip-build           do not (re)build Warp even if the native libs are missing
#   --force-build          rebuild Warp even if libs exist
#   --skip-smoke           skip the final smoke test
#   --env-file FILE        where to write env.sh (default: <venv>/../env.sh -> "<venv>.env.sh")
set -euo pipefail

log()  { printf '\n==> %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

ROCM_PATH_ARG="" WARP_DIR="" GC_DIR="" VENV="" HIP_ARCH="" PYSPEC="3.12" CACHE_DIR=""
SKIP_BUILD=0 FORCE_BUILD=0 SKIP_SMOKE=0 ENV_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rocm-path) ROCM_PATH_ARG="$2"; shift 2;;
    --warp-dir) WARP_DIR="$2"; shift 2;;
    --garmentcode-dir) GC_DIR="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --hip-arch) HIP_ARCH="$2"; shift 2;;
    --python) PYSPEC="$2"; shift 2;;
    --cache-dir) CACHE_DIR="$2"; shift 2;;
    --env-file) ENV_FILE="$2"; shift 2;;
    --skip-build) SKIP_BUILD=1; shift;;
    --force-build) FORCE_BUILD=1; shift;;
    --skip-smoke) SKIP_SMOKE=1; shift;;
    -h|--help) sed -n '2,25p' "$0"; exit 0;;
    *) die "unknown argument: $1 (try --help)";;
  esac
done
for v in ROCM_PATH_ARG WARP_DIR GC_DIR VENV; do
  [ -n "${!v}" ] || die "missing required argument for $v (try --help)"
done

abspath() { (cd "$1" 2>/dev/null && pwd) || return 1; }
ROCM_PATH=$(abspath "$ROCM_PATH_ARG") || die "--rocm-path '$ROCM_PATH_ARG' does not exist"
WARP_DIR=$(abspath "$WARP_DIR")       || die "--warp-dir '$WARP_DIR' does not exist"
GC_DIR=$(abspath "$GC_DIR")           || die "--garmentcode-dir '$GC_DIR' does not exist"
mkdir -p "$(dirname "$VENV")"
VENV="$(abspath "$(dirname "$VENV")")/$(basename "$VENV")"
[ -n "$ENV_FILE" ] || ENV_FILE="$VENV.env.sh"
[ -n "$CACHE_DIR" ] && { mkdir -p "$CACHE_DIR"; CACHE_DIR=$(abspath "$CACHE_DIR"); }

# ---------------------------------------------------------------- 1. prerequisites
log "1/7 Checking prerequisites"
[ -x "$ROCM_PATH/bin/hipcc" ] || die "$ROCM_PATH/bin/hipcc not found. --rocm-path must be the SDK root that contains bin/hipcc (for the pip SDK: .../site-packages/_rocm_sdk_devel)."
[ -d "$ROCM_PATH/lib" ] || die "$ROCM_PATH/lib missing"
[ -f "$WARP_DIR/build_lib.py" ] || die "$WARP_DIR/build_lib.py not found: not a Warp source tree"
[ -f "$WARP_DIR/warp/sim/__init__.py" ] || die "$WARP_DIR has no warp/sim: you need the garmentcode-rocm branch, not plain AMD/upstream Warp (>=1.10 dropped warp.sim)"
[ -f "$GC_DIR/pygarment/__init__.py" ] || [ -d "$GC_DIR/pygarment" ] || die "$GC_DIR/pygarment not found: not a GarmentCode checkout"
[ -f "$GC_DIR/system.template.json" ] || die "$GC_DIR/system.template.json not found"
command -v gcc >/dev/null || command -v cc >/dev/null || die "no C/C++ compiler (gcc) on PATH; Warp's host build needs one"
command -v git >/dev/null || die "git not found"

if [ -z "$HIP_ARCH" ]; then
  if [ -x "$ROCM_PATH/bin/rocm_agent_enumerator" ]; then
    HIP_ARCH=$("$ROCM_PATH/bin/rocm_agent_enumerator" 2>/dev/null | grep -E '^gfx[0-9a-f]+$' | grep -v '^gfx000$' | head -1 || true)
  fi
  if [ -n "$HIP_ARCH" ]; then echo "autodetected HIP arch: $HIP_ARCH"
  else HIP_ARCH=gfx1151; echo "could not autodetect HIP arch, defaulting to gfx1151"; fi
else
  echo "HIP arch (given): $HIP_ARCH"
fi

# venv tool: uv if available, else python -m venv
USE_UV=0
if command -v uv >/dev/null 2>&1; then USE_UV=1; fi
echo "venv tool: $([ $USE_UV = 1 ] && echo "uv ($(command -v uv))" || echo 'python -m venv')"
PYBIN=""
if [ $USE_UV = 0 ]; then
  for c in "$PYSPEC" "python$PYSPEC" python3; do
    if command -v "$c" >/dev/null 2>&1; then PYBIN=$(command -v "$c"); break; fi
  done
  [ -n "$PYBIN" ] || die "uv not found and no python interpreter matching '$PYSPEC' on PATH; install uv or pass --python /path/to/python3.12"
fi

# ---------------------------------------------------------------- 2. venv
log "2/7 Creating venv $VENV"
if [ -x "$VENV/bin/python" ]; then
  echo "venv exists, reusing"
else
  if [ $USE_UV = 1 ]; then uv venv --python "$PYSPEC" "$VENV" || die "uv venv failed (is python $PYSPEC available/downloadable?)"
  else "$PYBIN" -m venv "$VENV" || die "python -m venv failed"; fi
fi
VPY="$VENV/bin/python"
pipi() {
  if [ $USE_UV = 1 ]; then UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600} uv pip install --python "$VPY" "$@"
  else "$VPY" -m pip install "$@"; fi
}

# ---------------------------------------------------------------- 3. deps
log "3/7 Installing GarmentCode dependencies"
# Pins that matter (details in SETUP.md):
#   numpy          no pin: GarmentCode is patched for NumPy 2 (2-D np.cross replaced); tested with 1.26.4 and 2.5.4
#   pyopengl==3.1.9 pyrender 0.1.45 pins pyopengl==3.1.0 which breaks EGL rendering
#                  (ctypes.ArgumentError ... glGenTextures); installed in a second step to override that pin
# Everything else floats; the versions known to work are in port-docs/venv-gc-freeze.txt.
pipi numpy scipy pyyaml svgwrite svgpathtools psutil matplotlib CairoSVG nicegui \
     trimesh libigl pyrender cgal pillow || die "dependency install failed (network? see SETUP.md troubleshooting)"
pipi "pyopengl==3.1.9" || die "pyopengl install failed"

# ---------------------------------------------------------------- 4. build Warp
log "4/7 Building Warp for $HIP_ARCH"
have_libs() { ls "$WARP_DIR"/warp/bin/warp.so >/dev/null 2>&1 && ls "$WARP_DIR"/warp/bin/warp-clang.so >/dev/null 2>&1; }
if have_libs && [ $FORCE_BUILD = 0 ]; then
  echo "native libs already present in $WARP_DIR/warp/bin (use --force-build to rebuild)"
elif [ $SKIP_BUILD = 1 ]; then
  have_libs || die "--skip-build given but $WARP_DIR/warp/bin has no warp.so"
else
  mkdir -p "$WARP_DIR/warp/bin"
  chmod +x "$WARP_DIR/tools/packman/packman" "$WARP_DIR/tools/packman/python.sh" 2>/dev/null || true
  ( cd "$WARP_DIR"
    export ROCM_PATH LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # build_lib.py only needs numpy; use the venv's python so nothing outside it is touched
    "$VPY" build_lib.py --rocm-path="$ROCM_PATH" --hip-arch="$HIP_ARCH" ) \
    || die "Warp build failed. Check the output above (packman download needs network; a C++ compiler is required)."
  have_libs || die "build finished but warp/bin/warp.so is missing"
fi

# ---------------------------------------------------------------- 5. link
log "5/7 Linking Warp and pygarment into the venv (.pth files)"
SP=$("$VPY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$WARP_DIR" > "$SP/warp-amd.pth"
echo "$GC_DIR"   > "$SP/garmentcode-root.pth"
echo "wrote $SP/warp-amd.pth and garmentcode-root.pth"

# ---------------------------------------------------------------- 6. system.json
log "6/7 system.json"
if [ -f "$GC_DIR/system.json" ]; then echo "exists, leaving untouched"
else cp "$GC_DIR/system.template.json" "$GC_DIR/system.json"; echo "created $GC_DIR/system.json from template"; fi

# env.sh
{
  echo "# generated by setup_amd.sh -- source this before running anything"
  echo "export ROCM_PATH=\"$ROCM_PATH\""
  echo "export LD_LIBRARY_PATH=\"\$ROCM_PATH/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
  [ -n "$CACHE_DIR" ] && echo "export WARP_CACHE_PATH=\"$CACHE_DIR\""
  echo "export PYTHONDONTWRITEBYTECODE=1   # keep the checkouts free of __pycache__"
  echo "export MPLCONFIGDIR=\"\${MPLCONFIGDIR:-\${TMPDIR:-/tmp}/mpl-\$USER}\""
  echo "source \"$VENV/bin/activate\""
  echo "# GarmentCode scripts must be run with cwd = $GC_DIR"
} > "$ENV_FILE"
echo "wrote $ENV_FILE"

# ---------------------------------------------------------------- 7. smoke test
if [ $SKIP_SMOKE = 1 ]; then log "7/7 smoke test skipped"; exit 0; fi
log "7/7 Smoke test"
( set +u; . "$ENV_FILE"; set -u
  cd "$GC_DIR"
  # Warp kernels cannot be defined from stdin/exec (needs source file), so write a temp file
  SMOKE=$(mktemp -d)/smoke_test.py
  cat > "$SMOKE" <<'PY'
import sys
import numpy as np
import warp as wp
print("warp", wp.config.version, "from", wp.__file__)
wp.config.quiet = False
wp.init()
devs = wp.get_devices()
print("devices:", [str(d) for d in devs])
gpu = [d for d in devs if d.is_cuda]
if not gpu:
    sys.exit("SMOKE FAIL: no GPU device visible to Warp (check LD_LIBRARY_PATH/ROCM_PATH, amdgpu driver, render-group membership)")
d = gpu[0]
print("gpu:", d.name, "is_hip", getattr(d, "is_hip", "n/a"))
if not getattr(d, "is_hip", False):
    sys.exit("SMOKE FAIL: GPU device is not a HIP device")

@wp.kernel
def k(a: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(a, 0, 1.0)

a = wp.zeros(1, dtype=float, device=d)
wp.launch(k, dim=1000, inputs=[a], device=d)
wp.synchronize_device(d)
v = float(a.numpy()[0])
print("kernel result:", v)
if v != 1000.0:
    sys.exit("SMOKE FAIL: kernel returned %s, expected 1000" % v)
import warp.sim
import warp.collision.panel_assignment
print("import warp.sim OK")
import pygarment
print("import pygarment OK:", pygarment.__file__)
print("SMOKE TEST PASSED")
PY
  "$VENV/bin/python" "$SMOKE"
) || die "smoke test failed (see above and SETUP.md troubleshooting)"

log "Done. Next:  source $ENV_FILE && cd $GC_DIR"
