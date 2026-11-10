"""Run reference/scripts/dump_harness.py unchanged, with optional DIAGNOSTIC variants (ws8).

  python gpu_harness_variant.py [--no-fma] [--bvh median|lbvh] -- <dump_harness.py args...>

--no-fma : compile the warp.sim kernel modules with fuse_fp=False (no FMA contraction on HIP;
           the CPU build does not contract either).  Diagnostic only.
--bvh X  : force bvh_constructor=X for every wp.Mesh (default: port default, median on CPU,
           lbvh on GPU).
"""
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
sep = argv.index("--")
opts, rest = argv[:sep], argv[sep + 1:]

import warp as wp  # noqa: E402

if "--no-fma" in opts:
    sys.path.insert(0, here)
    import gpu_common
    gpu_common.disable_fma()
    print("[harness_variant] fuse_fp=False for", gpu_common.SIM_KERNEL_MODULES)
if "--bvh" in opts:
    bvh = opts[opts.index("--bvh") + 1]
    orig = wp.Mesh.__init__

    def mesh_init(self, *a, **kw):
        kw["bvh_constructor"] = bvh
        return orig(self, *a, **kw)
    wp.Mesh.__init__ = mesh_init
    print("[harness_variant] bvh_constructor forced to", bvh)

harness = os.environ.get("DUMP_HARNESS", os.path.join(here, "dump_harness.py"))
if not os.path.exists(harness):
    harness = os.path.join(here, "dump_harness.py")
sys.argv = [harness] + rest
runpy.run_path(harness, run_name="__main__")
