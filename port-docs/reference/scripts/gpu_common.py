"""Shared helpers for the GPU-validation scripts (workstream 8).

build_cloth(inputs_dir, device, gc_dir, graph=None, bvh=None) builds a GarmentCode
`Cloth` from a dump_harness `inputs` directory on the requested Warp device, exactly
as reference/scripts/dump_harness.py does, without touching the GarmentCode tree.

  graph=None  : GarmentCode default (capture a CUDA/HIP graph iff device.is_cuda)
  graph=False : no capture on a CUDA device: Cloth.create_graph is replaced by a
                no-op for the duration of construction and sim_use_graph set False,
                so update() launches every kernel eagerly (GarmentCode unmodified).
  bvh="median"/"lbvh"/None : override the bvh_constructor of every wp.Mesh built
                while constructing the Cloth (None = port default: median on CPU,
                lbvh on GPU).

Several Cloth objects (e.g. one on cpu and one on cuda:0) may live in one process.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_WORK = None


def setup(gc_dir):
    """Create a private cwd with system.json / assets symlink (once per process)."""
    global _WORK
    if _WORK is not None:
        return _WORK
    gc = Path(gc_dir).resolve()
    work = Path(tempfile.mkdtemp(prefix="gch_ws8_"))
    (work / "bodies").mkdir()
    system = {"output": str(work / "out"), "datasets_path": "", "datasets_sim": "",
              "sim_configs_path": str(gc / "assets" / "Sim_props"),
              "bodies_default_path": str(work / "bodies"), "body_samples_path": ""}
    (work / "system.json").write_text(json.dumps(system))
    (work / "assets").symlink_to(gc / "assets")
    os.chdir(work)
    sys.path.insert(0, str(gc))
    _WORK = work
    import warp as wp
    if os.environ.get("WARP_CACHE_PATH"):
        wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    wp.init()
    return work


SIM_KERNEL_MODULES = ("warp.sim.collide", "warp.sim.integrator_xpbd", "warp.sim.integrator_euler",
                      "warp.sim.utils", "warp.sim.inertia", "warp.sim.particles", "warp.collision.panel_assignment")


def disable_fma():
    """DIAGNOSTIC ONLY: compile the sim kernel modules with fuse_fp=False (-ffp-contract=off on
    HIP), i.e. without FMA contraction, like the CPU build.  Changes module hashes (recompile)."""
    import importlib
    import warp as wp
    for m in SIM_KERNEL_MODULES:
        wp.set_module_options({"fuse_fp": False}, module=importlib.import_module(m))


def build_cloth(inputs, device, gc_dir, graph=None, bvh=None, tag=None):
    work = setup(gc_dir)
    import warp as wp
    inp = Path(inputs).resolve()
    meta = json.loads((inp / "inputs_meta.json").read_text())
    name = meta["garment"]
    body_stem = meta["body_name"]
    bodies = work / "bodies"
    shutil.copy(inp / "body.obj", bodies / f"{body_stem}.obj")
    shutil.copy(inp / "body_measurements.yaml", bodies / f"{body_stem}.yaml")
    seg = "smpl_vert_segmentation.json" if meta["smpl_body"] else "ggg_body_segmentation.json"
    shutil.copy(inp / "body_segmentation.json", bodies / seg)

    import pygarment.data_config as data_config
    from pygarment.meshgen.garment import Cloth
    from pygarment.meshgen.sim_config import PathCofig, SimConfig

    props = data_config.Properties(str(inp / "sim_props.yaml"))
    out = work / "out" / (tag or f"{name}_{str(device).replace(':', '')}")
    paths = PathCofig(in_element_path=inp, out_path=str(out), in_name=name, body_name=body_stem,
                      smpl_body=meta["smpl_body"], add_timestamp=False)
    for src, dst in {"boxmesh.obj": paths.g_box_mesh, "sim_segmentation.txt": paths.g_mesh_segmentation,
                     "orig_lens.pickle": paths.g_orig_edge_len, "vertex_labels.yaml": paths.g_vert_labels,
                     "specification.json": paths.g_specs, "body_measurements.yaml": paths.body_mes}.items():
        if (inp / src).exists():
            shutil.copy(inp / src, dst)
    cfg = SimConfig(props["sim"]["config"])

    wp.set_device(device)
    orig_mesh_init = wp.Mesh.__init__
    orig_create_graph = Cloth.create_graph
    if bvh is not None:
        def mesh_init(self, *a, **kw):
            kw["bvh_constructor"] = bvh
            return orig_mesh_init(self, *a, **kw)
        wp.Mesh.__init__ = mesh_init
    if graph is False:
        Cloth.create_graph = lambda self: None
    try:
        cloth = Cloth(name, cfg, paths, caching=False)
    finally:
        wp.Mesh.__init__ = orig_mesh_init
        Cloth.create_graph = orig_create_graph
    if graph is False:
        cloth.sim_use_graph = False
    return cloth, cfg


def run_frames(cloth, cfg, n, copy_static=True, on_frame=None):
    """Run up to n frames like dump_harness (static check enabled). Returns frames run."""
    for frame in range(n):
        cloth.frame = frame
        cloth.run_frame()
        if copy_static:
            cloth.current_verts = np.array(cloth.current_verts, copy=True)
        if on_frame:
            on_frame(frame)
        if frame >= cfg.zero_gravity_steps and frame >= cfg.min_sim_steps:
            st, _ = cloth.is_static()
            if st:
                return frame + 1
    return n
