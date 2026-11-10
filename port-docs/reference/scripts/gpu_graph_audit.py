"""Graph-capture audit (ws8 question 1).

  python gpu_graph_audit.py --garmentcode-dir GC --inputs INPUTS [--frames N] [--out DIR]

1. Counts the wp.launch calls (and their dims) issued while GarmentCode captures its
   per-frame graph, dumps the captured HIP graph with wp.capture_debug_dot_print and
   counts its kernel / memset / memcpy nodes.  Every wp.launch with dim>0 must appear
   as a kernel node; mesh.refit() adds native kernel nodes.
2. Runs N frames on cuda:0 with graph capture (GarmentCode default) and N frames with
   eager launches (no capture), each from a freshly built Cloth, plus a second
   graph run, and compares positions frame by frame (graph vs graph = GPU run-to-run
   noise; graph vs eager = must be of the same size if the graph executes everything).
   Control: the eager run with one substep fewer in frame 0 (shows what skipped work
   looks like).
3. Error checks: wp.config.verify_cuda=True for the eager run (checks after every
   launch); hipDeviceSynchronize + hipGetLastError after every graph launch.
4. Refit check on the self-collision mesh after the run: each triangle's own AABB
   query (mesh_query_aabb) must return that triangle (stale BVH bounds would miss it).
"""
import argparse
import ctypes
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_common  # noqa: E402

import warp as wp  # noqa: E402

wp.config.quiet = True


@wp.kernel
def self_aabb_query(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), idx: wp.array(dtype=wp.int32),
                    found: wp.array(dtype=wp.int32), ncand: wp.array(dtype=wp.int32)):
    f = wp.tid()
    a = pts[idx[3 * f]]
    b = pts[idx[3 * f + 1]]
    c = pts[idx[3 * f + 2]]
    lo = wp.min(wp.min(a, b), c)
    hi = wp.max(wp.max(a, b), c)
    q = wp.mesh_query_aabb(mesh, lo, hi)
    i = int(0)
    n = int(0)
    while wp.mesh_query_aabb_next(q, i):
        n += 1
        if i == f:
            found[f] = 1
    ncand[f] = n


def hip_lib():
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    return None


def hip_check(lib):
    if lib is None:
        return "n/a"
    e1 = lib.hipDeviceSynchronize()
    e2 = lib.hipGetLastError()
    return (e1, e2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garmentcode-dir", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default="/tmp/claude-1000/ws8/graph_audit")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hip = hip_lib()
    gpu_common.setup(a.garmentcode_dir)
    from pygarment.meshgen.garment import Cloth

    # ---- 1. launches issued during capture
    rec = {"on": False, "launches": []}
    orig_launch = wp.launch

    def counting_launch(kernel, dim, *args, **kw):
        if rec["on"]:
            d = dim if isinstance(dim, int) else int(np.prod(dim))
            rec["launches"].append((kernel.key, d))
        return orig_launch(kernel, dim, *args, **kw)

    wp.launch = counting_launch
    orig_cg = Cloth.create_graph
    captures = []

    def cg(self):
        rec["on"] = True
        rec["launches"] = []
        orig_cg(self)
        rec["on"] = False
        captures.append(list(rec["launches"]))
        p = out / f"graph_{len(captures)}.dot"
        wp.capture_debug_dot_print(self.graph, str(p), verbose=False)

    Cloth.create_graph = cg
    g1, cfg = gpu_common.build_cloth(a.inputs, "cuda:0", a.garmentcode_dir, tag="g1")
    Cloth.create_graph = orig_cg
    rep = {}
    L = captures[0]
    cnt = Counter(k for k, d in L)
    rep["capture_launches_total"] = len(L)
    rep["capture_launches_dim0"] = sum(1 for k, d in L if d == 0)
    rep["capture_launches_by_kernel"] = dict(cnt)
    dot = (out / "graph_1.dot").read_text()
    # node labels in HIP dot output
    nodes = re.findall(r'^\s*"?(\w+)"?\s*\[.*?label="([^"]*)"', dot, flags=re.M)
    rep["dot_bytes"] = len(dot)
    kinds = Counter()
    kern_names = Counter()
    for _, lab in nodes:
        l0 = lab.split("\\n")[0].split("|")[0]
        if "MEMSET" in lab.upper():
            kinds["memset"] += 1
        elif "MEMCPY" in lab.upper():
            kinds["memcpy"] += 1
        elif "KERNEL" in lab.upper() or "_cuda_kernel_forward" in lab or "kernel" in lab.lower():
            kinds["kernel"] += 1
            m = re.search(r"(\w+?)_[0-9a-f]{6,}_cuda_kernel_forward", lab)
            kern_names[m.group(1) if m else l0[:60]] += 1
        else:
            kinds["other:" + l0[:30]] += 1
    rep["dot_nodes"] = len(nodes)
    rep["dot_node_kinds"] = dict(kinds)
    rep["dot_kernel_names"] = dict(kern_names)
    print(json.dumps(rep, indent=1))

    # ---- 2. graph vs eager vs graph
    wp.launch = orig_launch
    g2, _ = gpu_common.build_cloth(a.inputs, "cuda:0", a.garmentcode_dir, tag="g2")
    e1, _ = gpu_common.build_cloth(a.inputs, "cuda:0", a.garmentcode_dir, graph=False, tag="e1")
    assert g1.sim_use_graph and g2.sim_use_graph and not e1.sim_use_graph
    init = [c.state_0.particle_q.numpy() for c in (g1, g2, e1)]
    rep["init_identical"] = bool(all(np.array_equal(init[0], x) for x in init))

    traj = {"g1": [], "g2": [], "e1": []}
    errs = []
    times = {"g1": [], "g2": [], "e1": []}
    for name, c in (("g1", g1), ("g2", g2), ("e1", e1)):
        wp.config.verify_cuda = (name == "e1")
        for fr in range(a.frames):
            c.frame = fr
            t0 = time.perf_counter()
            c.run_frame()
            times[name].append(time.perf_counter() - t0)
            if name != "e1":
                errs.append((name, fr, hip_check(hip)))
            traj[name].append(c.state_0.particle_q.numpy().copy())
        wp.config.verify_cuda = False
    rep["hip_errors_after_graph_launch"] = [e for e in errs if e[2] not in ((0, 0), "n/a")]

    def d(x, y):
        return np.linalg.norm(x.astype(np.float64) - y, axis=1)

    rows = []
    for fr in range(a.frames):
        gg = d(traj["g1"][fr], traj["g2"][fr])
        ge = d(traj["g1"][fr], traj["e1"][fr])
        rows.append({"frame": fr, "g1_g2_max": float(gg.max()), "g1_g2_bitequal": bool(gg.max() == 0),
                     "g1_e1_max": float(ge.max()), "g1_e1_rms": float(np.sqrt((ge ** 2).mean())),
                     "g1_g2_rms": float(np.sqrt((gg ** 2).mean()))})
        print(rows[-1])
    rep["per_frame"] = rows
    rep["frame_times_ms"] = {k: [round(t * 1e3, 2) for t in v] for k, v in times.items()}

    # control: one substep fewer in frame 0 (eager)
    c, _ = gpu_common.build_cloth(a.inputs, "cuda:0", a.garmentcode_dir, graph=False, tag="ctl")
    c.sim_substeps -= 1
    c.frame = 0
    c.run_frame()
    c.sim_substeps += 1
    ctl = d(traj["g1"][0], c.state_0.particle_q.numpy())
    rep["control_one_substep_fewer_frame0_max"] = float(ctl.max())
    rep["control_one_substep_fewer_frame0_rms"] = float(np.sqrt((ctl ** 2).mean()))

    # ---- 4. refit completeness of the self-collision mesh
    for name, c in (("g1", g1), ("e1", e1)):
        m = c.model.particle_shape
        nf = len(m.indices) // 3
        found = wp.zeros(nf, dtype=wp.int32, device="cuda:0")
        ncand = wp.zeros(nf, dtype=wp.int32, device="cuda:0")
        wp.launch(self_aabb_query, dim=nf, inputs=[m.id, m.points, m.indices, found, ncand], device="cuda:0")
        # also check the mesh points equal particle_q (replace_mesh_points ran)
        same = np.array_equal(m.points.numpy(), c.state_0.particle_q.numpy())
        rep[f"refit_{name}"] = {"faces": nf, "self_found": int(found.numpy().sum()),
                                "mean_candidates": float(ncand.numpy().mean()),
                                "mesh_points_equal_particle_q": bool(same)}
    print(json.dumps({k: v for k, v in rep.items() if k != "per_frame"}, indent=1))
    (out / "report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
