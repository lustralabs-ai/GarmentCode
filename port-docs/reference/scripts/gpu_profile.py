"""GPU per-frame / per-kernel profile of the GarmentCode cloth sim (ws8 question 1 + 4).

  python gpu_profile.py --garmentcode-dir GC --inputs INPUTS --mode graph|eager|kernels
                        [--frames N] [--device cuda:0] [--bvh median|lbvh] [--out JSON]

mode graph   : GarmentCode default (graph capture on cuda); per-frame wall time of run_frame
               (run_frame ends with a device->host copy of particle_q, so it is synchronised).
mode eager   : no capture, every kernel launched eagerly; same timing.
mode kernels : eager + wp.timing_begin(TIMING_KERNEL|TIMING_KERNEL_BUILTIN)/timing_end per frame
               (synchronised), giving the GPU time of every kernel (incl. native refit kernels).
Every frame also records the contact counts produced by collide():
soft (body) contacts, edge-edge self contacts, point-triangle self contacts, globally
intersecting particles, and the per-element maxima vs the capacity (overflow check).
Static-equilibrium check as in dump_harness (copy mode); stops when static.
Works on --device cpu too (mode graph == eager there).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_common  # noqa: E402

import warp as wp  # noqa: E402

wp.config.quiet = True


def counts(model):
    r = {}
    m = model
    if getattr(m, "soft_contact_count", None) is not None:
        r["soft"] = int(m.soft_contact_count.numpy()[0])
    if getattr(m, "edge_contact_count", None) is not None:
        e = m.edge_contact_count.numpy()
        r["edge"] = int(e.sum())
        r["edge_max_per"] = int(e.max()) if e.size else 0
        r["edge_cap"] = int(m.edge_contact_max)
    if getattr(m, "point_tri_contact_count", None) is not None:
        p = m.point_tri_contact_count.numpy()
        r["pt"] = int(p.sum())
        r["pt_max_per"] = int(p.max()) if p.size else 0
        r["pt_cap"] = int(m.point_tri_contact_max)
    if getattr(m, "particle_global_self_intersection_count", None) is not None:
        r["isect"] = int(m.particle_global_self_intersection_count.numpy().sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garmentcode-dir", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--mode", choices=["graph", "eager", "kernels"], default="graph")
    ap.add_argument("--frames", type=int, default=100000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bvh", default=None)
    ap.add_argument("--no-static-stop", action="store_true")
    ap.add_argument("--out", required=True, help="absolute path of the JSON output")
    a = ap.parse_args()
    out = os.path.abspath(a.out)

    gpu_common.setup(a.garmentcode_dir)
    t0 = time.perf_counter()
    cloth, cfg = gpu_common.build_cloth(a.inputs, a.device, a.garmentcode_dir,
                                        graph=(None if a.mode == "graph" else False), bvh=a.bvh, tag="prof")
    t_build = time.perf_counter() - t0
    rec = {"args": vars(a), "build_s": t_build, "sim_use_graph": bool(cloth.sim_use_graph),
           "particles": int(cloth.model.particle_count), "springs": int(cloth.model.spring_count),
           "substeps": int(cloth.sim_substeps), "frames": []}
    kern_tot = defaultdict(float)
    kern_cnt = defaultdict(int)
    filt = wp.TIMING_KERNEL | getattr(wp, "TIMING_KERNEL_BUILTIN", 0)
    n_run = 0
    for fr in range(a.frames):
        cloth.frame = fr
        if a.mode == "kernels":
            wp.timing_begin(cuda_filter=filt)
        ts = time.perf_counter()
        cloth.run_frame()
        wall = time.perf_counter() - ts
        row = {"frame": fr, "wall_s": wall}
        if a.mode == "kernels":
            res = wp.timing_end()
            per = defaultdict(float)
            for r in res:
                per[r.name] += r.elapsed
                kern_tot[r.name] += r.elapsed
                kern_cnt[r.name] += 1
            row["gpu_kernel_ms"] = sum(per.values())
            row["kernels_ms"] = dict(per)
        row.update(counts(cloth.model))
        rec["frames"].append(row)
        cloth.current_verts = np.array(cloth.current_verts, copy=True)
        n_run = fr + 1
        if fr % 20 == 0:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if k != "kernels_ms"}), flush=True)
        if not a.no_static_stop and fr >= cfg.zero_gravity_steps and fr >= cfg.min_sim_steps:
            st, _ = cloth.is_static()
            if st:
                break
    rec["frames_run"] = n_run
    w = np.array([f["wall_s"] for f in rec["frames"]])
    rec["wall_summary"] = {"total_s": float(w.sum()), "median_ms": float(np.median(w) * 1e3),
                           "frame0_ms": float(w[0] * 1e3),
                           "median_1_39_ms": float(np.median(w[1:40]) * 1e3) if len(w) > 2 else None,
                           "median_40plus_ms": float(np.median(w[40:]) * 1e3) if len(w) > 41 else None}
    if kern_tot:
        rec["kernel_total_ms"] = dict(sorted(kern_tot.items(), key=lambda kv: -kv[1]))
        rec["kernel_launches"] = dict(kern_cnt)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec["wall_summary"]))
    if kern_tot:
        tot = sum(kern_tot.values())
        for k, v in list(rec["kernel_total_ms"].items())[:25]:
            print(f"{v:10.1f} ms {100 * v / tot:5.1f}%  n={kern_cnt[k]:6d}  {k}")


if __name__ == "__main__":
    main()
