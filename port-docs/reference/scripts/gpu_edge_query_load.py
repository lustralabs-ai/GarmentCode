"""Why are early frames slow on GPU?  Per-spring load of create_self_edge_contacts (ws8 Q1/Q4).

  python gpu_edge_query_load.py --garmentcode-dir GC --inputs INPUTS --frames 0,1,5,12,20,39,60,150 --out JSON

create_self_edge_contacts runs one thread per spring: an AABB query of half-width
edge_len/2 + 3*radius around the spring midpoint, k = number of candidate faces, then for each
of the 3k candidate edges a linear check_edge_exists scan over the slots already written
=> O(k^2) work per thread.  The kernel's GPU time is set by the slowest threads (largest k),
the CPU time by the sum.  This script runs the GPU sim (graph, GarmentCode default) and at the
requested frames reports, per spring: length, k; and aggregates max k, sum k, sum k^2,
and the share of sum k^2 coming from springs longer than 3x the median spring length
(i.e. the stitch springs that close the seams early in the simulation).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_common  # noqa: E402

import warp as wp  # noqa: E402

wp.config.quiet = True


@wp.kernel
def spring_query_load(x: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float), mesh: wp.uint64,
                      isect: wp.array(dtype=int), edges: wp.array(dtype=int),
                      k_out: wp.array(dtype=int), len_out: wp.array(dtype=float)):
    t = wp.tid()
    a = edges[2 * t]
    b = edges[2 * t + 1]
    pa = x[a]
    pb = x[b]
    el = wp.length(pa - pb)
    len_out[t] = el
    if isect[a] == 1 or isect[b] == 1:
        k_out[t] = -1
        return
    r = wp.max(radius[a], radius[b])
    md = el / 2.0 + r * 3.0
    mid = (pa + pb) / 2.0
    dv = wp.vec3(md, md, md)
    q = wp.mesh_query_aabb(mesh, mid - dv, mid + dv)
    f = int(0)
    n = int(0)
    while wp.mesh_query_aabb_next(q, f):
        n += 1
    k_out[t] = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garmentcode-dir", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--frames", default="0,1,5,12,20,39,60,150")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    want = sorted(int(v) for v in a.frames.split(","))
    gpu_common.setup(a.garmentcode_dir)
    cloth, cfg = gpu_common.build_cloth(a.inputs, "cuda:0", a.garmentcode_dir, tag="eql")
    m = cloth.model
    ns = m.spring_count
    k = wp.zeros(ns, dtype=int, device="cuda:0")
    ln = wp.zeros(ns, dtype=float, device="cuda:0")
    rest = m.spring_rest_length.numpy()
    rows = []

    def measure(fr):
        # state at the start of frame fr's collide(): state_0 after the previous frame
        wp.launch(spring_query_load, ns, inputs=[cloth.state_0.particle_q, m.particle_radius, m.particle_shape.id,
                                                 m.particle_self_intersection_particle, m.spring_indices, k, ln],
                  device="cuda:0")
        kk = k.numpy().astype(np.int64)
        ll = ln.numpy()
        valid = kk >= 0
        kv = np.where(valid, kk, 0)
        k2 = kv.astype(np.float64) ** 2
        med = np.median(ll)
        long_ = ll > 3 * med
        top = np.argsort(-kv)[:5]
        r = {"frame": fr, "springs": int(ns), "skipped_intersecting": int((~valid).sum()),
             "len_median": float(med), "len_max": float(ll.max()), "n_long_gt3med": int(long_.sum()),
             "k_max": int(kv.max()), "k_mean": float(kv.mean()), "k_p99": float(np.percentile(kv, 99)),
             "sum_k": int(kv.sum()), "sum_k2": float(k2.sum()), "max_k2": float(k2.max()),
             "share_sum_k2_long": float(k2[long_].sum() / max(k2.sum(), 1)),
             "top5": [{"spring": int(i), "k": int(kv[i]), "len": float(ll[i]), "rest": float(rest[i])} for i in top]}
        rows.append(r)
        print(json.dumps({kk_: v for kk_, v in r.items() if kk_ != "top5"}), flush=True)

    for fr in range(max(want) + 1):
        if fr in want:
            measure(fr)
        cloth.frame = fr
        cloth.run_frame()
        cloth.current_verts = np.array(cloth.current_verts, copy=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
