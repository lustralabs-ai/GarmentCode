"""Per-kernel CPU-vs-GPU fidelity check (ws8 question 2).

  python gpu_kernel_diff.py --garmentcode-dir GC --inputs INPUTS [--warm-frames F]
         [--gpu-bvh median|lbvh] [--frames 1] [--out report.json]

Builds the same garment twice in one process: on cpu (port default BVH = median) and on
cuda:0 (no graph capture; BVH = --gpu-bvh).  Optionally advances the CPU sim F frames.
Then, for --frames frames, the CPU sim runs normally and every wp.launch it issues is
*replayed in isolation on the GPU*: before the GPU launch every array argument is
overwritten with the CPU's pre-launch contents (uint64 mesh-id arrays excepted), the
GPU self-collision mesh is given the CPU's current points and refit, then the same
kernel is launched on cuda:0 with the GPU twins of all arguments.  After both, every
array argument is compared.  So each kernel is tested from bit-identical inputs and
errors cannot propagate from one launch to the next.

Contact-generation kernels write into slots reserved with atomic_add, so their slot
order is arbitrary on GPU; those outputs are compared as sets keyed by
(particle,shape) / (particle,face) / (edge,edge) instead of by slot.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_common  # noqa: E402

import warp as wp  # noqa: E402

wp.config.quiet = True


def arrays_of(obj):
    return {k: v for k, v in vars(obj).items() if isinstance(v, wp.array)}


class Replayer:
    def __init__(self, cpu, gpu):
        self.cpu, self.gpu = cpu, gpu
        self.map = {}
        self.names = {}
        for a, b, pre in ((cpu.model, gpu.model, "model"), (cpu.state_0, gpu.state_0, "state_0"),
                          (cpu.state_1, gpu.state_1, "state_1")):
            A, B = arrays_of(a), arrays_of(b)
            for k, v in A.items():
                if k in B and B[k] is not None and B[k].shape == v.shape and B[k].dtype == v.dtype:
                    self.map[id(v)] = B[k]
                    self.names[id(v)] = f"{pre}.{k}"
        self.struct_map = {}
        for k, v in vars(cpu.model).items():
            if hasattr(v, "_ctype") and not isinstance(v, wp.array):   # warp struct instance
                self.struct_map[id(v)] = getattr(gpu.model, k)
        print("struct args mapped:", [k for k, v in vars(cpu.model).items() if id(v) in self.struct_map])
        self.mesh_map = {cpu.model.particle_shape.id: gpu.model.particle_shape.id}
        # body/reference meshes: shape_geo.source and cloth_reference_shape_ids hold mesh ids
        cs = cpu.model.shape_geo.source.numpy()
        gs = gpu.model.shape_geo.source.numpy()
        for x, y in zip(cs, gs):
            self.mesh_map[int(x)] = int(y)
        self.records = []
        self.active = False
        self.unmapped = Counter()

    def sync_mesh(self):
        cm, gm = self.cpu.model.particle_shape, self.gpu.model.particle_shape
        wp.copy(gm.points, cm.points)
        gm.refit()

    def gpu_arg(self, x):
        if isinstance(x, wp.array):
            g = self.map.get(id(x))
            if g is None:
                self.unmapped[str(x.shape) + str(x.dtype)] += 1
                g = wp.array(x.numpy(), dtype=x.dtype, device="cuda:0")
                self.map[id(x)] = g
            return g
        if id(x) in self.struct_map:
            return self.struct_map[id(x)]
        if hasattr(x, "_ctype"):
            raise RuntimeError(f"unmapped struct argument {type(x)}")
        return x

    def launch(self, orig, kernel, dim, inputs=(), outputs=(), **kw):
        if not self.active:
            return orig(kernel, dim, inputs=inputs, outputs=outputs, **kw)
        args = list(inputs) + list(outputs)
        # translate mesh ids among scalar args
        gargs = []
        for x in args:
            if isinstance(x, (int, np.integer)) and int(x) in self.mesh_map and int(x) > 1 << 20:
                gargs.append(self.mesh_map[int(x)])
            else:
                gargs.append(self.gpu_arg(x))
        # pre-launch: copy CPU contents into the GPU twins
        for x, g in zip(args, gargs):
            if isinstance(x, wp.array) and x.dtype != wp.uint64:
                wp.copy(g, x)
        self.sync_mesh()
        wp.synchronize()
        orig(kernel, dim, inputs=inputs, outputs=outputs, **kw)
        gkw = dict(kw)
        gkw["device"] = "cuda:0"
        orig(kernel, dim, inputs=gargs[:len(inputs)], outputs=gargs[len(inputs):], **gkw)
        wp.synchronize()
        names = [self.names.get(id(x), f"arg{i}") if isinstance(x, wp.array) else None
                 for i, x in enumerate(args)]
        rec = compare(kernel.key, args, gargs, names, self.cpu.model)
        rec["dim"] = int(np.prod(dim)) if not isinstance(dim, int) else dim
        self.records.append(rec)


def fstats(c, g):
    c = c.astype(np.float64)
    g = g.astype(np.float64)
    d = np.abs(c - g)
    scale = float(np.abs(c).max()) if c.size else 0.0
    rel = d / np.maximum(np.abs(c), 1e-30)
    nz = d > 0
    out = {"n": int(c.size), "n_diff": int(nz.sum()), "max_abs": float(d.max()) if d.size else 0.0,
           "scale": scale}
    # relative error w.r.t. per-element magnitude, ignoring tiny elements (< 1e-6 * scale)
    big = np.abs(c) > 1e-6 * max(scale, 1e-30)
    out["max_rel_big"] = float(rel[big].max()) if big.any() else 0.0
    out["max_abs_over_scale"] = out["max_abs"] / scale if scale > 0 else (0.0 if out["max_abs"] == 0 else np.inf)
    if out["n_diff"]:
        i = int(np.argmax(d.reshape(d.shape[0], -1).max(axis=1)) if d.ndim > 1 else np.argmax(d))
        out["worst_index"] = i
        out["worst_cpu"] = c[i].tolist()
        out["worst_gpu"] = g[i].tolist()
    return out


def contact_sets(kind, A, model):
    """A: dict name->numpy of post-launch outputs.  Return dict key->value-vector."""
    S = {}
    if kind == "soft":
        n = min(int(A["model.soft_contact_count"][0]), model.soft_contact_max)
        for i in range(n):
            key = (int(A["model.soft_contact_particle"][i]), int(A["model.soft_contact_shape"][i]))
            S[key] = np.concatenate([A["model.soft_contact_body_pos"][i], A["model.soft_contact_body_vel"][i],
                                     A["model.soft_contact_normal"][i]])
    elif kind == "pt":
        n = min(int(A["model.point_tri_contact_count"][0]), model.point_tri_contact_max)
        f = A["model.point_tri_contact_filter"][:n]
        p = A["model.point_tri_contact_pairs"][:n]
        s = A["model.point_tri_contact_sidedness"][:n]
        for i in np.nonzero(~f)[0]:
            key = (int(p[i][0]), int(p[i][1]))
            S[key] = np.array([float(s[i])])
    elif kind == "ee":
        n = min(int(A["model.edge_contact_count"][0]), model.edge_contact_max)
        f = A["model.edge_contact_filter"][:n]
        p = A["model.edge_contact_pairs"][:n]
        nn = A["model.edge_contact_normal"][:n]
        for i in np.nonzero(~f)[0]:
            a, b, c, d = (int(v) for v in p[i])
            key = (a, b, min(c, d), max(c, d))
            if key in S:  # duplicate kept from either direction; record multiplicity
                key = key + (len([k for k in S if k[:4] == key]),)
            S[key] = nn[i]
    return S


SET_KERNELS = {
    "create_soft_contacts": ("soft", {"model.soft_contact_particle", "model.soft_contact_shape",
                                      "model.soft_contact_body_pos", "model.soft_contact_body_vel",
                                      "model.soft_contact_normal"}),
    "create_self_point_triangle_contacts": ("pt", {"model.point_tri_contact_pairs", "model.point_tri_contact_filter",
                                                   "model.point_tri_contact_sidedness"}),
    "create_self_edge_contacts": ("ee", {"model.edge_contact_pairs", "model.edge_contact_filter",
                                         "model.edge_contact_normal"}),
}


def compare(kname, args, gargs, names, model):
    rec = {"kernel": kname, "arrays": {}}
    C, G = {}, {}
    for x, g, nm in zip(args, gargs, names):
        if not isinstance(x, wp.array) or x.dtype == wp.uint64:
            continue
        c_np, g_np = x.numpy().copy(), g.numpy().copy()
        C[nm], G[nm] = c_np, g_np
    setk = SET_KERNELS.get(kname)
    for nm in C:
        if setk and nm in setk[1]:
            continue
        c_np, g_np = C[nm], G[nm]
        if c_np.dtype.kind == "f":
            rec["arrays"][nm] = fstats(c_np, g_np)
        else:
            nd = int((c_np != g_np).sum())
            r = {"n": int(c_np.size), "n_diff": nd}
            if nd and c_np.ndim == 1 and c_np.size < 10:
                r["cpu"], r["gpu"] = c_np.tolist(), g_np.tolist()
            rec["arrays"][nm] = r
    if setk:
        kind = setk[0]
        Sc, Sg = contact_sets(kind, C, model), contact_sets(kind, G, model)
        common = set(Sc) & set(Sg)
        only_c = sorted(set(Sc) - set(Sg))
        only_g = sorted(set(Sg) - set(Sc))
        maxd = 0.0
        worst = None
        for k in common:
            dd = float(np.abs(Sc[k].astype(np.float64) - Sg[k]).max())
            if dd > maxd:
                maxd, worst = dd, k
        rec["contacts"] = {"kind": kind, "cpu": len(Sc), "gpu": len(Sg), "common": len(common),
                           "only_cpu": len(only_c), "only_gpu": len(only_g),
                           "only_cpu_examples": [list(k) for k in only_c[:8]],
                           "only_gpu_examples": [list(k) for k in only_g[:8]],
                           "max_value_diff_common": maxd, "worst_key": list(worst) if worst else None}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garmentcode-dir", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--warm-frames", type=int, default=0)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--gpu-bvh", default="median")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-fma", action="store_true", help="diagnostic: compile sim kernels with fuse_fp=False")
    a = ap.parse_args()

    gpu_common.setup(a.garmentcode_dir)
    if a.no_fma:
        gpu_common.disable_fma()
    gpu, cfg = gpu_common.build_cloth(a.inputs, "cuda:0", a.garmentcode_dir, graph=False, bvh=a.gpu_bvh, tag="gpu")
    cpu, _ = gpu_common.build_cloth(a.inputs, "cpu", a.garmentcode_dir, tag="cpu")
    wp.set_device("cpu")

    # static model arrays identical after construction?
    model_diff = {}
    for k, v in arrays_of(cpu.model).items():
        g = getattr(gpu.model, k, None)
        if isinstance(g, wp.array) and v.dtype != wp.uint64 and g.shape == v.shape:
            cn, gn = v.numpy(), g.numpy()
            if not np.array_equal(cn, gn):
                model_diff[k] = float(np.abs(cn.astype(np.float64) - gn).max()) if cn.dtype.kind == "f" else int((cn != gn).sum())
    print("model arrays differing after construction (before any frame):", model_diff)

    t0 = time.time()
    for fr in range(a.warm_frames):
        cpu.frame = fr
        cpu.run_frame()
        cpu.current_verts = np.array(cpu.current_verts, copy=True)
    print(f"warmed CPU {a.warm_frames} frames in {time.time() - t0:.1f}s")
    # GPU host-side flags that update() changes (gravity, attachment) must follow the CPU's
    gpu.model.gravity = cpu.model.gravity
    gpu.model.attachment_constraint = cpu.model.attachment_constraint

    rp = Replayer(cpu, gpu)
    orig = wp.launch
    wp.launch = lambda kernel, dim, *args, **kw: rp.launch(orig, kernel, dim, *args, **kw)
    rp.active = True
    for fr in range(a.warm_frames, a.warm_frames + a.frames):
        cpu.frame = fr
        cpu.run_frame()
    rp.active = False
    wp.launch = orig
    print("unmapped arrays:", dict(rp.unmapped))

    # summary table
    agg = defaultdict(lambda: {"launches": 0, "worst_rel": 0.0, "worst_abs_over_scale": 0.0, "int_diffs": 0,
                               "contact_only_cpu": 0, "contact_only_gpu": 0, "contact_value_diff": 0.0,
                               "worst_array": None})
    for r in rp.records:
        A = agg[r["kernel"]]
        A["launches"] += 1
        for nm, s in r["arrays"].items():
            if "max_rel_big" in s:
                if s["max_abs_over_scale"] > A["worst_abs_over_scale"]:
                    A["worst_abs_over_scale"] = s["max_abs_over_scale"]
                    A["worst_array"] = nm
                A["worst_rel"] = max(A["worst_rel"], s["max_rel_big"])
            else:
                A["int_diffs"] += s["n_diff"]
        if "contacts" in r:
            c = r["contacts"]
            A["contact_only_cpu"] += c["only_cpu"]
            A["contact_only_gpu"] += c["only_gpu"]
            A["contact_value_diff"] = max(A["contact_value_diff"], c["max_value_diff_common"])
    print(f"\n{'kernel':40s} {'n':>3s} {'max|d|/scale':>12s} {'max rel':>10s} {'int diffs':>9s} "
          f"{'only_cpu':>8s} {'only_gpu':>8s} {'set val d':>9s}  worst array")
    for k, A in agg.items():
        print(f"{k:40s} {A['launches']:3d} {A['worst_abs_over_scale']:12.3e} {A['worst_rel']:10.3e} "
              f"{A['int_diffs']:9d} {A['contact_only_cpu']:8d} {A['contact_only_gpu']:8d} "
              f"{A['contact_value_diff']:9.2e}  {A['worst_array']}")
    for r in rp.records:
        if "contacts" in r:
            print(r["kernel"], json.dumps(r["contacts"]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "model_diff": model_diff, "records": rp.records,
                       "summary": agg}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
