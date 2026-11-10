"""Geometric sanity of a dump_harness run directory (ws8 question 3).

  python gpu_geom_sanity.py RUN_DIR [RUN_DIR ...] [--json OUT]

For the final cloth (final_verts.npy) against the body (body_verts/body_faces.npy):
  NaN/Inf, cloth/body bounding boxes, displacement from the initial cloth,
  signed distance of every cloth vertex to the body (Warp CPU mesh query, winding-number sign),
  fraction of vertices inside the body and max penetration depth,
  edge-length stretch relative to the initial cloth (max / 99th pct ratio) -> explosion check.
Units: cm (GarmentCode sim units).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

wp.config.quiet = True


@wp.kernel
def sdist(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    t = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, pts[t], 1.0e6)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        out[t] = wp.length(cp - pts[t]) * q.sign


def edges_of(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def analyse(d: Path):
    fv = np.load(d / "final_verts.npy").astype(np.float64)
    iv = np.load(d / "init_cloth_verts.npy").astype(np.float64)
    cf = np.load(d / "cloth_faces.npy").reshape(-1, 3)
    bv = np.load(d / "body_verts.npy").astype(np.float64)
    bf = np.load(d / "body_faces.npy").reshape(-1, 3)
    r = {"run": str(d), "nan_or_inf": bool(~np.isfinite(fv).all())}
    if r["nan_or_inf"]:
        return r
    r["cloth_bbox"] = [fv.min(0).round(2).tolist(), fv.max(0).round(2).tolist()]
    r["body_bbox"] = [bv.min(0).round(2).tolist(), bv.max(0).round(2).tolist()]
    r["cloth_in_body_bbox_frac"] = float(((fv >= bv.min(0) - 5) & (fv <= bv.max(0) + 5)).all(1).mean())
    r["disp_from_init_max"] = float(np.linalg.norm(fv - iv, axis=1).max())
    mesh = wp.Mesh(points=wp.array(bv.astype(np.float32), dtype=wp.vec3, device="cpu"),
                   indices=wp.array(bf.astype(np.int32).ravel(), dtype=int, device="cpu"),
                   support_winding_number=True)
    out = wp.zeros(len(fv), dtype=float, device="cpu")
    wp.launch(sdist, len(fv), inputs=[mesh.id, wp.array(fv.astype(np.float32), dtype=wp.vec3, device="cpu"), out],
              device="cpu")
    sd = out.numpy().astype(np.float64)  # + outside, - inside (winding number)
    r["dist_to_body_median"] = float(np.median(sd))
    r["dist_to_body_p95"] = float(np.percentile(sd, 95))
    r["inside_frac"] = float((sd < 0).mean())
    r["inside_count_gt_0p1cm"] = int((sd < -0.1).sum())
    r["max_penetration_cm"] = float(max(0.0, -sd.min()))
    e = edges_of(cf)
    l0 = np.linalg.norm(iv[e[:, 0]] - iv[e[:, 1]], axis=1)
    l1 = np.linalg.norm(fv[e[:, 0]] - fv[e[:, 1]], axis=1)
    ok = l0 > 1e-6
    ratio = l1[ok] / l0[ok]
    r["edge_ratio_max"] = float(ratio.max())
    r["edge_ratio_p99"] = float(np.percentile(ratio, 99))
    r["edge_ratio_min"] = float(ratio.min())
    s = d / "summary.json"
    if s.exists():
        js = json.loads(s.read_text())
        for k in ("frames_run", "stop_reason", "self_intersections", "body_intersections"):
            r[k] = js.get(k)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = [analyse(Path(d)) for d in a.runs]
    keys = ["frames_run", "self_intersections", "body_intersections", "nan_or_inf", "inside_frac",
            "max_penetration_cm", "dist_to_body_median", "edge_ratio_max", "edge_ratio_p99", "disp_from_init_max"]
    print("run," + ",".join(keys))
    for r in res:
        print(Path(r["run"]).name + "," + ",".join(
            f"{r.get(k):.4g}" if isinstance(r.get(k), float) else str(r.get(k)) for k in keys))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
