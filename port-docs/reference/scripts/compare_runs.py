#!/usr/bin/env python3
"""Compare two dump directories produced by dump_harness.py (numpy only).

Reports, over the frames present in both dumps:
  per-frame max / mean / RMS Euclidean vertex distance (sim length unit = cm),
  first frame where max distance exceeds --diverge-threshold,
  final-frame stats, intersection-count and frame-count differences.
Exit code 0 = within tolerances, 1 = a tolerance exceeded, 2 = usage/data error.

Tolerances (all optional; unset = report only):
  --tol-max            max vertex distance over all common frames   (cm)
  --tol-rms            RMS vertex distance over all common frames   (cm)
  --tol-final-max      max vertex distance in the final common frame (cm)
  --tol-final-rms      RMS in the final common frame                 (cm)
  --tol-frames         |frames_run_A - frames_run_B|
  --tol-self-int       |self_intersections_A - _B|
  --tol-body-int       |body_intersections_A - _B|
  --tol-self-int-rel / --tol-body-int-rel   relative (fraction of max(A,B,1))
  --require-same-stop  stop_reason must match

Example (bit-exactness check):  compare_runs.py A B --tol-max 0
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def load_frames(d: Path):
    out = {}
    for f in (d / "frames").glob("f*.npy"):
        m = re.match(r"f(\d+)\.npy$", f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--diverge-threshold", type=float, default=1e-3, help="cm; default 1e-3")
    for n in ["max", "rms", "final-max", "final-rms", "frames", "self-int", "body-int",
              "self-int-rel", "body-int-rel"]:
        p.add_argument(f"--tol-{n}", type=float, default=None)
    p.add_argument("--require-same-stop", action="store_true")
    p.add_argument("--csv", default=None, help="write per-frame stats here")
    p.add_argument("-q", "--quiet", action="store_true", help="only print summary")
    a = p.parse_args()

    A, B = Path(a.a), Path(a.b)
    try:
        sa = json.loads((A / "summary.json").read_text())
        sb = json.loads((B / "summary.json").read_text())
        fa, fb = load_frames(A), load_frames(B)
    except Exception as e:
        print(f"ERROR reading dumps: {e}", file=sys.stderr)
        return 2
    common = sorted(set(fa) & set(fb))
    if not common:
        print("ERROR: no common frames", file=sys.stderr)
        return 2

    rows = []
    first_div = None
    for fr in common:
        x, y = np.load(fa[fr]).astype(np.float64), np.load(fb[fr]).astype(np.float64)
        if x.shape != y.shape:
            print(f"ERROR: frame {fr} shape mismatch {x.shape} vs {y.shape}", file=sys.stderr)
            return 2
        d = np.linalg.norm(x - y, axis=1)
        d = np.where(np.isnan(d), np.inf, d)
        mx, mean, rms = float(d.max()), float(d.mean()), float(np.sqrt((d ** 2).mean()))
        rows.append((fr, mx, mean, rms))
        if first_div is None and mx > a.diverge_threshold:
            first_div = fr
    rows = np.array(rows)

    if not a.quiet:
        print(f"{'frame':>7} {'max':>12} {'mean':>12} {'rms':>12}   (cm)")
        step = max(1, len(rows) // 25)
        for i, r in enumerate(rows):
            if i % step == 0 or i == len(rows) - 1:
                print(f"{int(r[0]):7d} {r[1]:12.4e} {r[2]:12.4e} {r[3]:12.4e}")
    if a.csv:
        np.savetxt(a.csv, rows, delimiter=",", header="frame,max,mean,rms", comments="")

    last = rows[-1]
    fin_a, fin_b = np.load(A / "final_verts.npy").astype(np.float64), np.load(B / "final_verts.npy").astype(np.float64)
    fd = np.linalg.norm(fin_a - fin_b, axis=1) if fin_a.shape == fin_b.shape else np.array([np.inf])
    fin_max, fin_mean, fin_rms = float(fd.max()), float(fd.mean()), float(np.sqrt((fd ** 2).mean()))

    na, nb = sa["frames_run"], sb["frames_run"]
    sia, sib = sa["self_intersections"], sb["self_intersections"]
    boa, bob = sa["body_intersections"], sb["body_intersections"]
    print("\n== summary ==")
    print(f"A: {A}  device={sa['device']['wp_get_device']} frames={na} stop={sa['stop_reason']} "
          f"self={sia} body={boa} mean_frame_time={sa['mean_frame_time_s']:.3f}s")
    print(f"B: {B}  device={sb['device']['wp_get_device']} frames={nb} stop={sb['stop_reason']} "
          f"self={sib} body={bob} mean_frame_time={sb['mean_frame_time_s']:.3f}s")
    print(f"common frames compared: {len(common)} (first {common[0]}, last {common[-1]}); unit: cm")
    print(f"over all common frames: max-of-max={rows[:,1].max():.4e}  max-of-rms={rows[:,3].max():.4e}  "
          f"mean-of-rms={rows[:,3].mean():.4e}")
    print(f"first frame with max dist > {a.diverge_threshold:g} cm: {first_div}")
    print(f"last common frame {int(last[0])}: max={last[1]:.4e} mean={last[2]:.4e} rms={last[3]:.4e}")
    print(f"final_verts: max={fin_max:.4e} mean={fin_mean:.4e} rms={fin_rms:.4e}"
          + ("" if na == nb else f"   (WARNING: different final frames {na-1} vs {nb-1})"))
    print(f"frames: {na} vs {nb} (diff {abs(na-nb)}); self-int: {sia} vs {sib}; body-int: {boa} vs {bob}")

    fails = []

    def chk(name, val, tol):
        if tol is not None and not (val <= tol):
            fails.append(f"{name}: {val:.6g} > {tol:g}")

    chk("max vertex distance", rows[:, 1].max(), a.tol_max)
    chk("rms vertex distance (worst frame)", rows[:, 3].max(), a.tol_rms)
    chk("final max", fin_max, a.tol_final_max)
    chk("final rms", fin_rms, a.tol_final_rms)
    chk("frame count diff", abs(na - nb), a.tol_frames)
    chk("self-intersection diff", abs(sia - sib), a.tol_self_int)
    chk("body-intersection diff", abs(boa - bob), a.tol_body_int)
    chk("self-intersection rel diff", abs(sia - sib) / max(sia, sib, 1), a.tol_self_int_rel)
    chk("body-intersection rel diff", abs(boa - bob) / max(boa, bob, 1), a.tol_body_int_rel)
    if a.require_same_stop and sa["stop_reason"] != sb["stop_reason"]:
        fails.append(f"stop reason {sa['stop_reason']} != {sb['stop_reason']}")

    if fails:
        print("\nFAIL:")
        for f in fails:
            print("  " + f)
        return 1
    print("\nOK (within given tolerances)" if any(
        v is not None for k, v in vars(a).items() if k.startswith("tol_")) else "\n(no tolerances given; report only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
