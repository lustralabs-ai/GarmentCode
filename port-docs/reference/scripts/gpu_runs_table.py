"""Aggregate dump_harness runs and their divergence from a reference run (ws8 question 3).

  python gpu_runs_table.py --ref REF_RUN_DIR RUN_DIR [RUN_DIR ...] [--frames 0,10,38,100]

Per run: frames_run, stop_reason, self/body intersections, mean frame time, and the max / RMS
vertex distance (cm) to REF at selected frames and over all common frames (frames present
in both dumps, saved by dump_harness --save-every).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def frames(d):
    out = {}
    for f in (Path(d) / "frames").glob("f*.npy"):
        m = re.match(r"f(\d+)\.npy$", f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--frames", default="0,10,38,100,200")
    a = ap.parse_args()
    sel = [int(x) for x in a.frames.split(",")]
    rf = frames(a.ref)
    print("run,frames_run,stop,self_int,body_int,mean_frame_s," + ",".join(f"max@{f}" for f in sel)
          + ",max_all,rms_all,first_frame_max>1e-3")
    for d in [a.ref] + a.runs:
        s = json.loads((Path(d) / "summary.json").read_text())
        ff = frames(d)
        common = sorted(set(ff) & set(rf))
        mx, rms_all, first = [], [], None
        per = {}
        for f in common:
            x = np.load(ff[f]).astype(np.float64)
            y = np.load(rf[f]).astype(np.float64)
            dd = np.linalg.norm(x - y, axis=1)
            per[f] = dd.max()
            mx.append(dd.max())
            rms_all.append((dd ** 2).mean())
            if first is None and dd.max() > 1e-3:
                first = f
        cells = [f"{per[f]:.3g}" if f in per else "" for f in sel]
        print(",".join([Path(d).name, str(s.get("frames_run")), str(s.get("stop_reason")),
                        str(s.get("self_intersections")), str(s.get("body_intersections")),
                        f"{s.get('mean_frame_time_s', float('nan')):.4f}"] + cells
                       + [f"{max(mx):.3g}" if mx else "", f"{np.sqrt(np.mean(rms_all)):.3g}" if mx else "",
                          str(first)]))


if __name__ == "__main__":
    main()
