"""Quick visual check of dump_harness final states (ws8 question 3): matplotlib, no GL.

  python gpu_render_runs.py OUT.png RUN_DIR [RUN_DIR ...]

One column per run, rows = front view and side view; body in light grey, cloth coloured
by distance to the first run's final cloth (cm, 0..3) so differences are visible.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


def main():
    out = sys.argv[1]
    runs = [Path(p) for p in sys.argv[2:]]
    ref = np.load(runs[0] / "final_verts.npy")
    fig = plt.figure(figsize=(4 * len(runs), 8))
    cmap = plt.get_cmap("viridis")
    for j, d in enumerate(runs):
        fv = np.load(d / "final_verts.npy")
        cf = np.load(d / "cloth_faces.npy").reshape(-1, 3)
        bv = np.load(d / "body_verts.npy")
        bf = np.load(d / "body_faces.npy").reshape(-1, 3)
        dist = np.linalg.norm(fv - ref, axis=1) if fv.shape == ref.shape else np.zeros(len(fv))
        fc = cmap(np.clip(dist[cf].mean(1) / 3.0, 0, 1))
        for i, (elev, azim) in enumerate(((0, -90), (0, 0))):
            ax = fig.add_subplot(2, len(runs), i * len(runs) + j + 1, projection="3d")
            # GarmentCode meshes are y-up: plot (x, z, y)
            body = Poly3DCollection(bv[bf][:, :, [0, 2, 1]], facecolor=(0.85, 0.85, 0.85, 0.15), edgecolor="none")
            cloth = Poly3DCollection(fv[cf][:, :, [0, 2, 1]], facecolor=fc, edgecolor="none")
            ax.add_collection3d(body)
            ax.add_collection3d(cloth)
            lo, hi = bv.min(0), bv.max(0)
            c = (lo + hi) / 2
            r = (hi - lo).max() / 2
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[2] - r, c[2] + r)
            ax.set_zlim(c[1] - r, c[1] + r)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"{d.parent.name}/{d.name}" if d.name.startswith("cpu_run") else d.name, fontsize=9)
    fig.suptitle("final cloth; colour = distance to first column (0..3 cm)")
    fig.tight_layout()
    fig.savefig(out, dpi=80)


if __name__ == "__main__":
    main()
