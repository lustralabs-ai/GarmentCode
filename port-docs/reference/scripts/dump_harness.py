#!/usr/bin/env python
"""Deterministic dump harness for GarmentCode cloth simulation.

Two sub-commands:

  prepare : pattern spec (+ body + sim config) -> box mesh -> self-contained
            `inputs` directory (everything the simulation reads).
  run     : replay an `inputs` directory on a device (cpu / cuda:0), dumping
            per-frame vertex positions and statistics.

Only the pygarment API and public `warp` calls are used, so the same script
runs against the original NVIDIA fork and the ported Warp.  Nothing about
data locations is hard-coded: pass --garmentcode-dir.

Examples
  python dump_harness.py prepare --garmentcode-dir ~/GarmentCode \
      --spec ~/GarmentCode/assets/Patterns/shirt_mean_specification.json \
      --out data/inputs/shirt
  python dump_harness.py run --garmentcode-dir ~/GarmentCode \
      --inputs data/inputs/shirt --device cpu --out data/shirt/cpu_run1 \
      --max-frames 40
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- helpers
def _setup_cwd_and_path(gc_dir: Path, bodies_dir: Path, work_dir: Path):
    """pygarment reads ./system.json relative to cwd; give it a private one
    (no files are written into the GarmentCode directory)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    system = {
        "output": str(work_dir / "out"),
        "datasets_path": "", "datasets_sim": "",
        "sim_configs_path": str(gc_dir / "assets" / "Sim_props"),
        "bodies_default_path": str(bodies_dir),
        "body_samples_path": "",
    }
    (work_dir / "system.json").write_text(json.dumps(system, indent=2))
    if not (work_dir / "assets").exists():   # some code reads ./assets/... relative to cwd
        (work_dir / "assets").symlink_to(gc_dir / "assets")
    os.chdir(work_dir)
    sys.path.insert(0, str(gc_dir))


def _versions():
    import warp as wp
    v = {"python": sys.version.split()[0], "numpy": np.__version__,
         "warp": wp.config.version, "warp_file": str(Path(wp.__file__).parent),
         "platform": platform.platform(), "hostname": platform.node()}
    try:
        v["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        v["nvidia_smi"] = ""
    return v


# ----------------------------------------------------------------- prepare
def cmd_prepare(a):
    gc = Path(a.garmentcode_dir).resolve()
    out = Path(a.out).resolve()
    spec = Path(a.spec).resolve()
    bodies = gc / "assets" / "bodies"
    work = Path(tempfile.mkdtemp(prefix="gch_prepare_"))
    _setup_cwd_and_path(gc, bodies, work)

    import pygarment.data_config as data_config
    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.sim_config import PathCofig

    sim_cfg = Path(a.sim_config).resolve() if a.sim_config else gc / "assets/Sim_props/default_sim_props.yaml"
    props = data_config.Properties(str(sim_cfg))
    name = spec.stem.rpartition("_")[0]
    paths = PathCofig(in_element_path=spec.parent, out_path=str(work / "out"),
                      in_name=name, body_name=a.body, smpl_body=a.smpl, add_timestamp=False)
    res = props["sim"]["config"]["resolution_scale"]
    bm = BoxMesh(paths.in_g_spec, res)
    bm.load()
    bm.serialize(paths, store_panels=False, uv_config=props["render"]["config"]["uv_texture"])

    out.mkdir(parents=True, exist_ok=True)
    files = {
        "boxmesh.obj": paths.g_box_mesh,
        "sim_segmentation.txt": paths.g_mesh_segmentation,
        "orig_lens.pickle": paths.g_orig_edge_len,
        "vertex_labels.yaml": paths.g_vert_labels,
        "specification.json": paths.g_specs,
        "body_measurements.yaml": paths.in_body_mes,
        "body.obj": paths.in_body_obj,
        "body_segmentation.json": paths.body_seg,
        "sim_props.yaml": sim_cfg,
    }
    for dst, src in files.items():
        if Path(src).exists():
            shutil.copy(src, out / dst)
        else:
            print(f"WARNING: missing {src}")
    meta = {"garment": name, "body_name": a.body, "smpl_body": a.smpl,
            "resolution_scale": res, "source_spec": str(spec)}
    (out / "inputs_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"inputs for '{name}' written to {out}")
    shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------- run
def cmd_run(a):
    gc = Path(a.garmentcode_dir).resolve()
    inp = Path(a.inputs).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((inp / "inputs_meta.json").read_text())
    name = meta["garment"]

    # Stage inputs in a private area laid out the way PathCofig expects.
    work = Path(tempfile.mkdtemp(prefix="gch_run_"))
    bodies = work / "bodies"
    bodies.mkdir()
    body_stem = meta["body_name"]
    shutil.copy(inp / "body.obj", bodies / f"{body_stem}.obj")
    shutil.copy(inp / "body_measurements.yaml", bodies / f"{body_stem}.yaml")
    seg_name = "smpl_vert_segmentation.json" if meta["smpl_body"] else "ggg_body_segmentation.json"
    shutil.copy(inp / "body_segmentation.json", bodies / seg_name)
    _setup_cwd_and_path(gc, bodies, work)

    import warp as wp
    if os.environ.get("WARP_CACHE_PATH"):   # old forks ignore the env var; set it explicitly
        wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    wp.init()
    wp.set_device(a.device)   # must happen before Cloth() is built
    dev = wp.get_device()
    if a.device == "cpu" and dev.is_cuda:
        raise SystemExit("requested cpu but warp device is CUDA")
    if a.device != "cpu" and not dev.is_cuda:
        raise SystemExit(f"requested {a.device} but warp device is {dev}")

    import pygarment.data_config as data_config
    from pygarment.meshgen.garment import Cloth
    from pygarment.meshgen.sim_config import PathCofig, SimConfig

    props = data_config.Properties(str(inp / "sim_props.yaml"))
    paths = PathCofig(in_element_path=inp, out_path=str(work / "out"), in_name=name,
                      body_name=body_stem, smpl_body=meta["smpl_body"], add_timestamp=False)
    # PathCofig input spec path: <inp>/<name>_specification.json -> we only need
    # the copies in out_el, which Cloth reads (g_specs, labels, boxmesh, ...)
    for src, dst in {
        "boxmesh.obj": paths.g_box_mesh, "sim_segmentation.txt": paths.g_mesh_segmentation,
        "orig_lens.pickle": paths.g_orig_edge_len, "vertex_labels.yaml": paths.g_vert_labels,
        "specification.json": paths.g_specs, "body_measurements.yaml": paths.body_mes,
    }.items():
        if (inp / src).exists():
            shutil.copy(inp / src, dst)

    cfg = SimConfig(props["sim"]["config"])
    max_steps = cfg.max_sim_steps
    if a.max_frames is not None:
        max_steps = min(max_steps, a.max_frames)

    garment = Cloth(name, cfg, paths, caching=False)
    p_dev = garment.state_0.particle_q.device
    dev_info = {
        "requested": a.device, "wp_get_device": str(dev), "is_cuda": bool(dev.is_cuda),
        "is_cpu": bool(dev.is_cpu), "particle_q_device": str(p_dev),
        "model_device": str(garment.model.device), "sim_use_graph": bool(garment.sim_use_graph),
    }
    print("device info:", dev_info)
    if a.device == "cpu":
        assert dev_info["is_cpu"] and str(p_dev) == "cpu" and not garment.sim_use_graph, dev_info

    # Save exact initial state
    np.save(out / "init_cloth_verts.npy", np.asarray(garment.v_cloth_init, dtype=np.float32))
    np.save(out / "cloth_faces.npy", np.asarray(garment.f_cloth, dtype=np.int32))
    np.save(out / "body_verts.npy", np.asarray(garment.v_body, dtype=np.float32))
    np.save(out / "body_faces.npy", np.asarray(garment.f_body, dtype=np.int32))
    n_part = len(garment.current_verts)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    np.save(frames_dir / "f000000_init.npy", np.array(garment.current_verts, dtype=np.float32))

    # ---- simulation loop (mirrors pygarment.meshgen.simulation.sim_frame_sequence,
    #      minus the wall-clock / frame timeouts which would break determinism) ----
    # NOTE: on CPU wp.array.numpy() returns a VIEW of particle_q, so
    # garment.last_verts and garment.current_verts alias the same memory and
    # Cloth.is_static() sees zero change.  `--static-mode copy` (default) makes
    # copies so the stopping criterion behaves as on CUDA; `native` keeps the
    # original behaviour.
    stop_reason = "max_frames"
    frame_times, saved = [], []
    sim_start = time.time()
    last_frame = -1
    for frame in range(0, max_steps):
        garment.frame = frame
        t0 = time.perf_counter()
        garment.run_frame()
        dt = time.perf_counter() - t0
        frame_times.append(dt)
        if a.static_mode == "copy":
            garment.current_verts = np.array(garment.current_verts, copy=True)
            if garment.last_verts is not None:
                pass
        last_frame = frame
        cur = np.array(garment.current_verts, dtype=np.float32, copy=True)
        if frame % a.save_every == 0:
            np.save(frames_dir / f"f{frame:06d}.npy", cur)
            saved.append(frame)
        if a.static_mode == "copy":
            # last_verts was assigned the (copied) previous current_verts in update()
            pass
        static = False
        if frame >= cfg.zero_gravity_steps and frame >= cfg.min_sim_steps:
            static, _ = garment.is_static()
        if static:
            stop_reason = "static_equilibrium"
            break
    else:
        if max_steps >= cfg.max_sim_steps:
            stop_reason = "max_sim_steps"
        else:
            stop_reason = "frame_cap"   # harness --max-frames reached before max_sim_steps
    total = time.time() - sim_start

    final = np.array(garment.current_verts, dtype=np.float32, copy=True)
    np.save(out / "final_verts.npy", final)
    if last_frame not in saved:
        np.save(frames_dir / f"f{last_frame:06d}.npy", final)
        saved.append(last_frame)
    np.save(out / "frame_times.npy", np.array(frame_times, dtype=np.float64))

    n_self = garment.count_self_intersections()
    n_body = garment.count_body_intersections()
    summary = {
        "garment": name, "device": dev_info, "versions": _versions(),
        "length_unit": "cm (cloth coordinates; body scaled x100 from m; garment c_scale=1)",
        "n_particles": int(n_part), "n_cloth_faces": int(len(garment.f_cloth)),
        "frames_run": last_frame + 1, "last_frame_index": last_frame,
        "stop_reason": stop_reason, "max_sim_steps_cfg": int(cfg.max_sim_steps),
        "min_sim_steps_cfg": int(cfg.min_sim_steps), "frame_cap": a.max_frames,
        "static_mode": a.static_mode, "save_every": a.save_every, "saved_frames": saved,
        "self_intersections": int(n_self), "body_intersections": int(n_body),
        "total_time_s": total, "mean_frame_time_s": float(np.mean(frame_times)) if frame_times else 0,
        "median_frame_time_s": float(np.median(frame_times)) if frame_times else 0,
        "final_has_nan": bool(np.isnan(final).any()),
        "argv": sys.argv,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "frames_run", "stop_reason", "self_intersections", "body_intersections",
        "mean_frame_time_s", "final_has_nan")}, indent=2))
    shutil.rmtree(work, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    pp = sp.add_parser("prepare")
    pp.add_argument("--garmentcode-dir", required=True)
    pp.add_argument("--spec", required=True, help="*_specification.json")
    pp.add_argument("--out", required=True)
    pp.add_argument("--body", default="mean_all")
    pp.add_argument("--smpl", action="store_true")
    pp.add_argument("--sim-config", default=None)
    pp.set_defaults(fn=cmd_prepare)
    rp = sp.add_parser("run")
    rp.add_argument("--garmentcode-dir", required=True)
    rp.add_argument("--inputs", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--device", default="cpu", help="cpu | cuda:0")
    rp.add_argument("--max-frames", type=int, default=None)
    rp.add_argument("--save-every", type=int, default=1)
    rp.add_argument("--static-mode", choices=["copy", "native"], default="copy")
    rp.set_defaults(fn=cmd_run)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
