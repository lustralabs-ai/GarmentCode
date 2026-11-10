"""GarmentCode's real run_sim path with DEFAULT Warp device selection (ws8 question 5).

Mirror of GarmentCode/test_garment_sim.py __main__ (BoxMesh -> serialize -> run_sim -> stats),
with only these differences:
  * no wp.set_device: Warp picks its default device (cuda:0 when a GPU is present),
  * runs in a private cwd that symlinks GarmentCode's assets/ and system.json (GarmentCode tree
    is not written to), and the output goes to OUT_DIR instead of system.json's ./Logs/.

  python gpu_gc_run_default.py GARMENTCODE_DIR PATTERN_SPEC OUT_DIR [SIM_PROPS_YAML]
"""
import os
import sys
import tempfile
import time
from pathlib import Path

gc = Path(sys.argv[1]).resolve()
spec_path = Path(sys.argv[2]).resolve()
out_dir = Path(sys.argv[3]).resolve()
sim_cfg = sys.argv[4] if len(sys.argv) > 4 else "./assets/Sim_props/default_sim_props.yaml"
work = Path(tempfile.mkdtemp(prefix="gch_ws8_gc_"))
(work / "assets").symlink_to(gc / "assets")
(work / "system.json").symlink_to(gc / "system.json")
os.chdir(work)
sys.path.insert(0, str(gc))

import warp as wp  # noqa: E402

if os.environ.get("WARP_CACHE_PATH"):
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
wp.init()
print("warp", wp.__version__, wp.__file__, "default device", wp.get_device(), flush=True)

from pygarment.meshgen.boxmeshgen import BoxMesh  # noqa: E402
from pygarment.meshgen.simulation import run_sim  # noqa: E402
import pygarment.data_config as data_config  # noqa: E402
from pygarment.meshgen.sim_config import PathCofig  # noqa: E402

props = data_config.Properties(sim_cfg)
props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={})
props.set_section_stats('render', render_time={})
garment_name, _, _ = spec_path.stem.rpartition('_')
paths = PathCofig(in_element_path=spec_path.parent, out_path=str(out_dir), in_name=garment_name,
                  body_name='mean_all', smpl_body=False, add_timestamp=True)
t0 = time.time()
garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
garment_box_mesh.load()
garment_box_mesh.serialize(paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])
props.serialize(paths.element_sim_props)
run_sim(garment_box_mesh.name, props, paths, save_v_norms=False, store_usd=False, optimize_storage=False,
        verbose=False)
props.serialize(paths.element_sim_props)
print("WALL_S", round(time.time() - t0, 1))
print("STATS", dict(props['sim']['stats']), dict(props['render']['stats']))
print("OUT_DIR", paths.out_el)
