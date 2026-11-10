"""Workstream 9b: run GarmentCode's own pattern_data_sim.py (unchanged) with external timing instrumentation.
Monkeypatches (in this process only; no GarmentCode file is edited) record per garment:
meshgen (BoxMesh.load), boxmesh serialize, Cloth init (stage build + graph capture), every Cloth.run_frame (wall time;
each frame ends with a device->host copy of particle_q so it is synchronised), count_*_intersections, save_frame,
render_images, optimize storage, total template_simulation wall.
Usage (cwd = GarmentCode/):  python batch_sim_timed.py <timing_out_dir> -- <pattern_data_sim.py args>
e.g.  python batch_sim_timed.py ../batch_out/timing/main -- --data batch40_xxx --config default_sim_props.yaml --default_body
"""
import sys, os, time, json, runpy
from pathlib import Path
sys.path.insert(0, os.getcwd())
out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
rest = sys.argv[sys.argv.index('--') + 1:]

T0 = time.perf_counter()
import warp as wp
import pygarment.meshgen.datasim_utils as du
import pygarment.meshgen.simulation as simmod
import pygarment.meshgen.boxmeshgen as bmg
from pygarment.meshgen.garment import Cloth
IMPORT_S = time.perf_counter() - T0

REC = {}      # garment name -> dict
CUR = {}      # current record

def tm(fn, key, rec_name=None, accumulate=False):
    def w(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t
            if CUR:
                CUR[key] = CUR.get(key, 0.0) + dt if accumulate else dt
    return w

# template_simulation: top-level per garment
_tmpl = du.template_simulation
def tmpl(paths, props, caching=False):
    global CUR
    n_prev = sum(1 for k in REC if k == paths.in_tag or k.startswith(paths.in_tag + '.resim'))
    key = paths.in_tag if n_prev == 0 else f'{paths.in_tag}.resim{n_prev}'
    CUR = dict(name=paths.in_tag, attempt=n_prev, frames_ms=[], fail_events=[])
    REC[key] = CUR
    t = time.perf_counter()
    try:
        return _tmpl(paths, props, caching=caching)
    finally:
        CUR['template_total_s'] = time.perf_counter() - t
        CUR['t_end'] = time.time()
        json.dump(CUR, open(out_dir / f'{key}.json', 'w'))
du.template_simulation = tmpl

# batch_sim calls template_simulation through module global -> patched above.
_bl = bmg.BoxMesh.load
def bl(self, *a, **k):
    t = time.perf_counter()
    try:
        return _bl(self, *a, **k)
    finally:
        CUR['meshgen_load_s'] = time.perf_counter() - t
        try:
            CUR['verts'] = len(self.vertices); CUR['faces'] = len(self.faces)
        except Exception: pass
bmg.BoxMesh.load = bl
bmg.BoxMesh.serialize = tm(bmg.BoxMesh.serialize, 'boxmesh_serialize_s')

_ci = Cloth.__init__
Cloth.__init__ = tm(_ci, 'cloth_init_s')
_rf = Cloth.run_frame
def rf(self):
    t = time.perf_counter()
    try:
        return _rf(self)
    finally:
        CUR['frames_ms'].append((time.perf_counter() - t) * 1e3)
Cloth.run_frame = rf
def tmv(fn, key, vkey):
    def w(*a, **k):
        t = time.perf_counter(); r = fn(*a, **k)
        if CUR: CUR[key] = CUR.get(key, 0.0) + time.perf_counter() - t; CUR[vkey] = r
        return r
    return w
Cloth.count_self_intersections = tmv(Cloth.count_self_intersections, 'count_self_s', 'self_int')
Cloth.count_body_intersections = tmv(Cloth.count_body_intersections, 'count_body_s', 'body_int')
import pygarment.data_config as dc
_af = dc.Properties.add_fail
def af(self, section, fail_type, info):
    if CUR: CUR['fail_events'].append(fail_type)
    return _af(self, section, fail_type, info)
dc.Properties.add_fail = af
Cloth.is_static = tm(Cloth.is_static, 'is_static_s', accumulate=True)
Cloth.save_frame = tm(Cloth.save_frame, 'save_frame_s')
simmod.render_images = tm(simmod.render_images, 'render_s')
simmod.optimize_garment_storage = tm(simmod.optimize_garment_storage, 'optimize_storage_s')
_rs = simmod.run_sim
def rs(*a, **k):
    t = time.perf_counter()
    try:
        return _rs(*a, **k)
    finally:
        CUR['run_sim_s'] = time.perf_counter() - t
du.run_sim = rs

sys.argv = ['pattern_data_sim.py'] + rest
t = time.perf_counter()
code = 0
try:
    runpy.run_path('pattern_data_sim.py', run_name='__main__')
except SystemExit as e:
    code = e.code
json.dump(dict(import_s=IMPORT_S, main_wall_s=time.perf_counter() - t, exit=code, device=str(wp.get_device()),
               argv=rest), open(out_dir / '_run.json', 'w'))
print('RUN DONE exit', code, 'main wall', time.perf_counter() - t, 'import', IMPORT_S)
sys.exit(code or 0)
