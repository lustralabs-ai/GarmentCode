"""Step 4: pygarment render_images on a box mesh (stand-in for simulated garment).
Usage (from GarmentCode/): render_test.py BOXMESH_DIR NAME [OUTDIR]
Platform chosen via PYOPENGL_PLATFORM (pythonrender forces 'egl' on Linux; for osmesa the
env override is applied after import, see below)."""
import os, sys, shutil
from pathlib import Path
plat = os.environ.get('PYOPENGL_PLATFORM')   # remember request before pythonrender overwrites it
import numpy as np, trimesh
import pygarment.meshgen.render.pythonrender as pr
if plat:
    os.environ['PYOPENGL_PLATFORM'] = plat
from pygarment.meshgen.sim_config import PathCofig
import pygarment.data_config as data_config
src, name = Path(sys.argv[1]), sys.argv[2]
out = Path(sys.argv[3] if len(sys.argv) > 3 else '/tmp/claude-1000/ws4-render')
props = data_config.Properties('./assets/Sim_props/default_sim_props.yaml')
paths = PathCofig(in_element_path=src, out_path=out, in_name=name, body_name='mean_all', smpl_body=False)
for f in src.iterdir():
    shutil.copy(f, paths.out_el / f.name)
shutil.copy(src / f'{name}_boxmesh.obj', paths.g_sim)
body = trimesh.load('./assets/bodies/mean_all.obj', process=False)
print('PYOPENGL_PLATFORM =', os.environ.get('PYOPENGL_PLATFORM'))
pr.render_images(paths, np.asarray(body.vertices), np.asarray(body.faces), props['render']['config'])
for s in props['render']['config']['sides']:
    p = paths.render_path(s); im = np.asarray(__import__('PIL.Image', fromlist=['x']).open(p))
    print(p, im.shape, 'nonwhite/opaque px:', int(((im[..., :3] < 250).any(-1) & (im[..., 3] > 0)).sum()))
