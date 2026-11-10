"""Steps 2+3: generate sewing patterns for several designs, then build box meshes
(everything in test_garment_sim.py before run_sim). Run from GarmentCode/ root.
Usage: gen_and_mesh.py OUTDIR"""
import sys, copy, time, traceback, yaml
from pathlib import Path
from assets.garment_programs.meta_garment import MetaGarment
from assets.bodies.body_params import BodyParameters
from pygarment.meshgen.boxmeshgen import BoxMesh
from pygarment.meshgen.sim_config import PathCofig
import pygarment.data_config as data_config

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
body = BodyParameters('./assets/bodies/mean_all.yaml')
base = yaml.safe_load(open('./assets/design_params/default.yaml'))['design']
tshirt = yaml.safe_load(open('./assets/design_params/t-shirt.yaml'))['design']

def combo(upper, wb, bottom):
    d = copy.deepcopy(base)
    d['meta']['upper']['v'] = upper; d['meta']['wb']['v'] = wb; d['meta']['bottom']['v'] = bottom
    return d

designs = {'t-shirt': tshirt,
  'fitted_shirt': combo('FittedShirt', None, None),
  'shirt_pants': combo('Shirt', None, 'Pants'),
  'skirt_circle': combo(None, 'StraightWB', 'SkirtCircle'),
  'pencil_skirt': combo('FittedShirt', 'FittedWB', 'PencilSkirt'),
  'skirt_levels': combo(None, None, 'SkirtLevels'),
  'godet': combo(None, 'StraightWB', 'GodetSkirt'),
  'skirt2': combo(None, None, 'Skirt2'),
  'many_panels': combo(None, None, 'SkirtManyPanels')}
props = data_config.Properties('./assets/Sim_props/default_sim_props.yaml')
res = props['sim']['config']['resolution_scale']
for name, d in designs.items():
    t = time.time()
    try:
        g = MetaGarment(name, body, d)
        pattern = g.assembly()
        folder = pattern.serialize(out, tag='', to_subfolder=True, with_3d=False, with_text=False, view_ids=False, with_printable=False)
        body.save(folder)
        spec = Path(folder) / f'{pattern.name}_specification.json'
        t1 = time.time()-t
        paths = PathCofig(in_element_path=Path(folder), out_path=out/'mesh', in_name=pattern.name,
                          body_name='mean_all', smpl_body=False, add_timestamp=False)
        bm = BoxMesh(paths.in_g_spec, res); bm.load()
        bm.serialize(paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])
        print(f'OK   {name}: pattern {t1:.1f}s, mesh {time.time()-t-t1:.1f}s, V={len(bm.vertices)} F={len(bm.faces)} -> {paths.out_el}')
    except BaseException as e:
        print(f'FAIL {name}: {type(e).__name__}: {e}'); traceback.print_exc()
