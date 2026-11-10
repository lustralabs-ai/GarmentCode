"""Workstream 9b: sample N random designs on the DEFAULT body (mean_all) with GarmentCode's own sampler
machinery (pattern_sampler.py helpers + pygarment DesignSampler). Mirrors pattern_sampler.generate() minus the
random-body branch (the 5000-body-shape dataset is not installed here). Records pattern-generation time per garment.

Run from GarmentCode/:  python batch_sample.py --name batch40 --size 40 --seed 20261110
Writes datasets_path/<name>_<ts>/{dataset_properties.yaml,default_body/data/<garment>/...} and gen_times.json."""
import sys, os, time, json, random, argparse, traceback
from pathlib import Path
sys.path.insert(0, os.getcwd())
import pattern_sampler as ps
from pygarment.data_config import Properties
from assets.garment_programs.meta_garment import MetaGarment
from assets.bodies.body_params import BodyParameters
import pygarment as pyg
import assets.garment_programs.stats_utils as stats_utils

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='batch40'); ap.add_argument('--size', type=int, default=40)
ap.add_argument('--seed', type=int, default=20261110)
a = ap.parse_args()

sysp = Properties('./system.json')
props = Properties()
props.set_basic(design_file='./assets/design_params/default.yaml', body_default='mean_all',
                body_samples='5000_body_shapes_and_measures', size=a.size, name=a.name, to_subfolders=True)
props.set_section_config('generator')
props.set_section_stats('generator', panel_count={}, garment_types={}, garment_types_summary=dict(main={}, style={}))
ps.props = props  # pattern_sampler uses a module-global `props` in generate(); we do not call generate()
gen_config = props['generator']['config']; gen_config['random_seed'] = a.seed
random.seed(a.seed)
data_folder, default_path, _ = ps._create_data_folder(props, Path(sysp['datasets_path']))
sample_dir = default_path / 'data'
default_body = BodyParameters(Path(sysp['bodies_default_path']) / (props['body_default'] + '.yaml'))
sampler = pyg.DesignSampler(props['design_file'])
times = {}
t_all = time.time()
for i in range(a.size):
    n_try = 0; t_total0 = time.time()
    for _ in range(100):
        n_try += 1
        new_design = sampler.randomize()
        name = f'rand_{ps._id_generator()}'
        t0 = time.time()
        try:
            ps.assert_param_combinations(new_design)
            piece = MetaGarment(name, default_body, new_design)
            piece.assert_total_length()
            def_obj = props['body_default'] + ('_apart' if ps.has_pants(new_design) else '')
            default_body.params['body_sample'] = def_obj
            if piece.is_self_intersecting():
                continue
            pattern = ps._save_sample(piece, default_body, new_design, sample_dir)
            t_ok = time.time() - t0
            stats_utils.count_panels(pattern, props)
            stats_utils.garment_type(name, new_design, props)
            m = new_design['meta']
            times[name] = dict(idx=i, upper=m['upper']['v'], wb=m['wb']['v'], bottom=m['bottom']['v'],
                               pattern_gen_s=t_ok, attempts=n_try, pattern_gen_incl_rejects_s=time.time() - t_total0)
            print(f'[{i}] {name} upper={m["upper"]["v"]} wb={m["wb"]["v"]} bottom={m["bottom"]["v"]} '
                  f'gen {t_ok:.2f}s attempts {n_try}', flush=True)
            break
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            print(f'{name} rejected: {type(e).__name__}: {str(e)[:100]}', flush=True)
            for d in [sample_dir / name]:
                if d.exists():
                    import shutil; shutil.rmtree(d, ignore_errors=True)
props.serialize(data_folder / 'dataset_properties.yaml')
json.dump(dict(seed=a.seed, data_folder=str(data_folder), total_s=time.time() - t_all, garments=times),
          open(data_folder / 'gen_times.json', 'w'), indent=1)
print('DATA_FOLDER', data_folder.name, 'n=', len(times), 'total', time.time() - t_all)
