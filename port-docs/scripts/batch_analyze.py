"""Workstream 9b analysis: merge timing JSONs (batch_sim_timed.py), sampler gen_times.json and GarmentCode's
dataset_properties yaml into a per-garment table (csv + markdown) and build labelled contact sheets.
Usage: batch_analyze.py <sim_out_dir (datasets_sim/<D>/default_body)> <timing_dir> <gen_times.json> <out_prefix> [--sheets <dir>]"""
import sys, json, yaml, csv, glob
from pathlib import Path
import numpy as np

def settle(f, win=10, tol=1.5):
    """frames until per-frame time settles: first index i such that every rolling-median window starting at >= i is <= tol * steady,
    steady = median of the last 40% of frames."""
    f = np.asarray(f); n = len(f)
    if n < 3 * win: return n, float(np.median(f))
    steady = float(np.median(f[int(n * 0.6):]))
    rm = np.array([np.median(f[i:i + win]) for i in range(n - win + 1)])
    bad = np.where(rm > tol * steady)[0]
    return (int(bad[-1]) + 1 if len(bad) else 0), steady

def load(sim_dir, timing_dir, gen_json):
    sim_dir = Path(sim_dir)
    props = yaml.safe_load(open(sim_dir / 'dataset_properties_default_body.yaml'))
    st = props['sim']['stats']; fails = {k: set(v) for k, v in st['fails'].items()}
    gen = json.load(open(gen_json))['garments']
    rows = []
    files = sorted(Path(timing_dir).glob('rand_*.json'))
    recs = {}
    for tf in files:
        t = json.load(open(tf)); recs.setdefault(t['name'], {})[t.get('attempt', 0)] = t
    for name, att in sorted(recs.items(), key=lambda kv: gen.get(kv[0], {}).get('idx', 0)):
        t = att[0]; g = gen.get(name, {})
        fr = np.array(t['frames_ms']); n = len(fr)
        ff = [k for k, v in fails.items() if name in v]
        first = sorted(set(t.get('fail_events', [])))
        r = dict(name=name, upper=g.get('upper'), wb=g.get('wb'), bottom=g.get('bottom'),
                 verts=t.get('verts'), faces=t.get('faces'),
                 pattern_s=g.get('pattern_gen_s'), meshgen_s=t.get('meshgen_load_s'),
                 box_ser_s=t.get('boxmesh_serialize_s'), cloth_init_s=t.get('cloth_init_s'),
                 frames=n, flags_first='+'.join(first) if first else 'pass',
                 flags_final='+'.join(ff) if ff else 'pass', attempts=len(att),
                 self_int=t.get('self_int'), body_int=t.get('body_int'),
                 self_final=st['self_collisions'].get(name), body_final=st['body_collisions'].get(name),
                 render_s=t.get('render_s'), total_s=t.get('template_total_s'), run_sim_s=t.get('run_sim_s'),
                 resim_total_s=sum(a.get('template_total_s', 0) for k, a in att.items() if k > 0))
        r['flags'] = r['flags_first']
        if n:
            s_, steady = settle(fr)
            r.update(frames_sum_s=fr.sum() / 1e3, settle_frame=s_,
                     early_med_ms=float(np.median(fr[:s_])) if s_ else float('nan'), early_sum_s=fr[:s_].sum() / 1e3,
                     steady_med_ms=steady, steady_frames=n - s_, steady_sum_s=fr[s_:].sum() / 1e3, frame0_ms=fr[0],
                     max_ms=fr.max())
        r['other_s'] = (r['total_s'] or 0) - (r['meshgen_s'] or 0) - (r['box_ser_s'] or 0) - (r['cloth_init_s'] or 0) \
            - (r.get('frames_sum_s') or 0) - (r['render_s'] or 0)
        r['loop_other_s'] = (r['run_sim_s'] or 0) - (r['cloth_init_s'] or 0) - (r.get('frames_sum_s') or 0) - (r['render_s'] or 0)
        rows.append(r)
    return rows

if __name__ == '__main__':
    sim_dir, timing_dir, gen_json, prefix = sys.argv[1:5]
    rows = load(sim_dir, timing_dir, gen_json)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(prefix + '.csv', 'w') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    print(len(rows), 'rows ->', prefix + '.csv')
