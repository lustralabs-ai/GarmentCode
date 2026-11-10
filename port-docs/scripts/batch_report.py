"""Generate the markdown tables of port-docs/batch_validation.md from batch_out data.
usage: batch_report.py <batch_out> <out_dir>   (writes t_garment.md t_agg.md t_stab.md t_cat.md)"""
import csv, glob, json, os, sys, collections
import numpy as np
B, O = sys.argv[1], sys.argv[2]
R = list(csv.DictReader(open(f'{B}/table_main.csv')))
def f(r, c):
    try: return float(r[c])
    except Exception: return None
def fm(x, n=1):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'
short = lambda s: {'cloth_body_intersection': 'body', 'cloth_self_intersection': 'self', 'static_equilibrium': 'static',
                   'degenerate_triangles': 'degen', 'pass': 'pass'}.get
def sf(s):
    return '+'.join({'cloth_body_intersection': 'body', 'cloth_self_intersection': 'self', 'static_equilibrium': 'static',
                     'degenerate_triangles': 'degen'}.get(x, x) for x in s.split('+'))
def sim(r): return f(r, 'frames') not in (None, 0.0)
# per garment table
L = ['| id | design | V | F | pattern s | mesh s | box-ser s | init s | sim frames s | render s | other s | total s (1st attempt) | resim s | frames | settle fr. | early ms/fr | steady ms/fr | 1st outcome | final outcome | self | body |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
L[1] = '|' + '---|' * 21
for r in sorted(R, key=lambda r: r['name']):
    des = '/'.join(x for x in (r['upper'], r['wb'], r['bottom']) if x) or '-'
    s = sim(r)
    L.append('| ' + ' | '.join([r['name'][5:], des, r['verts'] or '-', r['faces'] or '-', fm(f(r, 'pattern_s'), 2), fm(f(r, 'meshgen_s'), 2),
        fm(f(r, 'box_ser_s'), 2), fm(f(r, 'cloth_init_s'), 2), fm(f(r, 'frames_sum_s')) if s else '-', fm(f(r, 'render_s'), 2) if s else '-',
        fm(f(r, 'other_s'), 2) if s else '-', fm(f(r, 'total_s')), fm(f(r, 'resim_total_s')) if f(r, 'resim_total_s') else '0',
        r['frames'] if s else '-', r['settle_frame'] if s else '-', fm(f(r, 'early_med_ms')) if s else '-', fm(f(r, 'steady_med_ms')) if s else '-',
        sf(r['flags_first']), sf(r['flags_final']), r['self_int'] if s else '-', r['body_int'] if s else '-']) + ' |')
open(f'{O}/t_garment.md', 'w').write('\n'.join(L) + '\n')
# categories
C = ['| outcome | first attempt | final (after GarmentCode resim) |', '|---|---|---|']
c1 = collections.Counter(sf(r['flags_first']) for r in R); c2 = collections.Counter(sf(r['flags_final']) for r in R)
for k in sorted(set(c1) | set(c2), key=lambda k: -c2.get(k, 0)):
    C.append(f'| {k} | {c1.get(k,0)} | {c2.get(k,0)} |')
C.append(f'| **flagged total** | {sum(v for k,v in c1.items() if k!="pass")} | {sum(v for k,v in c2.items() if k!="pass")} |')
open(f'{O}/t_cat.md', 'w').write('\n'.join(C) + '\n')
# aggregate (37 garments that reached simulation)
S = [r for r in R if sim(r)]
A = ['| stage (s unless noted) | n | median | p90 | max |', '|---|---|---|---|---|']
def row(label, vals, n=2):
    x = np.array([v for v in vals if v is not None and not np.isnan(v)])
    A.append(f'| {label} | {len(x)} | {np.median(x):.{n}f} | {np.quantile(x,.9):.{n}f} | {x.max():.{n}f} |')
row('pattern generation (all 40)', [f(r, 'pattern_s') for r in R])
for lab, c in [('meshing (BoxMesh.load)', 'meshgen_s'), ('box-mesh serialise', 'box_ser_s'), ('Cloth init (warm)', 'cloth_init_s'),
               ('sim, frames until settled (s)', 'early_sum_s'), ('sim, steady state (s)', 'steady_sum_s'), ('sim, all frames (s)', 'frames_sum_s'),
               ('render', 'render_s'), ('other (save, is_static, storage)', 'other_s'), ('TOTAL per garment, 1st attempt', 'total_s')]:
    row(lab, [f(r, c) for r in S])
row('frames simulated (count)', [f(r, 'frames') for r in S], 0)
row('frames until settled (count)', [f(r, 'settle_frame') for r in S], 0)
row('early ms/frame (median of early frames)', [f(r, 'early_med_ms') for r in S], 1)
row('steady ms/frame', [f(r, 'steady_med_ms') for r in S], 1)
row('frame-0 ms (first frame)', [f(r, 'frame0_ms') for r in S], 0)
row('vertices', [f(r, 'verts') for r in S], 0)
row('faces', [f(r, 'faces') for r in S], 0)
tot = np.array([f(r, 'total_s') for r in S]); rs = np.array([f(r, 'resim_total_s') or 0 for r in S])
A.append(''); A.append(f'Sum of first-attempt totals: {tot.sum():.0f} s; sum of GarmentCode resim totals: {rs.sum():.0f} s '
          f'({int((rs>0).sum())} garments resimulated).')
run = json.load(open(f'{B}/timing/main/_run.json'))
A.append(f'Batch wall (process main): {run["main_wall_s"]:.0f} s plus {run["import_s"]:.1f} s import.')
open(f'{O}/t_agg.md', 'w').write('\n'.join(A) + '\n')
# stability
names = [r['name'][5:] for r in R if r['name'][5:] and r['flags_first'] != 'pass' and r['flags_first'] != 'degenerate_triangles']
T = ['| garment | design | run | self int | body int | frames | attempt events (final attempt) | final verdict |', '|---|---|---|---|---|---|---|---|']
summ = []
for n in names:
    d = next(r for r in R if r['name'] == 'rand_' + n); des = '/'.join(x for x in (d['upper'], d['bottom']) if x)
    fin = []
    for t in ['main', 'rerun1', 'rerun2']:
        fs = sorted(glob.glob(f'{B}/timing/{t}/rand_{n}*.json'))
        if not fs: continue
        j = json.load(open(fs[-1])); ev = j['fail_events']
        v = sf('+'.join(ev)) if ev else 'pass'
        # verdict from final attempt: GarmentCode lists self/body/static flags
        fin.append(v != 'pass')
        T.append(f'| {n} | {des} | {t}{" (resim)" if len(fs)>1 else ""} | {j["self_int"]} | {j["body_int"]} | {len(j.get("frames_ms",[]))} | {v} | {"FLAG" if v!="pass" else "pass"} |')
    summ.append((n, sum(fin), len(fin)))
T.append(''); T.append('Flagged in 3 runs / flagged in N runs (final attempt): ' + ', '.join(f'{n} {a}/{b}' for n, a, b in summ))
open(f'{O}/t_stab.md', 'w').write('\n'.join(T) + '\n')
print('ok')
