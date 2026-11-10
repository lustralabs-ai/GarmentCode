"""Summarise gpu_profile.py JSONs (graph + kernels) for the batch garments -> markdown.
usage: batch_profile_summary.py <prof_dir> id1 id2 ...   (files <prof_dir>/<id>_graph.json, <id>_kernels.json)"""
import json, os, sys, numpy as np
P = sys.argv[1]
print('| garment | particles | springs | graph: frame0 ms | graph: median ms fr 1-39 | graph: median ms fr 40+ | graph: median last 400 fr | kernels: GPU ms/fr, fr 1-39 | kernels: GPU ms/fr, fr 40-299 | create_self_edge_contacts share fr 1-39 | share fr 40-299 | next kernel (share fr 40-299) | wall/GPU ratio fr 40+ |')
print('|' + '---|' * 13)
for g in sys.argv[2:]:
    gp, kp = f'{P}/{g}_graph.json', f'{P}/{g}_kernels.json'
    row = [g]
    try:
        G = json.load(open(gp)); w = np.array([x['wall_s'] for x in G['frames']]) * 1e3
        row += [str(G['particles']), str(G['springs']), f'{w[0]:.0f}', f'{np.median(w[1:40]):.1f}', f'{np.median(w[40:]):.1f}', f'{np.median(w[-400:]):.1f}']
    except Exception as e:
        row += ['n/a'] * 6
    try:
        K = json.load(open(kp)); fr = K['frames']
        def agg(a, b):
            tot = {}; gpu = []
            for x in fr[a:b]:
                gpu.append(x['gpu_kernel_ms'])
                for k, v in x['kernels_ms'].items(): tot[k] = tot.get(k, 0) + v
            s = sum(tot.values()) or 1
            return np.median(gpu), tot, s
        g1, t1, s1 = agg(1, 40); g2, t2, s2 = agg(40, 300)
        ek = [k for k in t1 if 'create_self_edge_contacts' in k][0]
        nxt = sorted((k for k in t2 if k != ek), key=lambda k: -t2[k])[0]
        wk = np.median([x['wall_s'] * 1e3 for x in fr[40:300]])
        row += [f'{g1:.0f}', f'{g2:.1f}', f'{100*t1[ek]/s1:.0f}%', f'{100*t2.get(ek,0)/s2:.0f}%', f'{nxt.split()[-1][:-9]} {100*t2[nxt]/s2:.0f}%', f'{wk/g2:.1f}x']
    except Exception as e:
        row += ['n/a'] * 6
    print('| ' + ' | '.join(row) + ' |')
