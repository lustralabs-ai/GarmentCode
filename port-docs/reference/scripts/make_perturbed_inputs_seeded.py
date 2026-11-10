#!/usr/bin/env python3
"""Like make_perturbed_inputs.py, but with a selectable RNG seed, so several
independent ulp-scale perturbations of the same garment can be made.
Adds uniform noise of +-eps (cm) to every box-mesh vertex.
usage: make_perturbed_inputs_seeded.py SRC DST EPS SEED"""
import shutil, sys, numpy as np
src, dst, eps, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
shutil.copytree(src, dst)
rng = np.random.default_rng(seed)
lines = open(f"{dst}/boxmesh.obj").read().split("\n")
out = []
for l in lines:
    if l.startswith("v "):
        p = np.array(l.split()[1:4], dtype=np.float64) + rng.uniform(-eps, eps, 3)
        l = "v %.9g %.9g %.9g" % tuple(p)
    out.append(l)
open(f"{dst}/boxmesh.obj", "w").write("\n".join(out))
