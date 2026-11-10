#!/usr/bin/env python3
"""Copy an inputs dir, adding uniform noise of +-eps (cm) to box-mesh vertices.
Used to estimate how fast the sim amplifies tiny (round-off-sized) differences,
as a proxy for GPU atomic-order nondeterminism.  usage: make_perturbed_inputs.py SRC DST EPS"""
import shutil, sys, re, numpy as np
src, dst, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])
shutil.copytree(src, dst)
rng = np.random.default_rng(0)
lines = open(f"{dst}/boxmesh.obj").read().split("\n")
out = []
for l in lines:
    if l.startswith("v "):
        p = np.array(l.split()[1:4], dtype=np.float64) + rng.uniform(-eps, eps, 3)
        l = "v %.9g %.9g %.9g" % tuple(p)
    out.append(l)
open(f"{dst}/boxmesh.obj", "w").write("\n".join(out))
