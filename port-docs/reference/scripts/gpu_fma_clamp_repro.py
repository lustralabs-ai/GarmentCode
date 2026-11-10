"""Velocity clamp threshold flip: integrate_particles damps when |v|<=25 and clamps to 25 when |v|>25.
Velocities that were clamped in the previous substep have |v| = 25 +- 1 ulp; the decision
`vel > 25` then depends on last-bit rounding of wp.length (FMA contraction / sqrt) and the
two branches differ by ~1% (exp(-0.25*25*dt)).  Compare CPU vs GPU decisions on identical inputs."""
import numpy as np
import warp as wp
wp.config.quiet = True
import sys
if "--no-fma" in sys.argv:
    wp.set_module_options({"fuse_fp": False})
    print("fuse_fp=False (no FMA contraction on GPU)")

@wp.kernel
def clamp_prev(d: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    v[t] = wp.normalize(d[t]) * 25.0  # what the previous substep's clamp produced

@wp.kernel
def integ(v: wp.array(dtype=wp.vec3), dt: float, vel_out: wp.array(dtype=float), v_new: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    v1 = v[t]
    vel = wp.length(v1)
    if vel > 0.0:
        v1 *= wp.exp(-0.25 * vel * dt)
    if vel > 25.0:
        v1 = wp.normalize(v1) * 25.0
    vel_out[t] = vel
    v_new[t] = v1

rng = np.random.default_rng(0)
N = 200000
d = rng.normal(size=(N, 3)).astype(np.float32)
dt = 1.0 / 600.0
res = {}
vprev = {}
for dev in ("cpu", "cuda:0"):
    dd = wp.array(d, dtype=wp.vec3, device=dev)
    v = wp.zeros_like(dd)
    wp.launch(clamp_prev, N, inputs=[dd, v], device=dev)
    vprev[dev] = v.numpy()
# feed identical (CPU-produced) clamped velocities to both devices
for dev in ("cpu", "cuda:0"):
    v = wp.array(vprev["cpu"], dtype=wp.vec3, device=dev)
    vel = wp.zeros(N, dtype=float, device=dev)
    vn = wp.zeros_like(v)
    wp.launch(integ, N, inputs=[v, dt], outputs=[vel, vn], device=dev)
    res[dev] = (vel.numpy(), vn.numpy())
print("normalize*25 differs cpu vs gpu in", int((vprev["cpu"] != vprev["cuda:0"]).any(1).sum()), "of", N)
vc, vg = res["cpu"][0], res["cuda:0"][0]
print("length differs in", int((vc != vg).sum()), "max |diff| ulp-ish", float(np.abs(vc - vg).max()))
cc, cg = vc > 25, vg > 25
print("clamp decision: cpu clamps", int(cc.sum()), "gpu clamps", int(cg.sum()), "disagree", int((cc != cg).sum()))
dv = np.linalg.norm(res["cpu"][1] - res["cuda:0"][1], axis=1)
print("max |v_new diff|", float(dv.max()), "(branch gap 25*(1-exp(-0.25*25*dt)) =", 25 * (1 - np.exp(-0.25 * 25 * dt)), ")")
