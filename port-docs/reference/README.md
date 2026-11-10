# Reference data for the GarmentCode / Warp port

Golden data from the ORIGINAL GarmentCode Warp fork (`garmentcode_hack` branch,
Warp 1.0.0-beta.6, commit `63baf6855`), CPU backend, built and run locally because
`alcyone` (the GTX 1070 box) was unreachable. **CUDA reference runs have NOT been
made yet** - see "Run on alcyone when it is back".

## Contents

```
scripts/dump_harness.py          prepare (pattern -> box mesh -> inputs dir) and run (replay + dump)
scripts/compare_runs.py          numpy-only comparison of two dump dirs, tolerance exit codes
scripts/run_cpu_reference.sh     runs every garment in data/inputs twice (run1/run2) in parallel
scripts/make_perturbed_inputs.py adds tiny noise to box-mesh vertices (sensitivity probe)
data/inputs/<garment>/           exact simulation inputs (boxmesh.obj, sim_segmentation.txt, orig_lens.pickle,
                                 vertex_labels.yaml, specification.json, body.obj + yaml + segmentation, sim_props.yaml)
data/<garment>/<backend>_<run>/  frames/fNNNNNN.npy (float32 [N,3] cloth positions, every 2nd frame + init + last),
                                 final_verts.npy, init_cloth_verts.npy, cloth_faces.npy, body_verts/faces.npy,
                                 frame_times.npy, summary.json (device proof, versions, stop reason, intersections)
data/ref_venv_freeze.txt         pip freeze of the reference venv
```
Length unit of all positions: **cm** (body mesh scaled x100 from metres, cloth is in cm, y up, body shifted so min y >= 0).

## Garments (all shipped in GarmentCode/assets/Patterns, body `mean_all`, `default_sim_props.yaml`)

| garment | character | particles | note |
|---|---|---|---|
| `shirt_mean` | top with sleeves | 7,990 | attachment labels absent -> attachment off, min_sim_steps cfg 0 |
| `dress_pencil` | dress / skirt | 14,425 | attachment constraints on (min 405 frames), many self-intersections |
| `js_mean_all` | jumpsuit (legs + sleeves) | 22,724 | attachment constraints on, body intersections on legs in early frames |

## Reference results (CPU, this machine: Ryzen AI MAX+ 395, 1 thread per sim)

| garment | frames | stop | self-int | body-int | s/frame |
|---|---|---|---|---|---|
| shirt_mean | 247 | static equilibrium | 13 | 0 | 0.44 |
| dress_pencil | 870 | static equilibrium | 434 | 0 | 0.80 |
| js_mean_all | 789 | static equilibrium | 0 | 0 | 1.24 |

(The earlier alcyone CUDA shirt run: 222 frames, 10 self-intersections. Different backend, so
a different but similar trajectory is expected; not directly checkable until CUDA runs exist.)

## Measured reproducibility

* **CPU run1 vs run2, same machine, concurrent processes: bit-identical** for all three garments - max vertex
  distance 0.0 over every saved frame, identical final positions, frame counts, intersection counts
  (`compare_runs.py runA runB --tol-max 0` exits 0). CPU path is deterministic.
* **Sensitivity probe** (`data/shirt_mean/cpu_perturb1e-5`): box-mesh vertices perturbed by +-1e-5 cm
  (about 1 float32 ulp at 100 cm) on the CPU backend. Divergence from the unperturbed run:
  frame 0: max 1.8e-3 cm; frame 6: max 0.23 cm; frame 8: rms 5.9e-3; by frame ~100 rms 0.25 cm;
  final frame: max 1.80 cm, mean 0.29 cm, rms 0.38 cm; self-int 12 vs 13, same frame count (247).
  So the simulation amplifies ulp-level differences by ~100x in the first frame and then saturates at roughly
  0.3 cm rms / 2 cm max (garment is ~140 cm tall). This is the best available proxy for GPU atomic-order
  nondeterminism and for a port that is not bit-exact. **Real GPU-vs-GPU and CPU-vs-CUDA numbers are unmeasured.**

## Recommended tolerances (to be re-validated against CUDA runs from alcyone)

(a) port-CPU vs reference-CPU. Target is bit-exact (`--tol-max 0`) when the same kernels compile to the same
arithmetic. A different Warp/LLVM version may legitimately change FMA contraction / math-function ulps, and
the sim amplifies that (see probe), so use two tiers:
* Early-frame check (decisive for porting bugs): frame 0 `max <= 1e-2 cm` (a 1-ulp input perturbation gives 1.8e-3 cm after frame 0), and frames <= 10 `max <= 0.5 cm`, rms <= 0.02 cm (probe: 0.28 cm max / 7e-3 rms at frame 10). Port bugs typically give errors well above this within a few frames; run with `--max-frames 11` on both sides or compare only those frames.
* Whole run (statistical, like the probe): worst-frame rms <= 0.6 cm, final-frame max <= 4 cm, same stop reason, frame count within 10%, self-int within 30% (or +-5), body-int within +-5.
  Example: `compare_runs.py ref port --tol-rms 0.6 --tol-final-max 4 --tol-frames 25 --tol-self-int-rel 0.3 --require-same-stop`.

(b) GPU vs GPU (e.g. ported HIP vs CUDA, or run vs run): statistical only; start from the same numbers as the
whole-run tier above (rms 0.6 cm / final max 4 cm / frames +-10% / intersections +-30%), since
the probe saturates there; tighten after alcyone gives real CUDA-vs-CUDA spread. Chaotic outcomes
(static-equilibrium reached at a different frame, borderline intersection counts) are expected for
dress_pencil (434 self-intersections) - compare its intersection count relatively.

## Commands (reproducing what was run here)

```
cd $WS
source venv-ref/bin/activate
export WARP_CACHE_PATH=/tmp/claude-1000/warp-cache-ws3 MPLCONFIGDIR=/tmp/claude-1000/mpl
GC=$PWD/GarmentCode; R=$PWD/reference
for g in shirt_mean dress_pencil js_mean_all; do
  python $R/scripts/dump_harness.py prepare --garmentcode-dir $GC \
     --spec $GC/assets/Patterns/${g}_specification.json --out $R/data/inputs/$g; done
# NOTE: warm the kernel cache first (one short run, see Gotchas), then:
$R/scripts/run_cpu_reference.sh $PWD/venv-ref $GC $R/data cpu --save-every 2
# compare
python $R/scripts/compare_runs.py $R/data/shirt_mean/cpu_run1 $R/data/shirt_mean/cpu_run2 --tol-max 0
```
Single run: `dump_harness.py run --garmentcode-dir $GC --inputs data/inputs/shirt_mean --device cpu --out OUT [--max-frames N] [--save-every K] [--static-mode copy|native]`.

The harness does not write into the GarmentCode dir: it builds a private cwd with its own `system.json`
(pygarment reads `./system.json`) and a symlink to `assets/` (box meshing reads `./assets/img/...`).
`WARP_CACHE_PATH` is honoured by setting `wp.config.kernel_cache_dir` (this old fork ignores the env var;
without it kernels go to `~/.cache/warp/1.0.0-beta.6`).

### Replay a saved input on another machine
Copy `data/inputs/<garment>` and `scripts/`, then run `dump_harness.py run --inputs <that dir> --garmentcode-dir <any
GarmentCode checkout>` (only needs `pygarment`, `assets/Sim_props` not needed; the sim config is inside the inputs
dir as `sim_props.yaml`). No pattern generation or meshing is redone; body mesh, box mesh, segmentation, labels are all in the dir.
The harness imports `pygarment` from `--garmentcode-dir` and `warp` from the active environment (use the ported Warp's venv).

## Environment (reference build)
Python 3.10.22, numpy 1.26.4, warp-lang 1.0.0b6 (editable from `$WS/warp-ref-cpu`,
a worktree of `garmentcode_hack` @ 63baf6855, built CPU-only: `python build_lib.py` with no CUDA_HOME;
`tools/packman/{packman,python.sh}` needed chmod +x), venv `venv-ref` (uv; `data/ref_venv_freeze.txt`),
GCC (system), LLVM from packman `clang+llvm-warp 15.0.7`. Machine: Linux 6.17, AMD Ryzen AI MAX+ 395. No NVIDIA driver.
Packages installed for GarmentCode: pyyaml scipy svgwrite svgpathtools psutil matplotlib CairoSVG trimesh libigl pyrender cgal
(no torch needed on the sim path; nicegui not needed).

## Findings / gotchas
1. **CPU path works for every kernel the sim uses** (collide, XPBD, edge-edge, triangle-particle, cloth_reference_drag,
   attachment constraints, intersection counting): all three garments ran to static equilibrium on CPU with no errors, NaNs
   or kernel failures; device proof is in each `summary.json` (`wp_get_device: cpu`, `particle_q_device: cpu`, `sim_use_graph: false`).
2. **Bug in GarmentCode's CPU static check (not fixed in the source):** on CPU `wp.array.numpy()` returns a *view* of
   `particle_q`, and with an even number of substeps (10) `state_0` is the same buffer every frame, so
   `Cloth.last_verts` and `current_verts` alias and `is_static()` sees zero change: the sim would stop the first
   frame past `min_sim_steps`/`zero_gravity_steps`. (Verified aliasing with a small test; the native-mode stop was not run.)
   The harness default `--static-mode copy` copies `current_verts` each frame so the criterion works as on CUDA. `native` keeps the original.
   Port validation must use the same mode on both sides.
3. **Segfaults when several processes JIT-compile into one shared kernel cache at once** (3/6 runs on first launch,
   1/6 on a "warm" relaunch). Not a sim bug: all passed when rerun. Warm the cache with one process or use a
   separate `WARP_CACHE_PATH` per process.
4. Harness loop mirrors `sim_frame_sequence` but omits the wall-clock frame/sim timeouts (they break determinism);
   `max_sim_steps` (2400) still applies.
5. `frames/` stores every 2nd frame (plus init and last); `compare_runs.py` compares the frames both dumps contain.
   Use `--save-every 1` for exact per-frame comparison.

## Run on alcyone when it is back (nothing below has been run)
```
# on alcyone: reuse ~/gch/venv and ~/gch/GarmentCode (do not modify); copy this reference/ dir over (scripts + data/inputs)
rsync -a $WS/reference/{scripts,data/inputs} alcyone:~/gch/ref/   # keep layout: ~/gch/ref/scripts, ~/gch/ref/data/inputs
ssh alcyone
cd ~/gch/ref; source ~/gch/venv/bin/activate
export WARP_CACHE_PATH=$HOME/gch/ref/warpcache MPLCONFIGDIR=$HOME/gch/ref/mpl
GC=$HOME/gch/GarmentCode
# 1) CUDA: full length, two runs each. Run serially on the single GPU, not in parallel:
for g in shirt_mean dress_pencil js_mean_all; do for r in run1 run2; do
  mkdir -p data/$g/cuda_$r
  nohup python scripts/dump_harness.py run --garmentcode-dir $GC --inputs data/inputs/$g --device cuda:0 \
        --out data/$g/cuda_$r --save-every 2 > data/$g/cuda_$r.log 2>&1; done; done; touch data/cuda.done   # wrap in nohup bash -c '...' &
# 2) CPU cross-check on alcyone with the NVIDIA fork there (same inputs; is it bit-identical to the 395's CPU run?):
for g in shirt_mean dress_pencil js_mean_all; do mkdir -p data/$g/cpu_alcyone; python scripts/dump_harness.py run \
  --garmentcode-dir $GC --inputs data/inputs/$g --device cpu --out data/$g/cpu_alcyone --save-every 2; done
# 3) rsync data/<garment>/{cuda_run1,cuda_run2,cpu_alcyone} back, then:
python scripts/compare_runs.py data/shirt_mean/cuda_run1 data/shirt_mean/cuda_run2        # GPU-vs-GPU spread
python scripts/compare_runs.py data/shirt_mean/cuda_run1 data/shirt_mean/cpu_run1         # CPU-vs-CUDA
python scripts/compare_runs.py data/shirt_mean/cpu_alcyone data/shirt_mean/cpu_run1 --tol-max 0   # cross-machine CPU bit-exactness
```
Then replace the provisional GPU tolerances above with ~2x the measured CUDA-vs-CUDA spread.
CUDA mode (`--device cuda:0`) asserts `is_cuda`, uses `sim_use_graph` as GarmentCode does, and was syntax-exercised only through the
CPU path here; the CUDA branch itself is untested. alcyone's Python is the same 3.10 / numpy<2 family; its
pygarment is identical code to the local one (inputs here were produced with the local GarmentCode checkout).
The CPU cross-check on alcyone may differ in the last bits if its compiler/LLVM differs from the packman LLVM used here.
