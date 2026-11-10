# Batch validation: 40 random garments, whole pipeline on the AMD Radeon 8060S (ROCm/HIP Warp 1.13)

Workstream 9b. Question: is whole-pipeline draping on this GPU practical, and how close to interactive?

## Short answer

- **Practical as a batch tool, not interactive.** First-attempt end-to-end time per garment (pattern, meshing, sim to static equilibrium or the 2400-frame cap, render, save) is **median 17.4 s, p90 58.7 s, max 67.6 s** over the 37 garments that reached simulation. Fastest are about 4 s.
- **Important caveat on those numbers: they are first attempt only.** GarmentCode itself resimulates once any garment flagged `static_equilibrium` (or other non-intersection category). That added **228 s over 5 garments** (K6AAOMDSH2 +48 s, QLKG5XNE5M +67 s, 3NCN2ULC5M +61 s, 6C7A6KMHCX, MYH6HZAQSE +35 s); the 40-garment batch wall was **1107 s (18.5 min)**, against 877 s for the sum of first attempts. A resimulated garment therefore costs up to about 2x. The tables below give both columns.
- **The simulation frame loop dominates** (median 12.9 s of 17.4 s; at p90 51 of 59 s). Everything else is small: pattern 0.08 s median, meshing 0.8 s, box-mesh serialise 1.6 s, cloth init 0.9 s, render 0.6 s.
- **Cold compile is a one-time ~107 s** (not in any warm number), see below.
- **Pass rate**: 31/40 pass after GarmentCode's own resim (29/40 on first attempt); 9 final flags = 3 meshing failures (degenerate triangles, platform-independent) + 6 simulation flags. 0 crashes. Comparable in kind to the 10/39 flagged on the NVIDIA GTX 1070 (0 crashes), but it is a different random sample and its flag mix is different (3 deterministic meshing failures here), so the rates are not directly comparable.
- Flags agree with what I see in the renders for 2 of 3 clearly broken garments; one broken garment (6C7A6KMHCX) passes GarmentCode's checks.

## Exact setup and commands

Paths are relative to the GarmentCode repo root (`GarmentCode_hack/GarmentCode`; `../batch_out`, `../venv-gc` are outside the repo). Working directory for all commands: the repo root.

Environment (every job):
```
export ROCM_PATH=/home/carl/py/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH=$ROCM_PATH/lib MPLCONFIGDIR=/tmp/claude-1000/mpl PYTHONDONTWRITEBYTECODE=1
export WARP_CACHE_PATH=/tmp/claude-1000/warp-cache-ws9b      # warm cache for the main batch; empty dir for the cold run
```
Interpreter `../venv-gc/bin/python` (numpy 2.5.4, Warp 1.13+rocm, device `cuda:0`).

1. **Sampling, seed 20261110** (the record of the seed is also in `batch_out/datasets/batch40_261110-14-02-22/dataset_properties.yaml`):
   `python ../port-docs/scripts/batch_sample.py --name batch40 --size 40 --seed 20261110`
   This mirrors `pattern_sampler.generate()` (same `pyg.DesignSampler`, `random.seed(seed)`, same helpers) for the default body only, and also records per-garment pattern-generation time. 40 garments in 23.3 s. The 5000-body dataset is not installed so GarmentCode's random-body branch is skipped; designs are sampled on the mean body (`mean_all`; Pants designs use `mean_all_apart`).
2. **Warm batch simulation** (copy of the dataset: `b40main`), wrapper runs GarmentCode's `pattern_data_sim.py` unchanged via `runpy`:
   `timeout 7200 ../venv-gc/bin/python ../port-docs/scripts/batch_sim_timed.py ../batch_out/timing/main -- --data b40main --config default_sim_props.yaml --default_body`
   Instrumentation is by monkeypatching from outside (no edits to sim code): `datasim_utils.template_simulation` and `run_sim`, `BoxMesh.load/serialize`, `Cloth.__init__/run_frame/is_static/save_frame`, the intersection counters, `render_images`, `optimize_garment_storage`, `Properties.add_fail`. Each GarmentCode attempt (first, then `.resim1`) gets its own record. Per-frame times come from timing `Cloth.run_frame`, which ends with a device-to-host copy, so each is synchronised.
3. **Analysis**: `python ../port-docs/scripts/batch_analyze.py <sim_out> ../batch_out/timing/main ../batch_out/datasets/batch40_261110-14-02-22/gen_times.json ../batch_out/table_main` then `batch_report.py ../batch_out <dir>` generated the tables in this file. "Settled" frame = first frame after which every 10-frame rolling median is <= 1.5x the steady-state median (steady = median of the last 40 % of frames).
4. **Contact sheets**: `python ../port-docs/scripts/batch_sheets.py ../batch_out/table_main.csv <sim_dir> port-docs/batch 8 main`
5. **Flag reruns** (2 more runs of each of the 8 garments flagged in the main run, including the 2 rescued by resim): datasets `rerun1`, `rerun2`, same wrapper into `../batch_out/timing/rerun{1,2}`.
6. **Profiling**: `port-docs/reference/scripts/dump_harness.py prepare` on the garment's specification, then `gpu_profile.py --mode graph --frames 1200` and `--mode kernels --frames 300`, both `--no-static-stop`.
7. A first full batch (`runA`, dataset `batch40_261110-14-02-22` itself) lost first-attempt timing for resimulated garments because my driver overwrote one JSON per name; I added per-attempt keys and re-ran the same designs as `b40main`. `runA` is kept for its final outcomes only. Its numbers agree in kind.

### Cold compile (reported separately)
One garment (rand_K6AAOMDSH2) simulated with an empty `WARP_CACHE_PATH`: first `Cloth.__init__` took **106.9 s** (module loads: `warp.sim.collide` 94.3 s, `integrator_xpbd` 6.4 s, `integrator_euler` 3.8 s, `utils` 1.1 s, rest < 0.1 s), whole process about 154 s. In the main batch every module load line in `main.log` says `(cached)` (milliseconds), so **no warm number includes compilation**.

### system.json
`system.json` is git-ignored and was edited (original values below; I left it pointing at `batch_out`, as requested):

| key | original | now |
|---|---|---|
| datasets_path | `""` | `$WS/batch_out/datasets` |
| datasets_sim | `""` | `$WS/batch_out/datasets_sim` |
| output, sim_configs_path, bodies_default_path, body_samples_path | `./Logs/`, `./assets/Sim_props`, `./assets/bodies`, `""` | unchanged |

## Outcomes by category

Counts over all 40 garments. "After resim" is GarmentCode's own behaviour (one automatic resimulation for non-intersection flags).

| outcome | first attempt | final (after GarmentCode resim) |
|---|---|---|
| pass | 29 | 31 |
| degen | 3 | 3 |
| self | 2 | 2 |
| body | 1 | 2 |
| body+self+static | 1 | 1 |
| body+static | 2 | 1 |
| static | 2 | 0 |
| **flagged total** | 11 | 9 |

## Per-stage time budget (37 garments that reached simulation; first attempt)

| stage (s unless noted) | n | median | p90 | max |
|---|---|---|---|---|
| pattern generation (all 40) | 40 | 0.08 | 1.16 | 1.65 |
| meshing (BoxMesh.load) | 37 | 0.78 | 1.36 | 2.34 |
| box-mesh serialise | 37 | 1.62 | 2.45 | 2.87 |
| Cloth init (warm) | 37 | 0.88 | 1.32 | 2.02 |
| sim, frames until settled (s) | 37 | 4.49 | 18.67 | 27.40 |
| sim, steady state (s) | 37 | 5.59 | 36.34 | 52.55 |
| sim, all frames (s) | 37 | 12.86 | 50.98 | 63.32 |
| render | 37 | 0.57 | 0.72 | 1.13 |
| other (save, is_static, storage) | 37 | 0.17 | 0.59 | 0.98 |
| TOTAL per garment, 1st attempt | 37 | 17.39 | 58.69 | 67.60 |
| frames simulated (count) | 37 | 580 | 2400 | 2400 |
| frames until settled (count) | 37 | 47 | 63 | 395 |
| early ms/frame (median of early frames) | 36 | 49.5 | 292.9 | 397.6 |
| steady ms/frame | 37 | 10.4 | 20.7 | 33.9 |
| frame-0 ms (first frame) | 37 | 1190 | 2839 | 3811 |
| vertices | 37 | 11750 | 20281 | 35169 |
| faces | 37 | 23192 | 40106 | 69706 |

Sum of first-attempt totals: 877 s; sum of GarmentCode resim totals: 228 s (5 garments resimulated).
Batch wall (process main): 1107 s plus 1.7 s import.

"early" = frames before the per-frame time settles; "steady" = after. The 3 meshing failures never reach simulation (they cost 0.3 to 3 s of pattern and meshing, then GarmentCode resimulates them and they fail identically).
`other` = save frame, static check and storage optimisation. Frame 0 is slow (0.6 to 3.8 s) and is part of the early frames. Early ms/frame is high because of the self-edge-contact hotspot (profiling section).

## Per-garment table

`total` is the **first attempt only**; `resim` is the extra time GarmentCode's automatic resimulation added (0 = none). `self`/`body` are intersection counts of the first attempt (threshold fail: self > 300, body > 35). `1st outcome` / `final outcome` are the GarmentCode verdicts before and after resim (static = static_equilibrium, degen = degenerate_triangles). Per-stage times in seconds. Sim frames s = sum of per-frame times of the first attempt.

| id | design | V | F | pattern s | mesh s | box-ser s | init s | sim frames s | render s | other s | total s (1st attempt) | resim s | frames | settle fr. | early ms/fr | steady ms/fr | 1st outcome | final outcome | self | body |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 02MWDQWIU6 | Shirt | 11794 | 23314 | 1.65 | 0.82 | 1.40 | 0.88 | 21.4 | 0.53 | 0.13 | 25.2 | 0 | 458 | 53 | 298.9 | 11.8 | pass | pass | 0 | 0 |
| 08HCQXAK18 | StraightWB/SkirtCircle | 7192 | 14142 | 0.02 | 0.30 | 0.90 | 0.61 | 3.1 | 0.43 | 0.09 | 5.4 | 0 | 406 | 16 | 18.9 | 6.1 | pass | pass | 0 | 0 |
| 0VXDPQ5CEJ | Shirt | 7321 | 14246 | 0.03 | 0.44 | 1.08 | 0.64 | 5.4 | 0.47 | 0.08 | 8.1 | 0 | 493 | 52 | 18.0 | 7.2 | pass | pass | 19 | 0 |
| 1N6OORV4FV | Shirt/GodetSkirt | 22067 | 43810 | 1.04 | 1.52 | 2.71 | 1.62 | 50.1 | 0.75 | 0.57 | 57.3 | 0 | 1284 | 56 | 137.2 | 29.4 | self | self | 478 | 0 |
| 2OA26O64HB | FittedShirt | 3221 | 6144 | 0.07 | 0.19 | 0.43 | 0.50 | 2.4 | 0.41 | 0.04 | 3.9 | 0 | 454 | 61 | 8.9 | 3.5 | pass | pass | 0 | 0 |
| 392XDT8845 | FittedShirt/StraightWB/PencilSkirt | 20080 | 39786 | 1.17 | 1.39 | 2.40 | 1.42 | 30.9 | 0.67 | 0.21 | 37.0 | 0 | 409 | 51 | 397.6 | 16.7 | pass | pass | 0 | 6 |
| 3NCN2ULC5M | Shirt/SkirtCircle | 14646 | 28995 | 1.65 | 1.12 | 1.82 | 1.04 | 56.4 | 0.59 | 0.68 | 61.7 | 60.5 | 2400 | 66 | 161.3 | 15.7 | static | pass | 187 | 0 |
| 4033EWPHPT | SkirtLevels | 7465 | 14555 | 0.06 | 0.36 | 1.26 | 0.68 | 6.0 | 0.66 | 0.09 | 9.1 | 0 | 551 | 37 | 30.2 | 7.7 | pass | pass | 20 | 0 |
| 5M54JCRT4R | FittedShirt/Pants | 0 | 0 | 0.83 | 0.25 | - | - | - | - | - | 0.3 | 0.3 | - | - | - | - | degen | degen | - | - |
| 6C7A6KMHCX | Shirt/FittedWB | 3091 | 5731 | 0.04 | 0.16 | 1.91 | 0.45 | 14.4 | 0.52 | 0.15 | 17.5 | 16.9 | 2400 | 50 | 13.7 | 5.2 | static | pass | 20 | 0 |
| 6IUF78BHK3 | Shirt/StraightWB | 8030 | 15768 | 0.05 | 0.51 | 0.85 | 0.65 | 10.7 | 0.46 | 0.12 | 13.3 | 0 | 802 | 88 | 23.7 | 7.2 | pass | pass | 62 | 0 |
| 7QH57VM6H8 | Shirt/PencilSkirt | 17580 | 34682 | 1.19 | 1.34 | 2.23 | 1.08 | 39.2 | 0.67 | 0.58 | 45.1 | 0 | 1596 | 32 | 188.1 | 20.3 | self | self | 438 | 0 |
| 8G1H25WK7T | SkirtManyPanels | 3016 | 5733 | 0.03 | 0.15 | 0.39 | 0.30 | 2.8 | 0.39 | 0.05 | 4.1 | 0 | 727 | 0 | - | 3.8 | pass | pass | 0 | 0 |
| 9UGKNR53BO | FittedShirt/Skirt2 | 3815 | 7418 | 0.05 | 0.18 | 0.69 | 0.47 | 5.3 | 0.42 | 0.08 | 7.2 | 0 | 580 | 395 | 6.5 | 4.4 | pass | pass | 13 | 0 |
| A8Y94BNOEV | FittedShirt | 7857 | 15478 | 0.96 | 0.50 | 1.06 | 0.71 | 16.1 | 0.48 | 0.09 | 18.9 | 0 | 450 | 58 | 197.8 | 9.7 | body | body | 0 | 41 |
| BIU2ZP9AZT | FittedWB/PencilSkirt | 15000 | 29701 | 0.06 | 0.92 | 1.86 | 0.96 | 8.1 | 0.57 | 0.17 | 12.6 | 0 | 552 | 31 | 31.0 | 10.6 | pass | pass | 0 | 0 |
| CBM3R65ITQ | FittedShirt/FittedWB/SkirtCircle | 35169 | 69706 | 0.08 | 2.34 | 2.07 | 2.02 | 52.3 | 1.13 | 0.98 | 60.8 | 0 | 1496 | 27 | 107.4 | 33.9 | pass | pass | 6 | 0 |
| CJGO789TMC | FittedShirt/FittedWB/Pants | 15342 | 30482 | 0.21 | 1.01 | 2.47 | 1.07 | 12.0 | 0.67 | 0.17 | 17.4 | 0 | 422 | 43 | 48.1 | 12.3 | pass | pass | 0 | 0 |
| FQF1EOSAQA | FittedShirt/FittedWB | 11444 | 22554 | 1.11 | 0.77 | 2.18 | 0.85 | 37.3 | 0.61 | 0.23 | 42.0 | 0 | 978 | 58 | 239.5 | 11.1 | pass | pass | 160 | 0 |
| FR9TCUJQIU | StraightWB/Skirt2 | 2684 | 5138 | 0.01 | 0.12 | 0.67 | 0.29 | 2.3 | 0.41 | 0.03 | 3.8 | 0 | 438 | 25 | 14.6 | 4.1 | pass | pass | 0 | 0 |
| G6E4BSAGSE | FittedShirt/StraightWB/SkirtManyPanels | 24298 | 48012 | 0.10 | 1.57 | 1.16 | 1.60 | 20.7 | 0.72 | 0.39 | 26.1 | 0 | 783 | 28 | 129.8 | 21.3 | pass | pass | 0 | 0 |
| HK2NDP64LK | FittedShirt/FittedWB/AsymmSkirtCircle | 20582 | 40585 | 0.11 | 1.32 | 2.87 | 1.19 | 17.3 | 0.71 | 0.33 | 23.7 | 0 | 726 | 35 | 72.5 | 19.3 | pass | pass | 0 | 0 |
| J9616TU8PH | FittedShirt/Pants | 12821 | 25316 | 0.93 | 0.83 | 2.26 | 1.01 | 23.0 | 0.65 | 0.14 | 27.9 | 0 | 408 | 48 | 287.0 | 12.5 | pass | pass | 0 | 26 |
| K6AAOMDSH2 | Shirt/StraightWB/Pants | 17579 | 34668 | 0.14 | 1.05 | 2.54 | 1.16 | 57.4 | 0.70 | 0.75 | 63.6 | 48.5 | 2400 | 27 | 109.2 | 21.7 | body+static | body | 42 | 87 |
| KE5F58MAUX | Shirt/Skirt2 | 0 | 0 | 0.05 | 0.44 | - | - | - | - | - | 0.4 | 0.4 | - | - | - | - | degen | degen | - | - |
| MYH6HZAQSE | Shirt/StraightWB | 5917 | 11472 | 1.01 | 0.51 | 1.93 | 0.69 | 30.9 | 0.62 | 0.25 | 34.9 | 34.9 | 2400 | 57 | 130.3 | 9.9 | body+static | body+static | 20 | 46 |
| N4F8XJHUWI | PencilSkirt | 9484 | 18675 | 0.03 | 0.54 | 0.77 | 0.81 | 5.3 | 0.48 | 0.12 | 8.0 | 0 | 534 | 30 | 21.1 | 7.2 | pass | pass | 0 | 0 |
| N8BM7C5KG8 | Shirt | 13714 | 27118 | 0.05 | 0.87 | 2.38 | 1.03 | 9.5 | 0.66 | 0.18 | 14.7 | 0 | 597 | 47 | 31.4 | 10.0 | pass | pass | 0 | 0 |
| PPIG5B12I5 | Shirt/AsymmSkirtCircle | 10243 | 20048 | 0.98 | 0.78 | 1.81 | 0.76 | 24.2 | 0.57 | 0.15 | 28.3 | 0 | 795 | 123 | 14.2 | 9.2 | pass | pass | 25 | 0 |
| QLKG5XNE5M | Shirt/StraightWB | 12840 | 25476 | 1.06 | 0.87 | 1.33 | 0.91 | 63.3 | 0.54 | 0.62 | 67.6 | 67.5 | 2400 | 47 | 302.0 | 19.9 | body+self+static | body+self+static | 800 | 50 |
| RW9T7P4LZK | FittedShirt/StraightWB | 4095 | 7926 | 0.08 | 0.24 | 0.56 | 0.47 | 3.8 | 0.41 | 0.05 | 5.6 | 0 | 414 | 49 | 25.0 | 5.5 | pass | pass | 0 | 0 |
| S1AJSFQ2X9 | FittedShirt | 3853 | 7434 | 0.08 | 0.19 | 0.61 | 0.47 | 3.1 | 0.40 | 0.05 | 4.8 | 0 | 418 | 61 | 13.3 | 4.0 | pass | pass | 0 | 0 |
| S9K5OZYN43 | SkirtCircle | 5545 | 10843 | 0.02 | 0.23 | 0.85 | 0.53 | 2.4 | 0.43 | 0.06 | 4.5 | 0 | 406 | 11 | 10.4 | 5.3 | pass | pass | 0 | 0 |
| SRCXHLDZLT | Shirt | 16333 | 32196 | 0.04 | 1.12 | 1.62 | 1.00 | 9.7 | 0.61 | 0.17 | 14.3 | 0 | 465 | 43 | 36.6 | 12.6 | pass | pass | 13 | 0 |
| ULW7Q90FG5 | FittedWB/Pants | 8257 | 16308 | 0.13 | 0.53 | 0.79 | 0.72 | 5.6 | 0.46 | 0.09 | 8.2 | 0 | 414 | 32 | 43.2 | 6.9 | pass | pass | 0 | 0 |
| WLSZYSN4O8 | FittedShirt/Pants | 18515 | 36818 | 1.16 | 1.17 | 2.43 | 1.23 | 32.4 | 0.66 | 0.19 | 38.1 | 0 | 406 | 56 | 337.0 | 13.8 | pass | pass | 0 | 0 |
| X2XEEDEBY7 | StraightWB/GodetSkirt | 13224 | 26152 | 0.06 | 0.90 | 2.19 | 0.94 | 11.8 | 0.62 | 0.17 | 16.6 | 0 | 595 | 27 | 112.3 | 11.6 | pass | pass | 0 | 0 |
| X5VKWSRC0Q | Shirt/StraightWB/SkirtLevels | 18370 | 36168 | 0.08 | 1.16 | 1.85 | 1.26 | 17.8 | 0.87 | 0.27 | 23.2 | 0 | 701 | 44 | 103.4 | 17.1 | pass | pass | 115 | 0 |
| XZW6XEZXJN | Shirt/Skirt2 | 0 | 0 | 0.05 | 0.09 | - | - | - | - | - | 0.1 | 0.1 | - | - | - | - | degen | degen | - | - |
| YYAKPNDJT4 | Shirt/StraightWB/PencilSkirt | 11750 | 23192 | 0.09 | 0.68 | 1.51 | 0.97 | 12.9 | 0.55 | 0.18 | 16.8 | 0 | 774 | 42 | 51.0 | 10.4 | pass | pass | 27 | 0 |

## Stability of flags (steps: main run + 2 reruns of every flagged garment)

Each row is the **final attempt** of that run (after GarmentCode's own resim if it did one); counts are of that final attempt. The 3 meshing failures are deterministic (identical failure in the numpy 1.26.4 / python 3.10 reference venv).

| garment | design | run | self int | body int | frames | attempt events (final attempt) | final verdict |
|---|---|---|---|---|---|---|---|
| K6AAOMDSH2 | Shirt/Pants | main (resim) | 51 | 87 | 1708 | body | FLAG |
| K6AAOMDSH2 | Shirt/Pants | rerun1 | 147 | 88 | 1474 | body | FLAG |
| K6AAOMDSH2 | Shirt/Pants | rerun2 | 151 | 87 | 1519 | body | FLAG |
| 1N6OORV4FV | Shirt/GodetSkirt | main | 478 | 0 | 1284 | self | FLAG |
| 1N6OORV4FV | Shirt/GodetSkirt | rerun1 (resim) | 130 | 0 | 1619 | pass | pass |
| 1N6OORV4FV | Shirt/GodetSkirt | rerun2 (resim) | 376 | 0 | 2400 | static+self | FLAG |
| A8Y94BNOEV | FittedShirt | main | 0 | 41 | 450 | body | FLAG |
| A8Y94BNOEV | FittedShirt | rerun1 | 0 | 40 | 887 | body | FLAG |
| A8Y94BNOEV | FittedShirt | rerun2 | 0 | 41 | 422 | body | FLAG |
| 6C7A6KMHCX | Shirt | main (resim) | 45 | 0 | 2267 | pass | pass |
| 6C7A6KMHCX | Shirt | rerun1 | 63 | 0 | 2025 | pass | pass |
| 6C7A6KMHCX | Shirt | rerun2 (resim) | 47 | 0 | 2400 | static | FLAG |
| 3NCN2ULC5M | Shirt/SkirtCircle | main (resim) | 218 | 0 | 2273 | pass | pass |
| 3NCN2ULC5M | Shirt/SkirtCircle | rerun1 | 174 | 0 | 807 | pass | pass |
| 3NCN2ULC5M | Shirt/SkirtCircle | rerun2 | 302 | 0 | 1158 | self | FLAG |
| QLKG5XNE5M | Shirt | main (resim) | 625 | 50 | 2400 | static+body+self | FLAG |
| QLKG5XNE5M | Shirt | rerun1 (resim) | 901 | 50 | 2400 | static+body+self | FLAG |
| QLKG5XNE5M | Shirt | rerun2 (resim) | 569 | 50 | 2400 | static+body+self | FLAG |
| 7QH57VM6H8 | Shirt/PencilSkirt | main | 438 | 0 | 1596 | self | FLAG |
| 7QH57VM6H8 | Shirt/PencilSkirt | rerun1 (resim) | 412 | 0 | 2297 | self | FLAG |
| 7QH57VM6H8 | Shirt/PencilSkirt | rerun2 (resim) | 447 | 0 | 2400 | static+self | FLAG |
| MYH6HZAQSE | Shirt | main (resim) | 189 | 46 | 2400 | static+body | FLAG |
| MYH6HZAQSE | Shirt | rerun1 | 7 | 46 | 585 | body | FLAG |
| MYH6HZAQSE | Shirt | rerun2 | 0 | 46 | 949 | body | FLAG |

Flagged in 3 runs / flagged in N runs (final attempt): K6AAOMDSH2 3/3, 1N6OORV4FV 2/3, A8Y94BNOEV 3/3, 6C7A6KMHCX 1/3, 3NCN2ULC5M 1/3, QLKG5XNE5M 3/3, 7QH57VM6H8 3/3, MYH6HZAQSE 3/3

Result: of the 6 garments flagged by simulation in the main run, **5 are stable** (K6AAOMDSH2, A8Y94BNOEV, QLKG5XNE5M, 7QH57VM6H8, MYH6HZAQSE flag in 3/3 runs) and **1 is marginal** (1N6OORV4FV, 2/3: self-intersections 478, 130, 376 around the 300 threshold). The 2 garments GarmentCode rescued by resim (6C7A6KMHCX, 3NCN2ULC5M) are both **marginal**: each is flagged in 1 of 3 runs. Body-intersection flags are stable because the count is nearly identical in every run (87/88/87, 41/40/41, 50, 46), so it is a property of the design and initial placement, not chaos; A8Y94BNOEV (about 41 vs threshold 35) is stable but close to the limit. `static_equilibrium` is chaotic: the same garment hits the 2400-frame cap in one run and settles in 600 to 1000 frames in another. Self-intersection counts vary by up to 2x run to run (QLKG5XNE5M 569 to 901) but stay on one side of 300 except for 1N6OORV4FV and 3NCN2ULC5M.

## Visual inspection

Contact sheets (front and back of 8 garments per sheet, JPEG, labelled with id, design, final flag and the first-attempt flag where it differs):
`port-docs/batch/main_01.jpg`, `main_02.jpg`, `main_03.jpg`, `main_04.jpg`, `main_05.jpg`. I looked at all five. (No sheets were made for the reruns; their renders are overwritten by resims and I judged flags from counts.)

| garment | what I see | GarmentCode verdict | agree? |
|---|---|---|---|
| K6AAOMDSH2 (Shirt/StraightWB/Pants) | shirt flapping/exploded at the shoulders, very large pants | FLAG (body) | yes |
| MYH6HZAQSE (Shirt) | top has fallen to the waist, one shoulder off | FLAG (body, static) | yes |
| 6C7A6KMHCX (Shirt/FittedWB) | crumpled collapsed garment around the waist with floating wings | **PASS** (first attempt was static; resim passed numeric checks) | **no, visually wrong yet passing**; flagged static again in rerun2 |
| 3NCN2ULC5M (Shirt/SkirtCircle) | hoodie visible, skirt not visible | PASS after resim | see below, explained (design, not sim fault) |
| CBM3R65ITQ | back of the strapless top is bunched and crumpled | pass | questionable |
| QLKG5XNE5M | plausible on the body | FLAG (body+self+static) | flag is numeric only, nothing visibly broken to me |
| hoods 02MWDQWIU6, 3NCN2ULC5M, 7QH57VM6H8, PPIG... | hood drapes over head/face | pass/flag as listed | probably design, not an error |
| SRCXHLDZLT | open-sided tunic | pass | probably the design |
| all other garments | plausible and on the body | pass | yes |
| 5M54JCRT4R, XZW6XEZXJN, KE5F58MAUX | no render (meshing failure) | FLAG (degen) | n/a |

So GarmentCode's flags agree with my reading for 2 of the 3 clearly broken garments; its numeric checks (intersection counts, static check) miss a collapsed-but-not-intersecting garment. Other flags (QLKG5XNE5M, 1N6OORV4FV, 7QH57VM6H8, A8Y94BNOEV) are numeric threshold flags where the render did not look obviously wrong to me; I looked at fixed-size sheet thumbnails, not at inspection-quality zooms.

**3NCN2ULC5M skirt question (resolved).** Its specification does contain skirt panels (`skirt_front`, `skirt_back`), but the design has a negative flare-skirt length (-0.115): the panels are only **10.2 cm and 6.3 cm tall**, starting at y = 89.6 cm, while the hoodie torso panels (50 cm tall, starting at y = 91.4 cm) and shirt length 1.2 cover that region. The skirt is a short stub hidden under the long hoodie. The garment is a degenerate design, not a simulation fault.

## Meshing failures (degenerate triangles)

5M54JCRT4R (FittedShirt/Pants), XZW6XEZXJN (Shirt/Skirt2), KE5F58MAUX (Shirt/Skirt2). GarmentCode flags `degenerate_triangles`, resimulates, and fails again. I re-ran the meshing step of these designs in the reference venv (`../venv-ref`: numpy 1.26.4, python 3.10) and they fail identically, so this is platform-independent and design-driven, not a GPU, ROCm or numpy-2 problem.

## What dominates the slow and most-flagged garments

The 3 slowest (first attempt): QLKG5XNE5M 67.6 s, K6AAOMDSH2 63.6 s, 3NCN2ULC5M 61.7 s (then CBM3R65ITQ 60.8 s, 1N6OORV4FV 57.3 s). The 3 most-flagged: QLKG5XNE5M (body+self+static in all 3 runs, both attempts), K6AAOMDSH2 and 7QH57VM6H8 / MYH6HZAQSE (flagged 3/3). I profiled QLKG5XNE5M, K6AAOMDSH2, 3NCN2ULC5M, MYH6HZAQSE and 1N6OORV4FV with `gpu_profile.py` (inputs built with `dump_harness.py prepare`, same specification). **K6AAOMDSH2's profile input used body `mean_all` instead of `mean_all_apart`** (the Pants design's body), so its numbers are indicative only. Profiles run 1200 frames in graph mode (as GarmentCode) and 300 frames in kernels mode (eager, GPU kernel timing per frame); static-stop disabled.

| garment | particles | springs | graph: frame0 ms | graph: median ms fr 1-39 | graph: median ms fr 40+ | graph: median last 400 fr | kernels: GPU ms/fr, fr 1-39 | kernels: GPU ms/fr, fr 40-299 | create_self_edge_contacts share fr 1-39 | share fr 40-299 | next kernel (share fr 40-299) | wall/GPU ratio fr 40+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QLKG5XNE5M | 12840 | 38318 | 3451 | 332.0 | 19.3 | 19.6 | 329 | 16.7 | 98% | 63% | reference_shape_drag_constraint 11% | 1.1x |
| K6AAOMDSH2 | 17579 | 52250 | 1829 | 115.1 | 21.5 | 21.6 | 109 | 19.7 | 93% | 54% | reference_shape_drag_constraint 9% | 1.1x |
| 3NCN2ULC5M | 14646 | 43643 | 2749 | 235.8 | 16.6 | 16.1 | 232 | 16.3 | 97% | 80% | solve_edge_edge_self_contact 4% | 1.1x |
| MYH6HZAQSE | 5917 | 17391 | 605 | 141.0 | 9.2 | 9.3 | 139 | 7.9 | 96% | 64% | reference_shape_drag_constraint 16% | 1.1x |
| 1N6OORV4FV | 22067 | 65879 | 3452 | 187.1 | 29.1 | 29.6 | 181 | 24.7 | 95% | 76% | solve_edge_edge_self_contact 6% | 1.1x |

Reading:
- **Why the slow ones are slow: they never reach static equilibrium, so they run the 2400-frame cap** (QLKG5XNE5M, K6AAOMDSH2 first attempt, 3NCN2ULC5M first attempt, MYH6HZAQSE first attempt) at 10 to 22 ms/frame; the steady state is where most of the time goes (e.g. QLKG5XNE5M: 2400 frames x about 20 ms, steady sum 52 s of 67 s). CBM3R65ITQ and 1N6OORV4FV are slow mainly from size (35k and 22k vertices, 29 to 34 ms/frame); ms/frame correlates with vertex count (r = 0.93 over the 37).
- **The early frames are dominated by `create_self_edge_contacts`** (93 to 98 % of GPU time in frames 1 to 39, 109 to 332 ms/frame GPU), reproducing the ws8 finding on 5 more designs.
- **Correction to the earlier model of steady state:** in frames 40 to 299 `create_self_edge_contacts` is still the largest kernel on these garments (54 to 80 % of GPU time; next kernel 4 to 16 %), and wall time is only about 1.1x the GPU kernel time. So for these large or long-spring garments the steady state is GPU-bound in that kernel, not launch/latency-bound as on the small test garments in gpu_validation.md. This is measured over frames 40 to 299 in eager (kernels) mode; the graph-mode medians (19 to 29 ms) match the batch's steady numbers.
- **Why they are flagged**: QLKG5XNE5M, K6AAOMDSH2 and MYH6HZAQSE carry body intersections of 50, 87 and 46 that are constant across runs (design/placement-driven, not chaos); QLKG5XNE5M and 1N6OORV4FV/7QH57VM6H8 also have self-intersection counts of 400 to 900 (near or well over 300) which vary 2x run to run. Static-equilibrium flags come from the cap being hit in some runs (chaotic).

## Findings and fixes

- Fixes in GarmentCode for the batch path: **none beyond the existing, uncommitted `data_config.py` Warp-1.13 patch from earlier workstreams** (I did not need to touch it again); no other Warp-1.13 incompatibility showed up in `pattern_sampler`/`pattern_data_sim`/`datasim_utils`. Only change for this workstream: `system.json` (above). No packages installed. No Warp, sim numerics or sim parameters changed.
- New files only: `port-docs/scripts/batch_sample.py`, `batch_sim_timed.py`, `batch_analyze.py`, `batch_sheets.py`, `batch_report.py`, `batch_profile_summary.py`; `port-docs/batch/main_0[1-5].jpg`; this document. Outputs are under `../batch_out/`.

## Measured / inferred / not verified

**Measured**: every time, frame count, settle frame, vertex/face count and intersection count in the tables; first-attempt and final verdicts; resim costs; cold compile time; flag stability from 3 runs; kernel shares from gpu_profile on 5 garments (3 trusted fully, K6AAOMDSH2 with the wrong body); the 3 meshing failures failing identically in the numpy-1 venv; the 3NCN2ULC5M panel sizes.

**Inferred**: "steady-state is GPU-bound on large garments" generalises beyond the five profiled; that the body-intersection counts are design/placement-driven (they are nearly constant across runs, but I did not trace where the penetration is); the visual verdicts on hoods and tunic being design; typical pass rate for the population beyond this one 40-garment sample.

**Not verified**: that this sample resembles the GTX 1070 sample (different seed and sampler path, one body only); the graph-mode kernel breakdown (kernel timing is eager); behaviour on non-mean bodies (dataset not installed); whether 6C7A6KMHCX's collapse is a sim fault or its design (not examined); K6AAOMDSH2's profile on the correct body; the settle-frame heuristic is mine, not GarmentCode's. All runs are single runs of a chaotic simulation; see the stability table.
